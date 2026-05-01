import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os

class CustomAudioDataset(Dataset):
    def __init__(self, train_folder, test_folder, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        audio_name = audio_name.replace("train/", "").replace("test/", "")

        train_audio_path = os.path.join(self.train_folder, audio_name)
        test_audio_path = os.path.join(self.test_folder, audio_name)

        if os.path.exists(train_audio_path):
            audio_path = train_audio_path
        elif os.path.exists(test_audio_path):
            audio_path = test_audio_path
        else:
            raise FileNotFoundError(f"Audio file {audio_name} not found in either train or test directory.")

        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.size(1) % 16 != 0:
            return None

        if self.transform:
            waveform = self.transform(waveform)
        return waveform, torch.tensor(label)


class PadOrTrim:
    def __init__(self, frame_size):
        self.frame_size = frame_size

    def __call__(self, waveform):
        length = waveform.size(1)
        if length % self.frame_size != 0:
            pad_length = self.frame_size - (length % self.frame_size)
            padded_waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            return padded_waveform
        return waveform

train_folder = '../MVTEC-AD-WAV/train'
train_csv = '../binary_labels/new_train_audio.csv'
test_folder = '../MVTEC-AD-WAV/test'
test_csv = '../binary_labels/new_test_audio.csv'

frame_size = 64
transform = PadOrTrim(frame_size=frame_size)

train_dataset = CustomAudioDataset(train_folder=train_folder, test_folder=test_folder, csv_file=train_csv, transform=transform)
test_dataset = CustomAudioDataset(train_folder=train_folder, test_folder=test_folder, csv_file=test_csv, transform=transform)

train_dataset = [item for item in train_dataset if item is not None]
test_dataset = [item for item in test_dataset if item is not None]

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Check the unique labels in the train dataset
all_labels = [item[1] for item in train_dataset]
unique_labels = set(all_labels)
print(f"Unique labels in train dataset: {unique_labels}")

# Ensure that the `classes` parameter includes all unique labels
class_weights = compute_class_weight('balanced', classes=list(unique_labels), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

class SampleRNN(nn.Module):
    def __init__(self, frame_size=64, seq_len=1024, dim=256, classes=2):
        super(SampleRNN, self).__init__()
        self.frame_size = frame_size
        self.seq_len = seq_len
        self.dim = dim
        self.classes = classes

        self.rnn1 = nn.GRU(input_size=frame_size, hidden_size=dim, num_layers=1, batch_first=True)
        self.rnn2 = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(seq_len, dim)
        self.fc2 = nn.Linear(dim, classes)

    def forward(self, x):
        batch_size = x.size(0)
        input_length = x.size(1)

        if input_length % self.frame_size != 0:
            raise ValueError(f"Input length {input_length} is not divisible by frame size {self.frame_size}")

        seq_len = input_length // self.frame_size

        if seq_len == 0:
            raise ValueError("Sequence length is zero. Adjust the frame_size or input length.")

        x = x.view(batch_size, seq_len, self.frame_size)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x.contiguous().view(batch_size, -1, self.dim)
        x = torch.relu(self.fc1(x))
        x = x.mean(dim=1)
        x = self.fc2(x)
        return x

class SampleRNNModel(LightningModule):
    def __init__(self, frame_size=64):
        super(SampleRNNModel, self).__init__()
        self.frame_size = frame_size
        self.model = SampleRNN(frame_size=frame_size, classes=2)  # Output classes for classification
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, x):
        input_length = x.shape[-1]
        
        # Apply zero-padding if needed
        if input_length % self.frame_size != 0:
            padding_size = self.frame_size - (input_length % self.frame_size)
            x = torch.nn.functional.pad(x, (0, padding_size), 'constant', 0)
        
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        waveforms, targets = batch

        # Check shapes of waveforms and targets
        print(f"Waveforms shape: {waveforms.shape}")
        print(f"Targets shape: {targets.shape}")

        # Ensure targets are single class labels
        targets = targets.view(-1)

        # Compute loss
        outputs = self(waveforms)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, targets = batch
        
        # Apply zero-padding to waveforms if needed
        input_length = waveforms.shape[-1]
        if input_length % self.frame_size != 0:
            padding_size = self.frame_size - (input_length % self.frame_size)
            waveforms = torch.nn.functional.pad(waveforms, (0, padding_size), 'constant', 0)
        
        targets = targets.view(-1)

        outputs = self(waveforms)
        loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        return {"val_loss": loss, "preds": preds, "labels": targets}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        precision = precision_score(labels.cpu(), preds.cpu())
        recall = recall_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu())
        self.log('val_loss', val_loss)
        self.log('val_accuracy', accuracy)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')

model = SampleRNNModel()
trainer = Trainer(callbacks=[early_stopping], max_epochs=200, accelerator='gpu', devices=1 if torch.cuda.is_available() else 'auto')

trainer.fit(model, train_loader, test_loader)

def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    return accuracy, precision, recall, f1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
