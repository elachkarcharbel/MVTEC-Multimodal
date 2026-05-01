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
        
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label


train_folder = '../MVTEC-AD-WAV/train'
train_csv = '../binary_labels/new_train_audio.csv'
test_folder = '../MVTEC-AD-WAV/test'
test_csv = '../binary_labels/new_test_audio.csv'

train_dataset = CustomAudioDataset(train_folder=train_folder, test_folder=test_folder, csv_file=train_csv)
test_dataset = CustomAudioDataset(train_folder=train_folder, test_folder=test_folder, csv_file=test_csv)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_dataset.data.iloc[:, 1].values)
class_weights = torch.tensor(class_weights, dtype=torch.float)

class WaveRNN(nn.Module):
    def __init__(self, dim=512, rnn_type='GRU', num_classes=2):
        super(WaveRNN, self).__init__()
        self.dim = dim
        self.rnn_type = rnn_type
        self.num_classes = num_classes

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=1, hidden_size=dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=1, hidden_size=dim, batch_first=True)

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Ensure the input tensor has three dimensions
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.transpose(1, 2)  # Transpose to match (batch, seq_len, 1)
        #print(x.size())  # Debugging print to check the size of x
        batch_size, seq_len, _ = x.size()
        x, _ = self.rnn(x)
        x = x.contiguous().view(batch_size, seq_len, self.dim)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class WaveRNNModel(LightningModule):
    def __init__(self):
        super(WaveRNNModel, self).__init__()
        self.model = WaveRNN()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        waveforms, labels = batch
        outputs = self(waveforms)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def validation_step(self, batch, batch_idx):
        waveforms, labels = batch
        outputs = self(waveforms)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        return {"val_loss": loss, "preds": preds, "labels": labels}

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

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=True,
    mode='min'
)

model = WaveRNNModel()
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
