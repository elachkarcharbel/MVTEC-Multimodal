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
import torch.nn.functional as F
import os
import numpy as np

# Custom Dataset
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

# Dataset Paths
train_folder = '../MVTEC-AD-WAV/train'
train_csv = '../binary_labels/new_train_audio.csv'
test_folder = '../MVTEC-AD-WAV/test'
test_csv = '../binary_labels/new_test_audio.csv'

# Dataset Loading
train_dataset = CustomAudioDataset(train_folder, test_folder, train_csv)
test_dataset = CustomAudioDataset(train_folder, test_folder, test_csv)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Class Weights for Imbalanced Data
class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_dataset.data.iloc[:, 1].values)
class_weights = torch.tensor(class_weights, dtype=torch.float)


# RawNet2 Model
class SincConv(nn.Module):
    """Sinc-based convolution layer from RawNet2"""
    def __init__(self, out_channels, kernel_size, sample_rate=44100):
        super(SincConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize filter banks
        self.low_hz = 30
        self.high_hz = sample_rate // 2 - 100
        self.band_hz = np.linspace(self.low_hz, self.high_hz, out_channels + 1)

        self.filters = nn.Parameter(torch.randn(out_channels, kernel_size), requires_grad=True)

    def forward(self, x):
        sinc_filters = torch.sin(self.filters) / self.filters
        sinc_filters = sinc_filters.to(x.device)
        sinc_filters = sinc_filters.unsqueeze(1)  # Shape: (out_channels, 1, kernel_size)
        x = F.conv1d(x, sinc_filters, stride=1, padding=self.kernel_size // 2)
        return x


class RawNet2(nn.Module):
    def __init__(self, num_classes=2, sample_rate=44100):
        super(RawNet2, self).__init__()

        # SincNet Layer
        self.sinc_conv = SincConv(out_channels=64, kernel_size=251, sample_rate=sample_rate)

        # Standard Conv Layers
        self.conv1 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        
        # GRU Layer
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # Fully Connected Layer
        self.fc = nn.Linear(128 * 2, num_classes)  # *2 for bidirectional GRU

    def forward(self, x):
        x = self.sinc_conv(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Prepare for GRU: (batch, time, features)
        x = x.permute(0, 2, 1)

        # GRU Processing
        x, _ = self.gru(x)

        # Global Average Pooling over time dimension
        x = torch.mean(x, dim=1)

        x = self.fc(x)
        return x


# PyTorch Lightning Model Wrapper
class RawNet2Model(LightningModule):
    def __init__(self):
        super(RawNet2Model, self).__init__()
        self.model = RawNet2(num_classes=2)
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
        return optim.Adam(self.parameters(), lr=0.001)

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


# Training Setup
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=True,
    mode='min'
)

model = RawNet2Model()
trainer = Trainer(callbacks=[early_stopping], max_epochs=200, accelerator='gpu', devices=1 if torch.cuda.is_available() else 'auto')

trainer.fit(model, train_loader, test_loader)


# Model Evaluation
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
