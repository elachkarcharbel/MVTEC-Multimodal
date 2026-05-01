# Import necessary libraries
import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from google.protobuf import descriptor as _descriptor
from google.protobuf.internal import api_implementation
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, folder, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        # Remove 'train/' or 'test/' prefix if it's present
        img_name = img_name.replace('train/', '').replace('test/', '')
        img_path = f"{self.folder}/{img_name}"
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the transformation including resizing
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to the new training and testing folders
new_train_folder = '../MVTEC-AD-Original-Normalized700x700/new_train'
new_test_folder = '../MVTEC-AD-Original-Normalized700x700/new_test'
new_train_csv = '../binary_labels/new_train.csv'
new_test_csv = '../binary_labels/new_test.csv'
max_epochs = 200

train_dataset = CustomDataset(folder=new_train_folder, csv_file=new_train_csv, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = CustomDataset(folder=new_test_folder, csv_file=new_test_csv, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_dataset.data.iloc[:, 1].values)
class_weights = torch.tensor(class_weights, dtype=torch.float)

class ViTModel(LightningModule):
    def __init__(self):
        super(ViTModel, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b7_ap', pretrained=True, num_classes=2)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
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

# Create the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=True,
    mode='min'
)

# Initialize the model and trainer
model = ViTModel()
#trainer = Trainer(callbacks=[early_stopping], max_epochs=max_epochs, devices=1 if torch.cuda.is_available() else "auto")
trainer = Trainer(callbacks=[early_stopping], max_epochs=max_epochs, accelerator='gpu', devices=1 if torch.cuda.is_available() else 'auto')

# Train the model
trainer.fit(model, train_loader, test_loader)

# Evaluate the model
def evaluate_model(model, dataloader, device):
    model.to(device)  # Move the model to the specified device
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the specified device
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    return accuracy, precision, recall, f1

# Calculate metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
