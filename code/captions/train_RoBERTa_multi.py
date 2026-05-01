import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class CaptionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'caption': caption,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary', zero_division=1)
    recall = recall_score(labels, preds, average='binary', zero_division=1)
    f1 = f1_score(labels, preds, average='binary', zero_division=1)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

class CustomLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Suppress the detailed logging by not printing the evaluation results
        pass

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

logging_steps = 107200  # Assuming batch size of 8

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=200,  # Increase epochs to 200
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=logging_steps,
    evaluation_strategy='epoch',
    save_strategy='epoch',  # Save the model every epoch
    load_best_model_at_end=True,  # Load the best model at the end of training
    save_total_limit=1,  # Limit the number of saved models
    metric_for_best_model='eval_loss',  # Metric to compare for best model
    greater_is_better=False,  # Since lower eval_loss is better
    log_level='warning',  # Set log level to warning
    log_level_replica='warning',  # Set log level for replica to warning
    disable_tqdm=False  # Keep the progress bar enabled
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,  # Early stopping after 5 epochs without improvement
    early_stopping_threshold=0.0  # Threshold for improvement
)

custom_logging_callback = CustomLoggingCallback()

def train_and_evaluate(folder, train_csv, test_csv):
    print(f"Processing folder: {folder}")

    # Calculate class weights based on the label distribution in your training dataset
    train_df = pd.read_csv(train_csv)
    device = model.device
    class_counts = train_df['label'].value_counts().sort_index()
    class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float).to(device)

    # Manually set class weights for Roberta's classifier
    model.classifier.out_proj.weight.data *= class_weights.view(-1, 1)
    
    train_dataset = CaptionDataset(train_csv, tokenizer, max_len=128)
    test_dataset = CaptionDataset(test_csv, tokenizer, max_len=128)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback, custom_logging_callback]  # Add the custom logging callback
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"Accuracy: {results['eval_accuracy']:.2f}")
    print(f"Precision: {results['eval_precision']:.2f}")
    print(f"Recall: {results['eval_recall']:.2f}")
    print(f"F1 Score: {results['eval_f1']:.2f}")

folders = ['../binary_labels/BLIP/BLIP_VIT-B_CapFilt-L',
           '../binary_labels/BLIP/BLIP_VIT-L',
           '../binary_labels/CLIP/CLIP(Flickr30k)+GPT2-large',
           '../binary_labels/CLIP/CLIP(Flickr30k)+GPT2-small',
           '../binary_labels/CLIP/CLIP(RN50+COCO)+GPT2',
           '../binary_labels/CLIP/CLIP(RN50+Conceptual)+GPT2',
           '../binary_labels/CLIP/CLIP(RN50x4+COCO)+GPT2',
           '../binary_labels/CLIP/CLIP(RN50x4+Conceptual)+GPT2',
           '../binary_labels/CLIP/CLIP(RN50x16+COCO)+GPT2',
           '../binary_labels/CLIP/CLIP(RN50x16+Conceptual)+GPT2',
           '../binary_labels/CLIP/CLIP(RN50x64+COCO)+GPT2',
           '../binary_labels/CLIP/CLIP(RN50x64+Conceptual)+GPT2',
           '../binary_labels/CLIP/CLIP(RN101+COCO)+GPT2',
           '../binary_labels/CLIP/CLIP(RN101+Conceptual)+GPT2',
           '../binary_labels/CLIP/CLIP(ViT-B-16+COCO)+GPT2',
           '../binary_labels/CLIP/CLIP(ViT-B-16+Conceptual)+GPT2',
           '../binary_labels/CLIP/CLIP(ViT-B-32+COCO)+GPT2',
           '../binary_labels/CLIP/CLIP(ViT-B-32+Conceptual)+GPT2',
           '../binary_labels/CLIP/CLIP(ViT-L-14@336px+COCO)+GPT2',
           '../binary_labels/CLIP/CLIP(ViT-L-14@336px+Conceptual)+GPT2',
           '../binary_labels/CLIP/CLIP(ViT-L-14+COCO)+GPT2',
           '../binary_labels/CLIP/CLIP(ViT-L-14+Conceptual)+GPT2']

for folder in folders:
    train_csv = os.path.join(folder, 'new_train.csv')
    test_csv = os.path.join(folder, 'new_test.csv')
    train_and_evaluate(folder, train_csv, test_csv)
