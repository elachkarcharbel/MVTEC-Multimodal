import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import AzureOpenAI

api_key = 'b2b2b4f3fca240409bce99def523d41e'
endpoint = 'https://ex-openaigpt4.openai.azure.com/'
api_version = '2024-05-01-preview'
deployment_name = 'gpt-4o-mini'

def safe_api_call(api_func, *args, **kwargs):
    while True:
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            print(f"API limit reached or error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)

class AzureGPTClassifier:
    def __init__(self, endpoint, api_version, deployment_name, api_key):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name

    def predict(self, prompt):
        response = safe_api_call(self.client.chat.completions.create,
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            top_p=1.0,
            n=1,
            logprobs=True
        )
        
        result = response.choices[0].message.content.strip()
        print(result)  # Debugging line
        
        try:
            return int(result)
        except ValueError:
            print(f"Unexpected: {result}")  # Debugging line
            return None

def create_prompt(caption):
    return f"Classify the following caption as 0 (normal) or 1 (abnormal):\n\n{caption}\n\nLabel:"

def train_and_evaluate(train_csv, test_csv):
    data = pd.read_csv(train_csv)
    captions = data['caption'].tolist()
    labels = data['label'].tolist()

    train_captions, test_captions, train_labels, test_labels = train_test_split(captions, labels, test_size=0.2, random_state=42)

    azure_model = AzureGPTClassifier(endpoint, api_version, deployment_name, api_key)

    train_predictions = []
    for caption in train_captions:
        train_predictions.append(azure_model.predict(create_prompt(caption)))
        time.sleep(1)  # Adding delay between requests
    
    train_predictions = [pred for pred in train_predictions if pred is not None]
    
    test_predictions = []
    for caption in test_captions:
        prediction = azure_model.predict(create_prompt(caption))
        if prediction is not None:
            test_predictions.append(prediction)
        time.sleep(1)  # Adding delay between requests
    
    while len(test_predictions) < len(test_labels):
        test_predictions.append(0)  # Adding a default prediction (0)

    print(f"Number of test labels: {len(test_labels)}")
    print(f"Number of test predictions: {len(test_predictions)}")

    if len(test_predictions) == len(test_labels):
        accuracy = accuracy_score(test_labels, test_predictions)
        precision = precision_score(test_labels, test_predictions)
        recall = recall_score(test_labels, test_predictions)
        f1 = f1_score(test_labels, test_predictions)
        
        print(f"Folder: {os.path.dirname(train_csv)}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
    else:
        print("Error: The number of predictions does not match the number of labels!")

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
    train_and_evaluate(train_csv, test_csv)
