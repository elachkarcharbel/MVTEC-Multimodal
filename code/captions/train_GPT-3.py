import openai
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

openai.api_key = 'your_openai_api_key'

class GPT3Classifier:
    def __init__(self, model='text-davinci-003'):
        self.model = model

    def predict(self, prompt):
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            top_p=1.0,
            n=1,
            logprobs=5
        )
        return response.choices[0].text.strip()

def create_prompt(caption):
    return f"Classify the following caption as 0 (normal) or 1 (abnormal):\n\n{caption}\n\nLabel:"

data = pd.read_csv('path/to/train.csv')
captions = data['caption'].tolist()
labels = data['label'].tolist()

train_captions, test_captions, train_labels, test_labels = train_test_split(captions, labels, test_size=0.2, random_state=42)

gpt3_model = GPT3Classifier()

train_predictions = [int(gpt3_model.predict(create_prompt(caption))) for caption in train_captions]
test_predictions = [int(gpt3_model.predict(create_prompt(caption))) for caption in test_captions]

accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions)
recall = recall_score(test_labels, test_predictions)
f1 = f1_score(test_labels, test_predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
