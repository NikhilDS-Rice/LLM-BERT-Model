
# Step 1: Install necessary libraries
!pip install transformers torch scikit-learn

# Step 2: Import libraries
import pandas as pd
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report
from transformers import get_scheduler
from sklearn.model_selection import train_test_split

!pip install datasets
import numpy as np


from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load the dataset (update the path to match your dataset location in Google Drive)
df = pd.read_csv('comments_labels1.csv')
# Display the first 5 rows of the dataset
print("First 5 rows of the dataset")
print(df.head())

# Display a random sample of 10 rows
print('\n10 Samples')
print(df.sample(10))

# Display summary statistics
print('\nDF - Dataset')
print(df.describe())

# Display data types and non-null counts
print(df.info())

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of labels
sns.countplot(x='pol', data=df)
plt.title('Label Distribution')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.show()

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Sample the dataset to reduce size for faster execution
sample_size = 10000  # Adjust sample size if needed
df_sample = df.sample(n=sample_size, random_state=42)

# Ensure 'likes' and 'replies' columns are numeric
df_sample['likes'] = pd.to_numeric(df_sample['likes'], errors='coerce').fillna(0).astype(int)
df_sample['replies'] = pd.to_numeric(df_sample['replies'], errors='coerce').fillna(0).astype(int)

# Re-encode labels to match CrossEntropyLoss requirements
df_sample['pol'] = df_sample['pol'].replace({-1: 0, 0: 1, 1: 2})

# Inspect the dataset
print(df_sample.head())
print(df_sample.info())
print(df_sample.shape)

# Define a custom PyTorch Dataset
class YouTubeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = str(self.data.iloc[index]['comment_text'])
        likes = self.data.iloc[index]['likes']
        replies = self.data.iloc[index]['replies']
        label = self.data.iloc[index]['pol']

        # Tokenize comment text
        encoding = self.tokenizer.encode_plus(
            comment,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt",
            return_attention_mask=True,
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'likes': torch.tensor(likes, dtype=torch.float),
            'replies': torch.tensor(replies, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long),
        }

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create a dataset and split into train/validation sets
dataset = YouTubeDataset(df_sample, tokenizer, max_len=128)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 classes: 0, 1, 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer, loss function, and scheduler
from transformers import AdamW
from transformers import get_scheduler

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_training_steps = len(train_loader) * 4  # Assuming 4 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

    print(f'Epoch {epoch} Training Loss: {train_loss / len(train_loader)}')

# Evaluation loop
model.eval()
val_loss = 0
predictions, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        val_loss += loss.item()

        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    print(f'Validation Loss: {val_loss / len(val_loader)}')

# Decode predictions back to original labels
decoded_predictions = [0 if pred == 1 else -1 if pred == 0 else 1 for pred in predictions]
decoded_true_labels = [0 if label == 1 else -1 if label == 0 else 1 for label in true_labels]

# Classification Report
print('Classification Report:')
print(classification_report(decoded_true_labels, decoded_predictions))

# Confusion Matrix
print('Confusion Matrix:')
cm = confusion_matrix(decoded_true_labels, decoded_predictions)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Load the dataset (update the path to match your dataset location in Google Drive)
df = pd.read_csv('/content/comments_labels1.csv')

# Sample the dataset to reduce size for faster execution
sample_size = len(df) # Adjust sample size if needed
df_sample = df.sample(n=sample_size, random_state=42)

# Ensure 'likes' and 'replies' columns are numeric
df_sample['likes'] = pd.to_numeric(df_sample['likes'], errors='coerce').fillna(0).astype(int)
df_sample['replies'] = pd.to_numeric(df_sample['replies'], errors='coerce').fillna(0).astype(int)

# Re-encode labels to match CrossEntropyLoss requirements
df_sample['pol'] = df_sample['pol'].replace({-1: 0, 0: 1, 1: 2})

# Inspect the dataset
print(df_sample.head())
print(df_sample.info())
print(df_sample.shape)

# Define a custom PyTorch Dataset
class YouTubeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = str(self.data.iloc[index]['comment_text'])
        likes = self.data.iloc[index]['likes']
        replies = self.data.iloc[index]['replies']
        label = self.data.iloc[index]['pol']

        # Tokenize comment text
        encoding = self.tokenizer.encode_plus(
            comment,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt",
            return_attention_mask=True,
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'likes': torch.tensor(likes, dtype=torch.float),
            'replies': torch.tensor(replies, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long),
        }

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create a dataset and split into train/validation sets
dataset = YouTubeDataset(df_sample, tokenizer, max_len=128)
# train_size = int(0.2 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset,val_dataset = train_test_split(dataset, test_size=0.8, random_state=42)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 classes: 0, 1, 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer, loss function, and scheduler
from transformers import AdamW
from transformers import get_scheduler

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_training_steps = len(train_loader) * 4  # Assuming 4 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

    print(f'Epoch {epoch} Training Loss: {train_loss / len(train_loader)}')

output_dir = "/content/saved_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

# Evaluation loop
model.eval()
val_loss = 0
predictions, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        val_loss += loss.item()

        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    print(f'Validation Loss: {val_loss / len(val_loader)}')

# Decode predictions back to original labels
decoded_predictions = [0 if pred == 1 else -1 if pred == 0 else 1 for pred in predictions]
decoded_true_labels = [0 if label == 1 else -1 if label == 0 else 1 for label in true_labels]

# Classification Report
print('Classification Report:')
print(classification_report(decoded_true_labels, decoded_predictions))

# Confusion Matrix
print('Confusion Matrix:')
cm = confusion_matrix(decoded_true_labels, decoded_predictions)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

