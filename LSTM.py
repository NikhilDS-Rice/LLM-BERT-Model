import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from torch.utils.tensorboard import SummaryWriter

nltk.download('punkt')
nltk.download('punkt_tab')

class CommentDataset(Dataset):
    def __init__(self, comments, labels, vocab, max_len):
        self.comments = comments
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments.iloc[idx]
        label = self.labels[idx]

        # Tokenizing the words and assigning them index values
        tokens = word_tokenize(str(comment).lower())
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Padding
        if len(indices) < self.max_len:
            indices.extend([self.vocab["<PAD>"]] * (self.max_len - len(indices)))
        else:
            indices = indices[:self.max_len]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        out = self.fc(hidden)
        return self.softmax(out)

# modifications to the dataset and generating the top 10000 most occuring words in the dataset
def preprocess_data(data, max_vocab_size=10000, max_len=50, train_frac=1.0, test_frac=1.0):
    data = data.drop(columns=["Unnamed: 0"])
    data = data.drop(columns=["likes"])
    data = data.drop(columns=["replies"])
    data = data.dropna(subset=["comment_text", "pol"])
    data["pol"] = data["pol"].replace(-1, 2).astype(int)

    unique_video_ids = data['video_id'].unique()
    np.random.shuffle(unique_video_ids)

    split_index = int(0.9 * len(unique_video_ids))
    train_video_ids = unique_video_ids[:split_index]
    test_video_ids = unique_video_ids[split_index:]

    train_data = data[data['video_id'].isin(train_video_ids)]
    test_data = data[data['video_id'].isin(test_video_ids)]

    train_sample = train_data.sample(frac=train_frac, random_state=42).reset_index(drop=True)
    test_sample = test_data.sample(frac=test_frac, random_state=42).reset_index(drop=True)

    comments = pd.concat([train_sample['comment_text'], test_sample['comment_text']])
    labels = pd.concat([train_sample['pol'], test_sample['pol']])
    tokenized_comments = [word_tokenize(comment.lower()) for comment in comments]
    all_tokens = [token for comment in tokenized_comments for token in comment]
    most_common_tokens = Counter(all_tokens).most_common(max_vocab_size - 2)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({token: idx + 2 for idx, (token, _) in enumerate(most_common_tokens)})

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return train_sample, test_sample, comments, labels, vocab, len(label_encoder.classes_)

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device, writer, clip_grad=1.0):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)
        writer.add_scalar('Train/Loss', train_losses[-1], epoch + 1)
        writer.add_scalar('Train/Accuracy', train_acc, epoch + 1)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100. * correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        writer.add_scalar('Validation/Loss', val_losses[-1], epoch + 1)
        writer.add_scalar('Validation/Accuracy', val_acc, epoch + 1)

        print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")

    return model, train_losses, train_accuracies, val_losses, val_accuracies

file_path = "youtube_comments_train.csv"
data = pd.read_csv(file_path)

# Preprocess data
train_sample, test_sample, comments, labels, vocab, num_classes = preprocess_data(data)

# Hyperparameters
max_len = 50
embedding_dim = 100
hidden_dim = 128
num_layers = 2
batch_size = 32
learning_rate = 0.001
epochs = 10

train_dataset = CommentDataset(train_sample['comment_text'], train_sample['pol'], vocab, max_len)
test_dataset = CommentDataset(test_sample['comment_text'], test_sample['pol'], vocab, max_len)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=num_classes,
    num_layers=num_layers
)

criterion = nn.CrossEntropyLoss()
model =  model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard writer
writer = SummaryWriter(log_dir="runs/LSTM_lr_0.0001")
# Training  the model
trained_model, train_losses, train_accuracies, val_losses, val_accuracies = train_model(
    model, train_loader, val_loader, epochs, criterion, optimizer, device, writer
)

writer.close()

torch.save(trained_model.state_dict(), "LSTM_comment_model.pth")

