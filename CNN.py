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


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, filter_size, num_filters):
        """
        CNN for text classification.
        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimension of word embeddings.
        :param num_classes: Number of output classes.
        :param filter_size: Size of the filter (e.g., 2 for bi-grams).
        :param num_filters: Number of filters.
        """
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional layer
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=filter_size
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)

        # ReLU activation function
        conved = torch.relu(self.conv(embedded))

        #maxpooling
        pooled = torch.max(conved, dim=2)[0]

        # dropout and fully connected layers
        out = self.dropout(pooled)
        out = self.fc(out)
        return out




# modifications to the dataset and generating the top 10000 most occuring words in the dataset
def preprocess_data(data, max_vocab_size=10000, max_len=50, train_frac=1, test_frac=1):

    data = data.drop(columns=["Unnamed: 0"])

    data["likes"] = pd.to_numeric(data["likes"], errors="coerce").fillna(0).astype(int)
    data["replies"] = pd.to_numeric(data["replies"], errors="coerce").fillna(0).astype(int)

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

from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device, writer):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

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

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

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

        print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")

    return model, train_losses, train_accuracies, val_losses, val_accuracies

file_path = "youtube_comments_train.csv"
data = pd.read_csv(file_path)

# Preprocess data
train_sample, test_sample, comments, labels, vocab, num_classes = preprocess_data(data)

# Hyperparameters
max_len = 50
embedding_dim = 100
filter_size = 3
num_filters = 128
num_classes = num_classes
learning_rate = 0.001
epochs = 10
batch_size = 32


train_dataset = CommentDataset(train_sample['comment_text'], train_sample['pol'], vocab, max_len)
test_dataset = CommentDataset(test_sample['comment_text'], test_sample['pol'], vocab, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = TextCNN(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    num_classes=num_classes,
    filter_size=filter_size,
    num_filters=num_filters,
)

criterion = nn.CrossEntropyLoss()
model = cnn_model.to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

# TensorBoard writer
writer = SummaryWriter(log_dir="runs/CNN_num_fil_256")

# Training the model
cnn_model, cnn_train_losses, cnn_train_accuracies, cnn_val_losses, cnn_val_accuracies = train_model(
    cnn_model, train_loader, val_loader, epochs, criterion, optimizer, device, writer
)

writer.close()

torch.save(cnn_model.state_dict(), "cnn_comment_model.pth")

