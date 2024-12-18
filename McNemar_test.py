import pandas as pd
import torch
from nltk.tokenize import word_tokenize

test_file_path = "youtube_comments_test.csv"
test_data = pd.read_csv(test_file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#make sure the LSTM class has been defined before running this
model = LSTMModel(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=num_classes,
    num_layers=num_layers,
)
model.load_state_dict(torch.load("LSTM_comment_model.pth", map_location=device))
model = model.to(device)
model.eval()


def test_model_on_device(test_data, model, vocab, device, max_len=50):

    predictions = []

    test_data['comment_text'] = test_data['comment_text'].fillna("").astype(str)

    for _, row in test_data.iterrows():
        comment = row['comment_text']

        tokens = word_tokenize(comment.lower())
        indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [vocab["<PAD>"]] * (max_len - len(indices))

        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        predictions.append(prediction)

    return predictions
#make sure vocabulary is created (done while training the model in previous codes)

predictions_model1 = test_model_on_device(test_data, model, vocab, device, max_len=50)

test_file_path = "youtube_comments_test.csv"
test_data = pd.read_csv(test_file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#make sure the class of this model is defined (run previous codes to do that)
model = BidirectionalLSTMModel(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=num_classes,
    num_layers=num_layers,
)
model.load_state_dict(torch.load("Bi-LSTM_comment_model.pth", map_location=device))
model = model.to(device)
model.eval()


def test_model_on_device(test_data, model, vocab, device, max_len=50):

    predictions = []

    test_data['comment_text'] = test_data['comment_text'].fillna("").astype(str)

    for _, row in test_data.iterrows():
        comment = row['comment_text']

        tokens = word_tokenize(comment.lower())
        indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [vocab["<PAD>"]] * (max_len - len(indices))

        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        predictions.append(prediction)

    return predictions

#make sure vocabulary is created (done while training the model in previous codes)
predictions_model2 = test_model_on_device(test_data, model, vocab, device, max_len=50)


def get_true_labels(csv_file_path):
    data = pd.read_csv(csv_file_path)

    if "pol" not in data.columns:
        raise ValueError("The required column 'pol' is missing in the file.")

    data['pol'] = data['pol'].replace(-1, 2)
    true_labels = data['pol'].tolist()

    return true_labels


csv_file_path = "youtube_comments_test.csv"

true_labels = get_true_labels(csv_file_path)

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.evaluate import mcnemar


def mcnemar_matrix(true_labels, model1_preds, model2_preds):
    contingency_table = np.zeros((2, 2), dtype=int)

    for true, pred1, pred2 in zip(true_labels, model1_preds, model2_preds):
        if pred1 == true and pred2 == true:
            contingency_table[0, 0] += 1  # Both correct
        elif pred1 == true and pred2 != true:
            contingency_table[0, 1] += 1  # Model 1 correct, Model 2 wrong
        elif pred1 != true and pred2 == true:
            contingency_table[1, 0] += 1  # Model 2 correct, Model 1 wrong
        else:
            contingency_table[1, 1] += 1  # Both wrong

    chi2, p = mcnemar(contingency_table, exact=True)
    print(f"P-value: {p}")

    fig, ax = plt.subplots(figsize=(6, 6))
    row_labels = ["CNN Correct", "CNN Wrong"]
    col_labels = ["LSTM Correct", "LSTM Wrong"]

    ax.set_xticks(np.arange(2) + 0.5)
    ax.set_yticks(np.arange(2) + 0.75)
    ax.set_xticklabels(col_labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels(row_labels, fontsize=12, fontweight='bold', rotation=90)

    ax.tick_params(left=False, bottom=False, labeltop=True, labelleft=True)
    ax.invert_yaxis()

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
            ax.text(j + 0.5, i + 0.5, contingency_table[i, j],
                    ha='center', va='center', fontsize=16, fontweight='bold')

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    plt.title("McNemar Test Contingency Table", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


mcnemar_matrix(true_labels, predictions_model1, predictions_model2)


