import pandas as pd
import torch
from nltk.tokenize import word_tokenize

test_file_path = "youtube_comments_test.csv"
test_data = pd.read_csv(test_file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#make sure the LSTM class has been defined before running this
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
predictions_Bi_LSTM = test_model_on_device(test_data, model, vocab, device, max_len=50)



def get_true_labels(csv_file_path):
    data = pd.read_csv(csv_file_path)

    if "pol" not in data.columns:
        raise ValueError("The required column 'pol' is missing in the file.")

    data['pol'] = data['pol'].replace(-1, 2)
    true_labels = data['pol'].tolist()

    return true_labels


csv_file_path = "youtube_comments_test.csv"

true_labels = get_true_labels(csv_file_path)


def calculate_accuracy(true_labels, pred_labels):
    correct_predictions = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)

    accuracy = correct_predictions / len(true_labels) * 100

    return accuracy


'''#perform when testing LSTM trained with unprocessed data
for i in range(len(predictions_Bi_LSTM)):
    if predictions_Bi_LSTM[i] == 1:
        predictions_Bi_LSTM[i] = 0.0
    elif predictions_Bi_LSTM[i] == 2:
        predictions_Bi_LSTM[i] = 1.0
    else:
        predictions_Bi_LSTM[i] = 2.0'''

accuracy = calculate_accuracy(true_labels, predictions_Bi_LSTM)
print(f"Accuracy: {accuracy:.2f}%")