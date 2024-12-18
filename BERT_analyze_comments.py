import pandas as pd
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from safetensors.torch import load_file
import torch

# Paths to the model files (update these paths to your local file locations)
model_path = './BERT model/model.safetensors'
config_path = './BERT model/config.json'
tokenizer_path = './BERT model/vocab.txt'
csv_file_path = './comments_data.csv'
output_file_path = './comments_analysis_output.csv'

# Load tokenizer, config, and model
tokenizer = BertTokenizer(vocab_file=tokenizer_path)
config = BertConfig.from_json_file(config_path)
model = BertForSequenceClassification(config)
state_dict = load_file(model_path)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Mapping IDs to labels
id2label = {0: "Negative", 1: "Positive", 2: "Neutral"}

# Define the sentiment prediction function
def predict_sentiment(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return id2label[prediction]

# Load the comments data
comments_data = pd.read_csv(csv_file_path)

# Apply sentiment analysis
comments_data['Sentiment'] = comments_data['Comment'].apply(predict_sentiment)

# Count the number of positive, negative, and neutral comments
sentiment_counts = comments_data['Sentiment'].value_counts()
print("Sentiment Counts:\n", sentiment_counts)

# Save the results to a new CSV file
comments_data.to_csv(output_file_path, index=False)
print(f"Sentiment analysis results saved to {output_file_path}")
