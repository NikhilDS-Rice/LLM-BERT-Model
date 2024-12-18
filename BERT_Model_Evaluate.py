from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from safetensors.torch import load_file
import torch

# Paths to the files
model_path = './BERT model/model.safetensors'
config_path = './BERT model/config.json'
tokenizer_path = './BERT model/vocab.txt'

# Load tokenizer
tokenizer = BertTokenizer(vocab_file=tokenizer_path)

# Load model configuration
config = BertConfig.from_json_file(config_path)

# Initialize the model with the configuration
model = BertForSequenceClassification(config)

# Load weights using safetensors
state_dict = load_file(model_path)
model.load_state_dict(state_dict)

# Mapping IDs to labels (update based on your model's id2label)
id2label = {0: "Negative", 1: "Positive",2: "Neutral"}

# Function to predict sentiment
def predict_sentiment(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return id2label[prediction]

# Loop for dynamic input
print("Sentiment Analysis - Enter a sentence to analyze or type 'exit' to quit.")
while True:
    user_input = input("Enter a sentence: ")
    if user_input.lower() == "exit":
        print("Exiting Sentiment Analysis. Goodbye!")
        break
    sentiment = predict_sentiment(user_input)
    print(f"Sentiment: {sentiment}")
