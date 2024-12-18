# COMP576 Project: Sentiment Analysis of YouTube Comments

This project focuses on analyzing YouTube comments to determine audience perception using various **deep learning** and **natural language processing (NLP)** techniques. The models implemented include LSTM, GRU, CNN, and the transformer-based **BERT** architecture.

---

## 📄 **Project Overview**

In the absence of YouTube's dislike count, comments serve as a key indicator of audience perception. This project systematically processes and analyzes these comments to determine sentiment using:

1. **Deep Learning Models**:
   - LSTM, Bi-LSTM, GRU, CNN, and BERT.
2. **Statistical Significance Testing**:
   - McNemar's Test for evaluating model performance.
3. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score.

BERT significantly outperformed other models, achieving **97% accuracy**, with improvements validated using McNemar’s test.

---

## 📁 **Repository Structure**

The repository contains the following files:

### **Code Files**
| File Name                          | Description                              |
|------------------------------------|------------------------------------------|
| `LSTM.py`                          | LSTM model implementation.               |
| `Bi-LSTM.py`                       | Bi-LSTM model implementation.            |
| `GRU.py`                           | GRU model implementation.                |
| `CNN.py`                           | CNN model implementation.                |
| `bert.py`                          | BERT model implementation.               |
| `Accuracy.py`                      | Script to compute model accuracies.      |
| `Accuracy_BarChart.py`             | Generates accuracy comparison charts.    |
| `McNemar_test.py`                  | McNemar’s Test for performance comparison.|
| `BERT_analyze_comments.py`         | BERT-specific comment analysis.          |
| `LSTM_unprocessed_dataset.py`      | LSTM model using raw dataset.            |
| `Baseline_accuracy.py`             | Baseline model accuracy calculation.     |

### **Notebooks**
- `BERT.ipynb`: Jupyter Notebook for BERT implementation.
- `Pull_all_Comments_and_Replies_for_YouTube_Playlists.ipynb`: Notebook to scrape YouTube comments.

### **Data**
- `datasets.7z`: Dataset containing labeled YouTube comments.

---

## 🔧 **Setup and Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Krish-Arulalan/COMP576_Project.git
   cd COMP576_Project
