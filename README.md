# Sentiment Analysis of YouTube Comments

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

<img width="477" alt="yt" src="https://github.com/user-attachments/assets/6664921a-fe07-4cc6-b295-f5502ffb91b6" />


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
- <img width="470" alt="piechart" src="https://github.com/user-attachments/assets/402c47c8-afcb-44c8-86d7-32a478aac9da" />

### **Results**
1. **Deep Learning Models Comparison**:
<img width="470" alt="compare" src="https://github.com/user-attachments/assets/3b3c8a58-a1bf-4356-b5a7-4487fb240986" />
 
2. **Confusion Matrix of BERT Model**: 
<img width="470" alt="confusionMatrixBERT" src="https://github.com/user-attachments/assets/a3fa949b-ae12-42d4-82c1-06c9a6aedded" />

3. **McNemar test on test dataset**: 
<img width="470" alt="McNemar test all models" src="https://github.com/user-attachments/assets/7ab2625e-aa32-44f8-a03a-b4f30dd87888" />




---

## 🔧 **Setup and Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NikhilDS-Rice/LLM-BERT-Model.git
