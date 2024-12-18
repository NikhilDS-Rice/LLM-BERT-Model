import matplotlib.pyplot as plt
import random

models = ["LSTM", "BI-LSTM", "GRU", "CNN", "BERT"]
accuracies = [87, 88, 86, 85, 97]  # Random values for now

colors = [ 'skyblue', 'lightgreen', 'darkblue', 'gold', 'plum']

plt.figure(figsize=(12, 5))
bars = plt.bar(models, accuracies, color=colors)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height()}%",
        ha='center',
        va='bottom',
        fontsize=30
    )

plt.ylabel("Accuracy (%)", fontsize=24)
plt.title("Comparison of Model Accuracy Across Architectures", fontsize=28)
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
