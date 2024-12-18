import pandas as pd

file_path = "youtube_comments_train.csv"
data = pd.read_csv(file_path)

print("Class Distribution in 'pol':")
class_counts = data['pol'].value_counts()
print(class_counts)

majority_class = class_counts.idxmax()
print(f"\nThe majority class is: {majority_class}")

total_samples = len(data)
majority_class_count = class_counts.max()
baseline_accuracy = majority_class_count / total_samples * 100

print(f"\nMajority Class Baseline Accuracy: {baseline_accuracy:.2f}%")
