import pandas as pd
from datasets import load_dataset, concatenate_datasets

try:
    df = pd.read_csv(self.data_file, encoding='utf-8')
except FileNotFoundError as exc:
    raise FileNotFoundError(f"Data file not found: {self.data_file}") from exc

labels = df['label'].unique().tolist()

# Load the full dataset
full_dataset = load_dataset('csv', data_files=self.data_file)['train']
train_subsets = []
eval_subsets = []
test_subsets = []

# Applying stratification to ensure balanced train and test subsets
unique_labels = full_dataset.unique('label_text')
for unique_label in unique_labels:
    # Filter the dataset for the current label
    label_dataset = full_dataset.filter(lambda example: example['label_text'] == unique_label)

    # Shuffle the dataset for this label
    label_dataset = label_dataset.shuffle(seed=42)

    # Split the dataset for this label into train, eval, and test subsets
    label_length = len(label_dataset)
    train_length = min(label_length, int(label_length * 0.7))  # 70% for training
    eval_length = min(max(label_length - train_length, 0), int(label_length * 0.15))  # 15% for evaluation
    test_length = min(max(label_length - train_length - eval_length, 0), int(label_length * 0.15))  # 15% for testing

    train_subset = label_dataset.select(range(train_length))
    eval_subset = label_dataset.select(range(train_length, train_length + eval_length))
    test_subset = label_dataset.select(range(train_length + eval_length, train_length + eval_length + test_length))

    # Add the subsets to the respective lists
    train_subsets.append(train_subset)
    eval_subsets.append(eval_subset)
    test_subsets.append(test_subset)

# Concatenate the subsets to form the final train and test datasets
train_dataset = concatenate_datasets(train_subsets)
eval_dataset = concatenate_datasets(eval_subsets)
test_dataset = concatenate_datasets(test_subsets)

# Shuffle the final datasets
train_dataset = train_dataset.shuffle(seed=42)
eval_dataset = eval_dataset.shuffle(seed=42)
test_dataset = test_dataset.shuffle(seed=42)
