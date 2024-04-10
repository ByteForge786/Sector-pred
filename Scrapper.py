 import pandas as pd
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments

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

# Assuming you have a model and a data_collator defined

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Adjust the number of epochs as needed
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=500,
)

# Define function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
print("Training started...")
trainer.train()

# Evaluate on the test set
print("Evaluation on the test set:")
trainer.evaluate(test_dataset)
