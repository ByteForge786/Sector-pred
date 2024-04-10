import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_multiclass_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = np.argmax(eval_pred.predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Example usage
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    "test_trainer"
)

trainer = Trainer(
    args=training_args,
    compute_metrics=compute_multiclass_metrics,
    # Other trainer setup...
)

# To evaluate
eval_results = trainer.evaluate()
print(eval_results)
