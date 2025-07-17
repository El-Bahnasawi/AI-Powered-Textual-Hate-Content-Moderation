import pandas as pd
import numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments)
from datasets import Dataset
import pandas as pd

model_name = 'distilbert-base-cased'

# desired_rows, total_rows = 35000, 360492

# # Generate a random sample of row indices to skip
# np.random.seed(42)  # Seed for reproducibility
# skip_indices = np.random.choice(np.arange(1, total_rows + 1), size=(total_rows - desired_rows), replace=False)

# # Read the specified number of rows randomly
# df = pd.read_csv("superset.csv", skiprows=lambda x: x in skip_indices)
# df.to_csv("superset_sample.csv", index=False)


tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load and prepare data
df = pd.read_csv("data/superset_sample.csv")
df['labels'] = df['labels'].astype(int)
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x['text'], padding=True, truncation=True, max_length=64), batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)