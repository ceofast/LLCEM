# TODO: Load and process datasets
# TODO: Tokenize: input texts
# TODO: Load models, train and infer
# TODO: Load metrics and evaluate models

import os
import torch
import evaluate
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import Trainer
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import TrainingArguments
from huggingface_hub import notebook_login
from transformers import DistilBertTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

hf_token = os.getenv("HF_TRANSFORMERS_CACHE")

emotions = load_dataset("dair-ai/emotion")

print(emotions)

train_ds = emotions["train"]
print(train_ds)

test_ds = emotions["test"]
print(test_ds)

len(train_ds)

train_ds[1]

train_ds.column_names # Columnar Format

train_ds.features

train_ds[:5]

train_ds["text"][:5]

emotions.set_format(type="pandas")

df = emotions["train"][:]
df.head()

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)

df.head()

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")

# TODO: Oversample & Undersample

df["Words Per Tweet"] = df["text"].str.rsplit().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers =False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

emotions.reset_format()

# TODO: Tokenization
######################
# Data Preprocessing #
######################

text = "It is fun to work with NLP using HuggingFace."

tokenized_text = list(text)
print(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

df = pd.DataFrame({"name": ["can", "efe", "ada"],
                   "label": [0, 1, 2]})

pd.get_dummies(df, dtype=int)

input_ids = torch.tensor(input_ids)

one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))

one_hot_encodings.shape

print(f"Token:{tokenized_text[0]}")

print(f"Tensor index: {input_ids[0]}")

print(f"One-hot: {one_hot_encodings[0]}")


tokenized_text = text.split()
print(tokenized_text)

model_ckpt = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

distrilbert_tokenize = DistilBertTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

tokenizer.convert_tokens_to_string(tokens)

tokenizer.vocab_size

tokenizer.model_max_length

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

print(tokenize(emotions["train"][:2]))

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

emotions_encoded["train"].column_names

num_labels = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references = labels)


notebook_login()

training_args = TrainingArguments(output_dir="distilbert-emotion",
                                  num_train_epochs=2,
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  load_best_model_at_end=True,
                                  push_to_hub=True,
                                  hub_token=hf_token,
                                  report_to=None)


trainer = Trainer(model = model,
                  args=training_args,
                  compute_metrics = compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

trainer.train()


preds_output = trainer.predict(emotions_encoded["validation"])

preds_output.metrics


y_preds = np.argmax(preds_output.predictions, axis=1)

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


y_valid = np.array(emotions_encoded["validation"]["label"])

labels = emotions["train"].features["label"].names

plot_confusion_matrix(y_preds, y_valid, labels)

trainer.push_to_hub(commit_message="Training completed!")

model_id = "ceofast/distilbert-emotion"

classifier = pipeline("text-classification", model=model_id)

custom_text = "ı watched a movie yesterday. It was really good."

preds = classifier(custom_text, return_all_scores = True)

preds_df = pd.DataFrame(preds[0])

plt.bar(labels, 100*preds_df["score"])
plt.title(f'"{custom_text}"')
plt.ylabel("Class probability (%)")
plt.show()

