import torch
from transformers import Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification

dataset = load_dataset("rotten_tomatoes")

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])


dataset = dataset.map(tokenize_dataset, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Hyperparameter Tunning
training_args = TrainingArguments(output_dir="my_bert_model",
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  num_train_epochs=2)

# Model Training
trainer = Trainer(model = model,
                  args = training_args,
                  train_dataset = dataset["train"],
                  eval_dataset=dataset["test"],
                  tokenizer=tokenizer,
                  data_collator=data_collator)

trainer.train()

# Model Inference
text = "I love NLP. It's fun to analyze the NLP tasks with HuggingFace"

inputs = tokenizer(text, return_tensors = "pt")

model_path = "/working/my_bert/checkpoint-1000"

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

with torch.no_grad():
    logits = model(**inputs).logits


logits.argmax().item()

