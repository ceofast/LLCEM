from tensorflow import math
from datasets import load_dataset
from transformers import AutoTokenizer
from tensorflow.keras.optimizers import Adam
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification

dataset = load_dataset("rotten_tomatoes")

print(dataset)

print(dataset["test"][0])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenizer(dataset["train"][0]["text"])

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation = True)


dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                        return_tensors="tf")

my_model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

tf_train_set = my_model.prepare_tf_dataset(dataset["train"],
                                        shuffle=True,
                                        batch_size=16,
                                        collate_fn=data_collator)

tf_validation_set = my_model.prepare_tf_dataset(dataset["validation"],
                                                shuffle=False,
                                                batch_size=16,
                                                collate_fn=data_collator)


my_model.compile(optimizer=Adam(3e-5))

my_model.fit(x=tf_train_set,
             validation_data = tf_validation_set,
             epochs = 2)


text = "I love NLP. It's fun to analyze NLP tasks with Hugging Face"

tokenized_text = tokenizer(text, return_tensors="tf")

logits = my_model(**tokenized_text).logits

int(math.argmax(logits, axis=-1)[0])

