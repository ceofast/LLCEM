from datasets import load_dataset

emotions = load_dataset("dair-ai/emotion")

emotions

emotions["train"][0]

from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"suprise"}
label2id = {"sadness":0, "joy":1, "love":2, "anger":3, "fear":4, "suprise":5}

from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(model_ckpt,
                                                             num_labels=6,
                                                             id2label=id2label,
                                                             label2id=label2id)

import os

hf_token = os.getenv("HF_TRANSFORMERS_CACHE")

tf_train_set = model.prepare_tf_dataset(emotions_encoded["train"],
                                        shuffle=True,
                                        batch_size=16,
                                        collate_fn=data_collator)

tf_validation_set = model.prepare_tf_dataset(emotions_encoded["validation"],
                                             shuffle=True,
                                             batch_size=16,
                                             collate_fn=data_collator)

import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer)

from transformers.keras_callbacks import KerasMetricCallback

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)

from transformers.keras_callbacks import PushToHubCallback

push_to_hub_callback = PushToHubCallback(output_dir="emotiom_analysis_with_distilbert", tokenizer=tokenizer)

model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=[metric_callback, push_to_hub_callback])

custom_text = "I watched a movie yesterday. It was really awesome"

from transformers import pipeline

classifier = pipeline("sentiment-analysis", model = "ceofast/emotiom_analysis_with_distilbert")

inputs = tokenizer(custom_text, return_tensors = "tf")

logits = model(**inputs).logits

predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])

model.config.id2label[predicted_class_id]

