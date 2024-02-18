from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


text = "Ce film est magnifique. Je l 'ai beaucoup aime."

classifier(text)

# From Tensorflow
from transformers import TFAutoModelForAudioClassification

encoding = tokenizer("This film is nice. I liked it.")
print(encoding)

batch = tokenizer(["I like NLP", "We hope you don't hate it."],
                  max_length = 512,
                  truncation = True,
                  padding = True,
                  return_tensors = "pt")

print(batch)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

outputs = model(**batch)

from torch import nn

predictions = nn.functional.softmax(outputs.logits, dim=1)
print(predictions)

# Model Save
save_directory = "./save_pretrained"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)


# Model Loading
model = AutoModelForSequenceClassification.from_pretrained("./save_pretrained")