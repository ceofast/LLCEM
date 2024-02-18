import pandas as pd
from transformers import pipeline

# Duygu Analizi
classifier = pipeline("sentiment-analysis")


text = "It's great to learn NLP for me."

classifier(text)

outputs = classifier(text)
pd.DataFrame(outputs)

# Zero-shot sınıflandırma

classifier = pipeline("zero-shot-classification")

text = "This is a tutorial about Hugging Face"

labels = ["tech", "education", "business"]

outputs = classifier(text, labels)

pd.DataFrame(outputs)

generator = pipeline("text-generation")

prompt = "This tutorial will walk you through how to"

outputs = generator(prompt)

print(outputs)

# Metnin boyutunu 50 ile sınırlandırmak için

outputs = generator(prompt, max_length = 50)
print(outputs[0]["generated_text"])


# Özelleşmiş model çağırmak için.
generator = pipeline("text-generation", model="distilgpt2")

outputs = generator(prompt, max_length=50)
print(outputs[0]["genetated_text"])


## NER ##

ner = pipeline("ner", group_entities=True)

text = "My name is Tirendaz from Turkey. HuggingFace is a nice platform."

outputs = ner(text)
pd.DataFrame(outputs)

reader = pipeline("question-answering")
text = "My name is Tirendaz and I love Berlin."

question = "Where do I like"

outputs = reader(question=question, context=text)
pd.DataFrame([outputs])


# Metin Özetleme #
summarizer = pipeline("summarization")

text = """James Stephen "Jimmy" Donaldson[b] (born May 7, 1998), better known by his online alias MrBeast, is an American YouTuber, online personality, entrepreneur, and philanthropist. He is known for his fast-paced and high-production videos, which feature elaborate challenges and large giveaways.[11] With over 240 million subscribers, he is the most-subscribed individual on YouTube and the second-most-subscribed channel overall.

Donaldson grew up in Greenville, North Carolina. He began posting videos to YouTube in early 2012, at the age of 13,[12] under the handle MrBeast6000. His early content ranged from Let's Plays to "videos estimating the wealth of other YouTubers".[13] He went viral in 2017 after his "counting to 100,000" video earned tens of thousands of views in just a few days, and he has become increasingly popular ever since, with most of his videos gaining tens of millions of views.[13] His videos became increasingly grand and extravagant.[14] Once his channel took off, Donaldson hired some of his childhood friends to co-run the brand. As of 2023, the MrBeast team is made up of over 250 people, including Donaldson himself.[15] Other than MrBeast, Donaldson runs the YouTube channels Beast Reacts, MrBeast Gaming, MrBeast 2 (formerly MrBeast Shorts)[16] and the philanthropy channel Beast Philanthropy.[17][18] He formerly ran MrBeast 3 (initially MrBeast 2), which is now inactive.[19][20]

Donaldson is the founder of MrBeast Burger and Feastables; and a co-creator of Team Trees, a fundraiser for the Arbor Day Foundation that has raised over $23 million;[21][22] and Team Seas, a fundraiser for Ocean Conservancy and The Ocean Cleanup that has raised over $30 million.[23] Donaldson won the Creator of the Year award four years in a row at the Streamy Awards in 2020, 2021, 2022, and 2023; he also won the Favorite Male Creator award twice at the Nickelodeon Kids' Choice Awards in 2022 and 2023. In 2023, Time named him as one of the world's 100 most influential people. He has ranked on the Forbes list for the highest paid YouTube creator in 2022[24] and has an estimated net worth of $500 million.[25]"""

outputs = summarizer(text, max_length=60, clean_up_tokenization_spaces = True)

print(outputs[0]["summary_text"])

# Metin Çeviri #
translator = pipeline("translation_en_to_de")

text = "I hope you enjoy it"

outputs = translator(text, clean_up_tokenization_spaces = True)

print(outputs[0]["translation_text"])

# İngilizce'den Türkçe'ye çeviri modeli : opus-mt-big-en-tr

