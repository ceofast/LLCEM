from transformers import pipeline
from datasets import load_dataset
from datasets import Audio

pipe = pipeline("automatic-speech-recognition")

# MNINST14
dataset = load_dataset("PolyAI/minds14", name = "en-US", split="train")

dataset.features

dataset = dataset.cast_column("audio", Audio(sampling_rate=pipe.feature_extractor.sampling_rate),)

data = dataset[:4]["audio"]
result = pipe(data)

print([d["text"] for d in result])