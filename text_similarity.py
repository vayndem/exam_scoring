import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import json
import random
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pyaspeller import YandexSpeller

factory = StemmerFactory()
stemmer = factory.create_stemmer()


def typo2(text):
    speller = YandexSpeller(lang='en')
    result = speller.spell(text)

    corrected_text = text
    for typo in result:
        incorrect_word = typo['word']
        if typo['s']:
            corrected_word = typo['s'][0]
            corrected_text = corrected_text.replace(incorrect_word, corrected_word, 1)

    return corrected_text

def preprocess_input(text):
    lower = text.lower()
    trimmer = lower.translate(str.maketrans("", "", string.punctuation))
    stem = stemmer.stem(trimmer)
    output = typo2(stem)
    return output




# Load pretrained BERT model and tokenizer
# bert-base-multilingual-uncased
model_name = 'indobenchmark/indobert-base-p2'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


# input dengan ketikan
# data1 = input("Masukkan kata untuk jawaban soal: ")
# sentence1 = preprocess_input(data1)
# data2 = input("Masukkan kata untuk jawaban soal: ")
# sentence2 = preprocess_input(data2)


# Input sentences
sentence1 = "Python merupakan bahasa pemrograman komputer yang biasa dipakai untuk membangun situs, software/aplikasi, mengotomatiskan tugas dan melakukan analisis data. Bahasa pemrograman ini termasuk bahasa tujuan umum"
sentence2 = "c++ merupakan bahasa pemrograman yang baik"

# Tokenize the sentences and encode them
inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True)
inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True)

# Get embeddings for the sentences
with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

# Extract embeddings from token-level representations and normalize them
embedding1 = normalize(outputs1.last_hidden_state.mean(dim=1).numpy())
embedding2 = normalize(outputs2.last_hidden_state.mean(dim=1).numpy())

# Calculate Cosine Similarity between embeddings
similarity = cosine_similarity(embedding1, embedding2)

# Set a threshold for similarity
threshold = 0.8

# Determine if the sentences are similar or not
prediction = "Similar" if similarity[0][0] > threshold else "Not Similar"

# Print the results
print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Prediction: {prediction} (Cosine Similarity: {similarity[0][0]})")
