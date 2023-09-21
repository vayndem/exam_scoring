import torch
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Input sentences
sentence1 = "Python merupakan bahasa pemrograman komputer yang biasa dipakai untuk membangun situs, software/aplikasi, mengotomatiskan tugas dan melakukan analisis data. Bahasa pemrograman ini termasuk bahasa tujuan umum"
sentence2 = "c++ merupakan bahasa pemrograman yang baik"

# Get embeddings for the sentences
embeddings1 = model.encode(sentence1, convert_to_tensor=True)
embeddings2 = model.encode(sentence2, convert_to_tensor=True)

# Calculate Cosine Similarity between embeddings
similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

# Use a dynamic threshold based on the distribution of similarity scores
# You may need to collect and preprocess a dataset for this purpose
# For now, you can use a static threshold
threshold = 0.8

if similarity.item() > threshold:
    prediction = "Similar"
else:
    prediction = "Not Similar"

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Prediction: {prediction} (Cosine Similarity: {similarity.item()})")
