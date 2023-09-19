from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

# Contoh teks yang akan dibandingkan
essay1 = "my name is not buddy"
essay2 = "im eating some food"


# Tokenisasi teks dan konversi menjadi tensor
tokens1 = tokenizer(essay1, return_tensors='pt')
tokens2 = tokenizer(essay2, return_tensors='pt')

# Inferensi menggunakan model
with torch.no_grad():
    output1 = model(**tokens1)
    output2 = model(**tokens2)

# Hitung similarity score anntara kedua hasil output
similarity_score = torch.cosine_similarity(output1.last_hidden_state.mean(dim=1), output2.last_hidden_state.mean(dim=1)).item()

similarity_score = round(similarity_score, 3)

print(similarity_score*100)


