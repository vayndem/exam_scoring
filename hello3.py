import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import json
import random
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pyaspeller import YandexSpeller

def load(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data

# load dictionary
mydict = load('dict.json')

# Membuat objek StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def getSinonim(kalimat):
    kalimat_split = kalimat.split()
    sinonim_kalimat = []

    for kata in kalimat_split:
        if kata in mydict.keys():
            sinonim_list = mydict[kata]['sinonim']
            if sinonim_list:
                i = random.randint(0, len(sinonim_list) - 1)
                sinonim_kata = sinonim_list[i]
                sinonim_kalimat.append(sinonim_kata)
            else:
                sinonim_kalimat.append(kata)  # Jika tidak ada sinonim, gunakan kata itu sendiri
        else:
            sinonim_kalimat.append(kata)  # Jika kata tidak ada dalam kamus, gunakan kata itu sendiri

    return ' '.join(sinonim_kalimat)  # Menggabungkan kembali kata-kata menjadi kalimat

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

# Contoh penggunaan
data = input("Masukkan kata untuk jawaban soal: ")
output = preprocess_input(data)

# Contoh data pelatihan (jawaban dan label)
jawaban_siswa = [output] + [getSinonim(output) for _ in range(4)]
label = [1, 1, 1, 1, 1]


# Ekstraksi fitur menggunakan vektor TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(jawaban_siswa)

# Membangun model K-NN dengan metrik jarak Euclidean
k = 5
model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
model.fit(X, label)

# Input jawaban dari pengguna
jawaban_pengguna = input("Masukkan jawaban Anda: ")

jawaban_pengguna = preprocess_input(jawaban_pengguna)

# Ekstraksi fitur dari jawaban pengguna
X_pengguna = vectorizer.transform([jawaban_pengguna])

# Hitung kemiripan menggunakan metrik jarak Euclidean
distances, indices = model.kneighbors(X_pengguna)

# Tampilkan nilai kemiripan sebagai persentase (dalam hal ini, semakin kecil jaraknya, semakin mirip)
min_distance = np.min(distances)
percentage_similarity = 100 / (1 + min_distance)  # Normalisasi nilai jarak

if percentage_similarity >= 50:  # Ubah threshold sesuai kebutuhan
    print(f"Jawaban Anda mirip dengan salah satu jawaban dalam data pelatihan ({percentage_similarity:.2f}% kemiripan).")
else:
    print("Jawaban Anda tidak mirip dengan jawaban dalam data pelatihan.")
