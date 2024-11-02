import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

# Inicializa o Flask
app = Flask(__name__)

# Função para extrair texto de uma URL
def extract_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
        return content
    else:
        return "Erro ao acessar a URL"

# Função para pré-processar o texto
def preprocess_text(text):
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=500, padding="post", truncating="post")
    return np.array(padded), tokenizer.word_index

# Função para criar e treinar um modelo de rede neural simples
def create_and_train_model(padded_content):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=500),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Treinamento do modelo se houver dados
    if padded_content.shape[0] > 0:
        labels = np.array([1])  # Rótulo fictício, modifique conforme necessário
        model.fit(padded_content, labels, epochs=10, verbose=1)
    return model

# Rota principal do site
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        content = extract_text_from_url(url)
        padded_content, word_index = preprocess_text(content)
        model = create_and_train_model(padded_content)
        return render_template("index.html", content=content, message="Aprendizado Completo!")
    return render_template("index.html", content="", message="")

# Inicia o servidor Flask
if __name__ == "__main__":
    app.run(debug=True)
