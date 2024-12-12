from flask import Flask, request, jsonify
from flask_cors import CORS  # Importar CORS
import random
import json
import pickle
from keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import os

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('backend/intents.json').read())
words = pickle.load(open('backend/words.pkl', 'rb'))
classes = pickle.load(open('backend/classes.pkl', 'rb'))
model = load_model('backend/maceta.keras')

app = Flask(__name__)
CORS(app)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i['responses'])
    return "Lo siento, no entiendo esa pregunta."

@app.route('/chatbot', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message")

    if message:
        tag = predict_class(message)
        response = get_response(tag, intents)
        return jsonify({"response": response})
    return jsonify({"error": "No message received"})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
