

    # ...existing code...
from flask import Flask, request, jsonify, render_template, render_template_string
import nltk
import os
import pickle
import numpy as np
from  nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
                     
ps = PorterStemmer() 

# NLTK data folder next to this file
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# download required resources (quietly)
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
#nltk.download('punkt_tag', download_dir=nltk_data_dir)
# preload stopwords to avoid runtime lookups
STOPWORDS = set(stopwords.words('english'))

# load vectorizer and model using absolute paths
base_dir = os.path.dirname(__file__)
tfIdf = pickle.load(open(os.path.join(base_dir, 'vectorizer.pkl'), 'rb'))

model_path = os.path.join(base_dir, 'model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            prediction = "Error: Message cannot be empty."
        else:
            cleaned = transform_text(message)
            features = tfIdf.transform([cleaned])
            result = model.predict(features)[0]
            prediction = "Spam" if result == 1 else "Not Spam"
    return render_template('index.html', prediction=prediction)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in STOPWORDS and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for spam detection."""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    cleaned = transform_text(message)
    features = tfIdf.transform([cleaned])
    result = model.predict(features)[0]
    return jsonify({"prediction": "Spam" if result == 1 else "Not Spam"})

if __name__ == "__main__":
    app.run(debug=True)