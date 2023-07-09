import os
import dotenv
import requests
from torch import from_numpy, tensor
from torch.nn.functional import softmax
from flask import Flask, request, jsonify, send_from_directory

from transformers import AutoTokenizer, AutoModelForSequenceClassification

dotenv.load_dotenv()
API_KEY = os.getenv('API_KEY')
headers = {"Authorization": f"Bearer {API_KEY}"}

API_URL = "https://api-inference.huggingface.co/models/caballeroch/FakeNewsClassifierDistilBert"

model, tokenizer = None, None

app = Flask(__name__, static_folder='../client')

@app.route('/')
def serve_index():
    return send_from_directory('../client', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../client', filename)

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    result = classify_text(text)
    # result = classify_with_loaded_model(text)

    return jsonify({'result': result})

def load_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained('caballeroch/FakeNewsClassifierDistilBert')
    tokenizer = AutoTokenizer.from_pretrained('caballeroch/FakeNewsClassifierDistilBert')
    
    return model, tokenizer

def classify_with_loaded_model(text):
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    logits = model(
        input_ids=tensor(inputs['input_ids']), 
        attention_mask=tensor(inputs['attention_mask'])
    ).logits

    y_proba = softmax(logits)
    
    classification = int(y_proba[0][0].item() * 100)
    print(classification)
    
    return classification
    
def classify_text(text):
    payload = {
        "inputs": text
    }
    response = requests.post(API_URL, json=payload)
    result = response.json()

    print(result)
    if result[0][0]['label'] == 'LABEL_0':
        classification = int(result[0][0]['score']*100)
    else:
        classification = int(result[0][1]['score']*100)
    print(classification)
    
    return classification

if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    app.run()
