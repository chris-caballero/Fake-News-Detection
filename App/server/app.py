import os
import dotenv
import requests
from flask import Flask, request, jsonify, send_from_directory

dotenv.load_dotenv()
API_KEY = os.getenv('API_KEY')
headers = {"Authorization": f"Bearer {API_KEY}"}

API_URL = "https://api-inference.huggingface.co/models/caballeroch/FakeNewsClassifierDistilBert"
headers = {"Authorization": "Bearer {}".format(API_KEY)}

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
    return jsonify({'result': result})

def classify_text(text):
    payload = {
        "inputs": text
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    print(result)
    if result[0][0]['label'] == 'LABEL_0':
        classification = int(result[0][0]['score']*100)
    else:
        classification = int(result[0][1]['score']*100)
    print(classification)
    
    return classification

if __name__ == '__main__':
    app.run()
