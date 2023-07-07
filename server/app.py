from flask import Flask, request, jsonify
from flask import send_from_directory

app = Flask(__name__, static_folder='../client')

@app.route('/')
def serve_index():
    return send_from_directory('../client', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('client', filename)

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    # Call your machine learning model to classify the text
    result = classify_text(text)
    return jsonify({'result': result})

def classify_text(text):
    # Your machine learning code to classify the text
    # Return the classification result
    return 'Positive'  # Replace with the actual classification result