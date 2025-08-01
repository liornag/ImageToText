from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pytesseract
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/scan-receipt', methods=['POST'])

def scan_receipt():
    print("===> Flask: Received POST request to /scan-receipt")
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        text = pytesseract.image_to_string(Image.open(filepath), lang='heb+eng')
        os.remove(filepath)  # delete temp file
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return jsonify({'items': lines})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5100)