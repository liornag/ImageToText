from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pytesseract
from PIL import Image, ImageOps
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS
import json
import re

# === Flask setup ===
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load credentials ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Image preprocessing ===
def preprocess_image(path):
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageOps.invert(img)
    img.save(path)
    return path

# === Clean OCR garbage
def clean_ocr_text(text):
    return (
        text.replace('\u200f', '')
            .replace('\u200e', '')
            .replace(':', ' ')
            .replace('₪', '')
            .replace('ILS', '')
    )

# === GPT Analysis
def ask_gpt_to_analyze(lines):
    prompt = (
        "You will receive raw text extracted from a shopping receipt (in Hebrew or English). "
        "Your task is to extract only the **purchased items**, and return them as a clean JSON array.\n\n"
        "Each item must include:\n"
        "- name (product name)\n"
        "- quantity (if missing, default to 1)\n"
        "- price (number only in NIS)\n\n"
        "Ignore any business details, date, time, totals, tips, payment types, waiter name, or summary lines.\n"
        "תחזיר רק פריטים שנרכשו: שם, כמות ומחיר. אל תחזיר פרטי עסק, סה\"כ, עודף, כרטיס אשראי, טיפ או תאריך.\n\n"
        "Output ONLY valid JSON array. Do NOT add any explanation, title, or markdown.\n\n"
        "Example input:\n"
        "קולה 2 14.00\n"
        "צ'יפס קטן 1 10.00\n"
        "Ice Coffee 9.50\n"
        "סה\"כ 33.50\n\n"
        "Expected Output:\n"
        "[\n"
        "  {\"name\": \"קולה\", \"quantity\": 2, \"price\": 14.00},\n"
        "  {\"name\": \"צ'יפס קטן\", \"quantity\": 1, \"price\": 10.00},\n"
        "  {\"name\": \"Ice Coffee\", \"quantity\": 1, \"price\": 9.50}\n"
        "]\n\n"
        f"Now analyze this receipt:\n{chr(10).join(lines)}"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at analyzing Hebrew and English receipts."},
            {"role": "user", "content": prompt}
        ]
    )

    gpt_output = response.choices[0].message.content.strip()
    try:
        # 💡 Try to extract only JSON array from response
        json_match = re.search(r"\[\s*{.*?}\s*]", gpt_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            print("❌ Could not parse JSON from GPT output:\n", gpt_output)
            raise ValueError("Invalid JSON from GPT")
    except Exception as e:
        print("⚠️ GPT returned invalid JSON:", e)
        raise ValueError("GPT returned invalid JSON")

# === API endpoint
@app.route('/scan-receipt', methods=['POST'])
def scan_receipt():
    print("📥 Received file from client.")

    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        preprocess_image(path)
        raw_text = pytesseract.image_to_string(Image.open(path), lang='heb+eng')
        os.remove(path)

        cleaned_text = clean_ocr_text(raw_text)
        lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
        print("📝 OCR lines:", lines)

        gpt_result = ask_gpt_to_analyze(lines)
        print("✅ GPT result:", gpt_result)

        return jsonify({'items': gpt_result})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5100)
