from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os, json, re, pytesseract, requests
from PIL import Image, ImageOps, ImageFilter
from openai import OpenAI
from dotenv import load_dotenv

# Flask setup
app = Flask(__name__)
CORS(app, origins="http://localhost:5173", supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load credentials
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Image preprocessing
def preprocess_image(path):
    img = Image.open(path).convert("L")  
    img = img.resize((img.width * 2, img.height * 2))  
    img = img.filter(ImageFilter.MedianFilter())
    img = ImageOps.autocontrast(img)
    img = ImageOps.invert(img)
    img = img.point(lambda x: 0 if x < 140 else 255, mode='1')  
    img.save(path)
    return path


# Clean OCR garbage
def clean_ocr_text(text):
    return text.replace('\u200f', '').replace('\u200e', '').replace(':', ' ').replace('₪', '').replace('ILS', '')

# GPT logic
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
    print("GPT raw output:", gpt_output)  # DEBUG

    try:
        json_match = re.search(r"\[\s*{.*?}\s*]", gpt_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(gpt_output)
    except Exception as e:
        raise ValueError("GPT returned invalid JSON") from e

# API endpoint
@app.route('/scan-receipt', methods=['POST'])
def scan_receipt():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = '.jpg'
    filename = base + ext
    path = os.path.join(UPLOAD_FOLDER, filename)

    print("Saving file:", path)
    file.save(path)

    try:
        preprocess_image(path)

        print("===> Starting OCR")
        text = pytesseract.image_to_string(Image.open(path), lang='heb+eng', config='--oem 3 --psm 6')
        print("\n=== OCR TEXT ===")
        print(text)
        print("===> OCR Done")

        os.remove(path)

        lines = [clean_ocr_text(line) for line in text.split('\n') if line.strip()]

        print("===> Sending to GPT")
        try:
            items = ask_gpt_to_analyze(lines)
        except Exception as gpt_error:
            print("GPT Error:", gpt_error)
            return jsonify({'error': 'Failed to process receipt with GPT'}), 500

        print("===> GPT Response Received")

        # Forward to main Node.js server to save
        token = request.headers.get('Authorization', '').replace("Bearer ", "")
        try:
            resp = requests.post(
                'http://localhost:5000/save-items',
                json={'items': items},
                headers={'Authorization': f'Bearer {token}'},
                timeout=5
            )
            if not resp.ok:
                print("Save to DB failed:", resp.text)
        except Exception as save_error:
            print("Error saving to Node server:", save_error)

        print("===> Returning JSON to client")
        return jsonify({'items': items})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5100)
