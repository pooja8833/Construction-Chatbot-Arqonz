from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import requests
import json

# Load .env settings
load_dotenv()

app = Flask(__name__)

# Config from .env
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:instruct")

CONSTRUCTION_KEYWORDS = [
    "cement", "concrete", "steel", "beam", "column", "slab", "brick", "sand",
    "foundation", "plaster", "construction", "contractor", "project",
    "architecture", "engineer", "builder", "cost", "estimate", "bim"
]

def is_construction_query(text: str) -> bool:
    return any(k in text.lower() for k in CONSTRUCTION_KEYWORDS)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "⚠️ Please enter a message."})

    if not is_construction_query(message):
        return jsonify({"reply": "🤖 I'm trained as a construction assistant. Try asking me about cement, concrete, steel, or building projects!."})

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"You are a construction expert. Answer clearly:\n\nUser: {message}\nAssistant:",
            "stream": False
        }

        print("Sending to Ollama:", payload)

        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        print("Response:", resp.status_code)

        if resp.status_code == 200:
            reply = ""
            try:
                # Normal case: single JSON object
                data = resp.json()
                reply = data.get("response") or data.get("output") or ""
            except ValueError:
                # Fallback: handle multiple JSON objects (streamed lines)
                text = resp.text
                parts = [json.loads(line) for line in text.splitlines() if line.strip()]
                reply = " ".join(p.get("response", "") or p.get("output", "") for p in parts)

            return jsonify({"reply": reply.strip() if reply else "⚠️ No reply from model."})
        else:
            return jsonify({"reply": f"⚠️ Ollama error {resp.status_code}: {resp.text}"})
    except Exception as e:
        return jsonify({"reply": f"⚠️ Error contacting Ollama: {str(e)}"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 2000))
    app.run(debug=True, port=port)        