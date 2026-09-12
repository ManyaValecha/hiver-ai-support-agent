"""
src/app.py — Local development server.
For production/Vercel deployment, see api/index.py.
"""
import os
import sys

# Allow running directly from src/ directory
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_from_directory
from agent import SupportAgent
from generate_synthetic_data import generate_golden_dataset

# ── Bootstrap: generate dataset if needed ────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_FILE = os.path.join(DATA_DIR, 'AppleSupport_golden_labelled.csv')

if not os.path.exists(DATA_FILE):
    print("Dataset not found — generating fresh golden dataset...")
    os.chdir(os.path.dirname(__file__))   # ensure relative paths work
    generate_golden_dataset(num_samples=300)

print("Loading Hiver Support Agent...")
agent = SupportAgent(DATA_FILE)
print(f"✅ Agent ready — {len(agent.df)} training samples, {len(agent.intents)} intents.")

# ── Flask App ────────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'static')
app = Flask(__name__, static_folder=STATIC_DIR)


@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True) or {}
    user_message = str(data.get('message', '')).strip()

    if not user_message:
        return jsonify({"error": "Field 'message' is required."}), 400
    if len(user_message) > 500:
        return jsonify({"error": "Message exceeds 500 character limit."}), 400

    intent, conf, top3 = agent.classify_intent(user_message)
    action, reason     = agent.route_decision(user_message, intent, conf)
    reply              = agent.draft_reply(user_message, intent, conf)
    similar            = agent.get_similar_cases(user_message, intent, n=3)

    return jsonify({
        "intent":        intent,
        "confidence":    round(conf, 3),
        "top3_intents":  [{"label": l, "score": s} for l, s in top3],
        "action":        action,
        "reason":        reason,
        "reply":         reply,
        "similar_cases": similar,
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status":           "ok",
        "intents":          agent.intents,
        "training_samples": len(agent.df),
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
