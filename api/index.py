"""
api/index.py — Vercel Serverless entry-point for the Hiver AI Support Agent.

Vercel's Python runtime expects a WSGI-compatible `app` object at module level.
We import the Flask app from src/ and expose it here.
The dataset is bundled in the repo (data/) so no filesystem writes are needed
at runtime — Vercel's /tmp is ephemeral and read-only for the source tree.
"""

import sys
import os

# Make src/ importable from the Vercel function context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# The golden dataset is committed to the repo under data/
# Resolve its absolute path relative to this file so it works on any machine
DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'AppleSupport_golden_labelled.csv'
)

from flask import Flask, request, jsonify, send_from_directory
from agent import SupportAgent  # noqa: E402  (imported after sys.path patch)

# ── Initialise the agent once (module-level, not per-request) ────────────────
_agent = SupportAgent(DATA_PATH)

# ── Flask app ────────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'static')
app = Flask(__name__, static_folder=STATIC_DIR)

# Security headers applied to every response
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Tight CSP: allow fonts from Google, everything else self-hosted
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:;"
    )
    return response


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    """Serve the SPA from static/."""
    if path and os.path.exists(os.path.join(STATIC_DIR, path)):
        return send_from_directory(STATIC_DIR, path)
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    POST /api/chat
    Body: { "message": "<customer query string>" }
    Returns: intent, confidence, top3_intents, action, reason, reply, similar_cases
    """
    # Input validation
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True) or {}
    user_message = str(data.get('message', '')).strip()

    if not user_message:
        return jsonify({"error": "Field 'message' is required and must not be empty."}), 400

    # Hard cap — prevents prompt-injection-style abuse
    if len(user_message) > 500:
        return jsonify({"error": "Message exceeds maximum length of 500 characters."}), 400

    # Run AI pipeline
    intent, conf, top3 = _agent.classify_intent(user_message)
    action, reason  = _agent.route_decision(user_message, intent, conf)
    reply           = _agent.draft_reply(user_message, intent, conf)
    similar         = _agent.get_similar_cases(user_message, intent, n=3)

    return jsonify({
        "intent":        intent,
        "confidence":    round(conf, 3),
        "top3_intents":  [{"label": lbl, "score": sc} for lbl, sc in top3],
        "action":        action,
        "reason":        reason,
        "reply":         reply,
        "similar_cases": similar,
    })


@app.route('/api/health', methods=['GET'])
def health():
    """GET /api/health — liveness probe."""
    return jsonify({
        "status":           "ok",
        "intents":          _agent.intents,
        "training_samples": len(_agent.df),
    })


# ── Local dev entry-point ────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
