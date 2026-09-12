

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
