"""
Insights agent – Flask HTTP API.

Endpoints:
  POST /start          Start a new conversation for a user
  POST /message        Send a message in an ongoing conversation
  POST /reset          Reset a user's conversation

Request body for /start and /message:
  { "user_id": "u123", "user_name": "Jane" }

/message additionally requires:
  { "user_id": "u123", "user_name": "Jane", "text": "Tell me about Acme Corp" }

Run with:
    python app.py
"""

import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from database.airtable_client import AirtableClient
from agents.insights_agent import InsightsAgent
from agents.state import StateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = AirtableClient()
state_manager = StateManager()
agent = InsightsAgent(db=db, state_manager=state_manager)

app = Flask(__name__)


def _get_body():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "").strip()
    user_name = data.get("user_name", user_id).strip()
    return data, user_id, user_name


@app.post("/start")
def start():
    _, user_id, user_name = _get_body()
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    reply = agent.start_conversation(user_id, user_name)
    return jsonify({"reply": reply})


@app.post("/message")
def message():
    data, user_id, user_name = _get_body()
    text = data.get("text", "").strip()
    if not user_id or not text:
        return jsonify({"error": "user_id and text are required"}), 400

    state = state_manager.get(user_id)
    if state is None:
        reply = agent.start_conversation(user_id, user_name)
    else:
        reply = agent.handle_message(user_id, user_name, text)
    return jsonify({"reply": reply})


@app.post("/reset")
def reset():
    _, user_id, user_name = _get_body()
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    reply = agent.start_conversation(user_id, user_name)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
