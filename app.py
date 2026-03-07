"""
Slack bot entry point for the Insights agent.

Supports:
- /insights slash command to start a new conversation
- DM / app_mention messages to continue a conversation
- App Home shortcut button (optional future surface)

Run with:
    python app.py
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from database.airtable_client import AirtableClient
from agents.insights_agent import InsightsAgent
from agents.state import StateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialise shared services
# ---------------------------------------------------------------------------
db = AirtableClient()
state_manager = StateManager()
insights_agent = InsightsAgent(db=db, state_manager=state_manager)

# ---------------------------------------------------------------------------
# Slack app
# ---------------------------------------------------------------------------
app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)


def _get_user_name(client, user_id: str) -> str:
    """Resolve a Slack user ID to a display name."""
    try:
        result = client.users_info(user=user_id)
        profile = result["user"].get("profile", {})
        return profile.get("display_name") or profile.get("real_name") or user_id
    except Exception:
        return user_id


# ---------------------------------------------------------------------------
# /insights slash command – start a fresh conversation
# ---------------------------------------------------------------------------
@app.command("/insights")
def handle_insights_command(ack, body, client, respond):
    ack()
    user_id = body["user_id"]
    user_name = body.get("user_name") or _get_user_name(client, user_id)

    try:
        reply = insights_agent.start_conversation(user_id, user_name)
    except Exception as e:
        logger.exception("Error starting Insights conversation")
        reply = "Sorry, something went wrong starting the Insights agent. Please try again."

    respond(text=reply, response_type="ephemeral")


# ---------------------------------------------------------------------------
# DM messages – continue an existing conversation
# ---------------------------------------------------------------------------
@app.event("message")
def handle_dm_message(event, client, say):
    # Only handle DMs (channel_type == "im") and ignore bot messages
    if event.get("channel_type") != "im":
        return
    if event.get("subtype") or event.get("bot_id"):
        return

    user_id = event["user"]
    user_name = _get_user_name(client, user_id)
    user_text = event.get("text", "").strip()

    if not user_text:
        return

    # "restart" / "new" shortcuts
    if user_text.lower() in {"restart", "new", "/insights", "start over", "reset"}:
        reply = insights_agent.start_conversation(user_id, user_name)
    else:
        try:
            # Ensure there is an active conversation; start one if not
            state = state_manager.get(user_id)
            if state is None:
                reply = insights_agent.start_conversation(user_id, user_name)
            else:
                reply = insights_agent.handle_message(user_id, user_name, user_text)
        except Exception:
            logger.exception("Error handling DM message")
            reply = "Sorry, I ran into an error. Try typing *restart* to begin again."

    say(text=reply)


# ---------------------------------------------------------------------------
# @mentions in channels – same flow as DM
# ---------------------------------------------------------------------------
@app.event("app_mention")
def handle_mention(event, client, say):
    user_id = event["user"]
    user_name = _get_user_name(client, user_id)
    # Strip the bot mention from the text
    raw_text = event.get("text", "")
    user_text = raw_text.split(">", 1)[-1].strip() if ">" in raw_text else raw_text.strip()

    if not user_text or user_text.lower() in {"help", "hi", "hello", "start"}:
        reply = insights_agent.start_conversation(user_id, user_name)
    else:
        try:
            state = state_manager.get(user_id)
            if state is None:
                reply = insights_agent.start_conversation(user_id, user_name)
            else:
                reply = insights_agent.handle_message(user_id, user_name, user_text)
        except Exception:
            logger.exception("Error handling mention")
            reply = "Sorry, I ran into an error. Try mentioning me again or use the **/insights** command."

    say(text=reply, thread_ts=event.get("ts"))


# ---------------------------------------------------------------------------
# App Home tab – quick-start button
# ---------------------------------------------------------------------------
@app.event("app_home_opened")
def handle_app_home(event, client):
    user_id = event["user"]
    client.views_publish(
        user_id=user_id,
        view={
            "type": "home",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            "*Welcome to the Insights Agent* :mag:\n\n"
                            "Get a quick briefing on any company or role in our database, "
                            "or contribute what you've learned from conversations and interviews."
                        ),
                    },
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Click below to start a new Insights conversation:"},
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Start Insights"},
                        "action_id": "start_insights_home",
                        "style": "primary",
                    },
                },
            ],
        },
    )


@app.action("start_insights_home")
def handle_start_insights_home(ack, body, client):
    ack()
    user_id = body["user"]["id"]
    user_name = body["user"].get("username") or _get_user_name(client, user_id)

    reply = insights_agent.start_conversation(user_id, user_name)

    # Open a DM with the user and send the greeting there
    dm = client.conversations_open(users=user_id)
    channel_id = dm["channel"]["id"]
    client.chat_postMessage(channel=channel_id, text=reply)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    logger.info("Insights agent is running...")
    handler.start()
