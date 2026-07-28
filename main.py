import json
import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# --- Configuration ---
# It is highly recommended to use environment variables for secrets on Render
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")

# Render gives you a public URL (e.g., https://my-bot.onrender.com). 
# We'll use that to construct the log URL automatically if deployed.
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
LOG_URL = f"{RENDER_EXTERNAL_URL}/log" 

LOG_FILE = "run.jsonl"
# -------------------------------------------

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

conversation_history = {}

def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    system_prompt = (
        "You are a careful data analyst. The user's LAST message asks a data-analysis "
        "question and tells you exactly what JSON shape to reply with. Work out the "
        "real answer (use any public data you know, e.g. MOSPI statistics, general "
        "world knowledge, or arithmetic on numbers given in the message). "
        "Reply with ONLY that exact JSON object and absolutely nothing else — no "
        "explanation, no markdown, no code fences, just the raw JSON."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}] + history[-6:],
    )

    reply_text = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply_text})

    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        start, end = reply_text.find("{"), reply_text.rfind("}")
        parsed = json.loads(reply_text[start:end + 1])

    parsed["log_url"] = LOG_URL
    final_reply = json.dumps(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)

# Initialize the Telegram Application
ptb_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Define the FastAPI Lifespan to run the Telegram bot concurrently
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the Telegram bot polling
    await ptb_app.initialize()
    await ptb_app.start()
    await ptb_app.updater.start_polling()
    print("Telegram bot is polling...")
    yield
    # Shutdown: Gracefully stop the bot
    print("Stopping Telegram bot...")
    await ptb_app.updater.stop()
    await ptb_app.stop()
    await ptb_app.shutdown()

# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

# --- FastAPI Routes ---

@app.get("/status")
def health_check():
    """Route for your Cronjob / Keep-alive service to ping."""
    return {"status": "alive", "timestamp": time.time()}

@app.get("/log")
def serve_log_file():
    """Serves the run.jsonl file. It can be downloaded via wget or a browser."""
    if os.path.exists(LOG_FILE):
        return FileResponse(
            path=LOG_FILE, 
            filename="run.jsonl", 
            media_type="application/jsonl+json"
        )
    return {"error": "Log file not found yet. Send a message to the bot first."}
