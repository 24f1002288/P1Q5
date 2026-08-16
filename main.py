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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")

# Use the exact correct route for wget compliance
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
LOG_URL = f"{RENDER_EXTERNAL_URL}/run.jsonl" 

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
        # JSONL format: one JSON object per line
        f.write(json.dumps(event) + "\n")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    # Updated system prompt to strictly enforce the grading spec
    system_prompt = (
        "You are an expert data analyst. The user's LAST message asks a data-analysis "
        "question and provides the EXACT JSON shape you must reply with. "
        "Work out the answer using your internal knowledge (e.g., MOSPI statistics). "
        "Your final reply MUST be a single, valid JSON object containing exactly two keys: "
        "'answer' (shaped exactly as the user requested) and 'log_url'. "
        "Output ONLY raw JSON. No markdown formatting, no code fences, no explanations."
    )

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[{"role": "system", "content": system_prompt}] + history[-6:],
            temperature=0.1, # Lower temperature for more reliable JSON formatting
        )
        reply_text = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply_text})
    except Exception as e:
        reply_text = '{"answer": "error generating response"}'

    # Robust extraction: Strip markdown if the LLM accidentally adds it
    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        start, end = reply_text.find("{"), reply_text.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(reply_text[start:end + 1])
            except Exception:
                parsed = {"answer": reply_text}
        else:
            parsed = {"answer": reply_text}

    # GRADING SPEC ENFORCEMENT: 
    # The final payload MUST be exactly {"answer": <data>, "log_url": <url>}
    
    # 1. If the LLM forgot the "answer" wrapper and just returned the raw data, wrap it.
    if "answer" not in parsed:
        parsed = {"answer": parsed}

    # 2. Rebuild the final object from scratch to guarantee NO extra hallucinated root keys exist
    final_reply_obj = {
        "answer": parsed["answer"],
        "log_url": LOG_URL
    }

    # Convert to a clean JSON string with no extra spaces/markdown
    final_reply = json.dumps(final_reply_obj)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)

# Initialize the Telegram Application
ptb_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Define the FastAPI Lifespan to run the Telegram bot concurrently
@asynccontextmanager
async def lifespan(app: FastAPI):
    await ptb_app.initialize()
    await ptb_app.start()
    await ptb_app.updater.start_polling()
    print("Telegram bot is polling...")
    yield
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

# Updated route to exactly match the desired filename for wget
@app.get("/run.jsonl")
def serve_log_file():
    """Serves the run.jsonl file. It can be downloaded via wget naturally."""
    if os.path.exists(LOG_FILE):
        return FileResponse(
            path=LOG_FILE, 
            filename="run.jsonl", 
            media_type="application/jsonl+json"
        )
    return {"error": "Log file not found yet. Send a message to the bot first."}
