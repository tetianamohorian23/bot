import os
import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, CallbackContext
from dotenv import load_dotenv
import mysql.connector


# Загрузка переменных окружения из .env
#load_dotenv()
TOKEN = os.getenv("TOKEN")

# Путь к модели
MODEL_PATH = "tetianamohorian/hate_speech_model"

db_config = {
    "host": os.environ.get("MYSQL_ADDON_HOST"),
    "user": os.environ.get("MYSQL_ADDON_USER"),
    "password": os.environ.get("MYSQL_ADDON_PASSWORD"),
    "database": os.environ.get("MYSQL_ADDON_DB")
}


def save_violator(username, message):
    """Сохраняет нарушителя в базу данных"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = "INSERT INTO violators (username, message) VALUES (%s, %s)"
        cursor.execute(query, (username, message))
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        logging.error(f"Ошибка MySQL: {err}")


# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)

# Загрузка модели и токенизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

def classify_text(text):
    """Функция для классификации текста"""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).item()

    return "🛑 Nenávistná reč" if pred == 1 else "✅ OK"

async def check_message(update: Update, context: CallbackContext):
    """Проверяет сообщения в чате и реагирует на токсичные сообщения"""
    message_text = update.message.text
    result = classify_text(message_text)

    if result == "🛑 Nenávistná reč":
        username = update.message.from_user.username or "unknown"
        await update.message.reply_text("⚠️ Upozornenie! Dodržiavajte kultúru komunikácie.")
        await update.message.delete()  # Автоматическое удаление токсичных сообщений
        
        # Логирование токсичного сообщения
        logging.warning(f"Toxická správa od {update.message.from_user.username}: {message_text}")
        save_violator(username, message_text)

async def start(update: Update, context: CallbackContext):
    """Отправляет приветственное сообщение при запуске бота"""
    await update.message.reply_text("Ahoj! Sledujem kultúru komunikácie v chate!")

def main():
    """Запуск бота"""
    app = Application.builder().token(TOKEN).build()

    # Добавление обработчиков команд и сообщений
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_message))

    # Запуск бота
    app.run_polling()

if __name__ == "__main__":
    main()
