import os
import smtplib
from dotenv import load_dotenv

# Lädt die Variablen aus der .env Datei
load_dotenv()

bot_email = os.getenv("BOT_GMAIL")
app_password = os.getenv("GMAIL_PASSWORD")
recipient_email = os.getenv("GMAIL")

# Leerzeichen aus dem App-Passwort entfernen (wichtige Fehlerquelle!)
if app_password:
    app_password = app_password.replace(" ", "")

print(f"Versuche Login für: {bot_email}")

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(bot_email, app_password)
        print("✅ Login erfolgreich!")

        # Kurze Test-Mail senden
        msg = f"From: {bot_email}\nTo: {recipient_email}\nSubject: Test\n\nLogin hat geklappt!"
        server.sendmail(bot_email, recipient_email, msg)
        print(f"✅ Test-Mail erfolgreich an {recipient_email} gesendet!")

except Exception as e:
    print(f"❌ Fehler: {e}")