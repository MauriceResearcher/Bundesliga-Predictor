import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd


def send_prediction_email(predictions_df, performances_dict, recipient_email):
    """Verschickt die berechneten Tipps und Validierungs-Metriken aller Modelle als strukturierte HTML-E-Mail via SMTP."""
    # Zugangsdaten für den Mail-Versand aus Umgebungsvariablen laden
    bot_email = os.getenv("BOT_GMAIL")  # Absender (Bot-Mail)
    app_password = os.getenv("GMAIL_PASSWORD")  # Google App-Passwort

    if not bot_email or not app_password:
        print(
            "Fehler: GMAIL oder GMAIL_PASSWORD Umgebungsvariable ist nicht gesetzt."
        )
        return

    spieltag = predictions_df["Spieltag"].iloc[0]
    subject = f"⚽ Bundesliga-Tipps & Analyse – Spieltag {spieltag}"

    # Liste aller enthaltenen Modelle aus den Spalten filtern
    model_names = [
        col.replace("_Tipp", "")
        for col in predictions_df.columns
        if col.endswith("_Tipp")
    ]

    html_sections = []

    for model_name in model_names:
        # 1. Spezifische Spalten für das jeweilige Modell filtern
        model_df = predictions_df[
            [
                "Heimteam",
                "Auswärtsteam",
                f"{model_name}_xG",
                f"{model_name}_Tipp",
            ]
        ].copy()
        model_df.columns = [
            "Heimteam",
            "Auswärtsteam",
            "xG (Heim : Auswärts)",
            "Tipp",
        ]

        # HTML-Tabelle für das Modell generieren
        html_table = model_df.to_html(
            index=False, classes="prediction-table", border=0
        )

        # 2. Performance-Daten extrahieren
        perf = performances_dict.get(model_name, {})
        tr_acc = perf.get("train_acc", 0.0) * 100
        val_acc = perf.get("val_acc", 0.0) * 100
        val_mae = perf.get("val_mae", 0.0)

        # HTML-Block für dieses Modell zusammensetzen
        section_html = f"""
        <div class="model-card">
            <h3>🤖 Modell: {model_name.upper()}</h3>
            {html_table}
            <div class="metrics-box">
                <strong>📊 Modell-Confidence & Validierung (Historische Daten):</strong><br>
                • <b>Train Accuracy (1X2):</b> {tr_acc:.1f}%<br>
                • <b>Validation Accuracy (1X2):</b> {val_acc:.1f}%<br>
                • <b>Durchschnittl. Tor-Abweichung (MAE):</b> {val_mae:.2f} Tore
            </div>
        </div>
        <hr class="divider">
        """
        html_sections.append(section_html)

    # Gesamt-HTML aufbauen
    html_content = f"""
    <html>
      <head>
        <style>
          body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; padding: 20px; }}
          .container {{ max-width: 800px; margin: 0 auto; background: #ffffff; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
          h2 {{ color: #1e293b; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
          h3 {{ color: #2563eb; margin-top: 20px; margin-bottom: 10px; }}
          .prediction-table {{ border-collapse: collapse; width: 100%; margin-bottom: 12px; font-size: 14px; }}
          .prediction-table th, .prediction-table td {{ border: 1px solid #e2e8f0; text-align: left; padding: 10px; }}
          .prediction-table th {{ background-color: #1e293b; color: white; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; }}
          .prediction-table tr:nth-child(even) {{ background-color: #f8fafc; }}
          .metrics-box {{ background-color: #f1f5f9; border-left: 4px solid #3b82f6; padding: 12px; font-size: 13px; line-height: 1.6; border-radius: 0 4px 4px 0; }}
          .divider {{ border: 0; height: 1px; background: #e2e8f0; margin: 25px 0; }}
          .footer {{ font-size: 12px; color: #64748b; text-align: center; margin-top: 20px; }}
        </style>
      </head>
      <body>
        <div class="container">
          <h2>Bundesliga Vorhersagen — Spieltag {spieltag}</h2>
          <p>Hier sind die berechneten Vorhersagen (xG-Erwartungswerte und Tendenzen) deiner KI-Modelle:</p>

          {"".join(html_sections)}

          <div class="footer">
            <i>Automatisch generiert durch deine Multi-Modell Bundesliga Pipeline.</i>
          </div>
        </div>
      </body>
    </html>
    """

    # MIMEMultipart E-Mail Objekt zusammenbauen
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = bot_email
    msg["To"] = recipient_email
    msg.attach(MIMEText(html_content, "html"))

    # Versenden über SSL via Gmail SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(bot_email, app_password)
            server.send_message(msg)
        print(
            f"E-Mail für Spieltag {spieltag} erfolgreich an {recipient_email} versendet!"
        )
    except Exception as e:
        print(f"Fehler beim Versenden der E-Mail: {e}")