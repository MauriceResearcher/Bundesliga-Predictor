import ezgmail


def send_prediction_email(predictions_df, recipient_email):
    """Verschickt die berechneten Tipps als strukturierte HTML-E-Mail via EZGmail."""
    if not ezgmail.LOGGED_IN:
        ezgmail.init()

    matchday = predictions_df["matchday"].iloc[0]
    subject = f"Bundesliga-Tipps für Spieltag {matchday}"

    # HTML-Tabelle generieren
    html_table = predictions_df.to_html(index=False, classes="table")

    html_content = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
          th {{ background-color: #4CAF50; color: white; }}
          tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
      </head>
      <body>
        <h2>Bundesliga Vorhersagen - Spieltag {matchday}</h2>
        <p>Hier sind die aktuellen Wahrscheinlichkeiten für den kommenden Spieltag:</p>
        {html_table}
        <br>
        <p><i>Automatisch generiert durch dein PyTorch Bundesliga-Modell.</i></p>
      </body>
    </html>
    """

    try:
        # KORREKTUR: body nimmt den HTML-String auf, mimeSubtype='html' teilt ezgmail mit,
        # dass es als HTML gerendert werden soll.
        ezgmail.send(
            recipient_email, subject, body=html_content, mimeSubtype="html"
        )
        print(
            f"E-Mail für Spieltag {matchday} erfolgreich via Gmail API versendet!"
        )
    except Exception as e:
        print(f"Fehler beim Versenden der E-Mail: {e}")