"""Interprétation des résultats EDA via Claude (Anthropic)."""

import anthropic


def interpret(summary: str) -> str:
    """Envoie un résumé EDA à Claude et retourne son interprétation."""
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "Tu es un expert en analyse de données. "
                    "Voici le résumé statistique d'un dataset :\n\n"
                    f"{summary}\n\n"
                    "Fournis une interprétation claire : tendances, anomalies, "
                    "valeurs manquantes, et recommandations pour la suite."
                ),
            }
        ],
    )

    return message.content[0].text
