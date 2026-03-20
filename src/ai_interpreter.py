"""Interprétation des résultats EDA via Claude (Anthropic)."""

import anthropic


def interpret(summary: str) -> str:
    """Envoie un résumé EDA à Claude et retourne son interprétation."""
    client = anthropic.Anthropic()

    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Tu es un expert en analyse de données. "
                        "Voici le résumé statistique d'un dataset :\n\n"
                        f"{summary}\n\n"
                        "Fournis une interprétation claire : tendances, anomalies, "
                        "valeurs manquantes, doublons, et recommandations pour la suite."
                    ),
                }
            ],
        )
        return message.content[0].text
    except anthropic.AuthenticationError:
        return "Erreur : clé API Anthropic invalide ou absente (ANTHROPIC_API_KEY)."
    except anthropic.RateLimitError:
        return "Erreur : quota API dépassé. Réessayez dans quelques instants."
    except anthropic.APIConnectionError:
        return "Erreur : impossible de joindre l'API Anthropic. Vérifiez votre connexion réseau."
    except anthropic.APIError as e:
        return f"Erreur API Anthropic : {e}"
