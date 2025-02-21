from typing import List, Dict
from google.cloud import translate_v2 as translate

class Translator:
    def __init__(self):
        self.client = translate.Client()

    def translate_batch(self, texts: List[str], target_lang: str) -> List[Dict]:
        """Translate a batch of texts to target language."""
        results = []
        for text in texts:
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            try:
                result = self.client.translate(text, target_language=target_lang)
                results.append(result)
            except Exception as e:
                print(f"Translation error: {e}")
                results.append({
                    "input": text,
                    "translatedText": text,  # Return original text on error
                    "detectedSourceLanguage": "unknown",
                    "error": str(e)
                })
        return results