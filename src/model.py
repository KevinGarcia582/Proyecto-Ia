"""Carga y ejecución de la red entrenada localmente."""
from pathlib import Path
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "modelo_seleccionado.keras"
MAX_WORDS = 12000
MAX_LEN = 250
_MODEL = None
_WORD_INDEX = None

def _load():
    global _MODEL, _WORD_INDEX
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No existe {MODEL_PATH}. Ejecuta primero el notebook de entrenamiento.")
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
        vocab_path = ROOT / "data" / "imdb_word_index.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"No existe {vocab_path}; ejecuta el notebook de entrenamiento para preparar IMDb.")
        _WORD_INDEX = json.loads(vocab_path.read_text(encoding="utf-8"))
    return _MODEL, _WORD_INDEX

def model_status():
    return {"ready": MODEL_PATH.exists(), "name": MODEL_PATH.name if MODEL_PATH.exists() else "Sin entrenar"}

def load_model_resource():
    """Precarga y conserva los pesos/modelo para evitar recargas entre inferencias."""
    return _load()[0]

def encode_review(text: str):
    _, word_index = _load()
    tokens = [1]
    normalized = text.lower().translate(str.maketrans({char: " " for char in string.punctuation}))
    for word in normalized.split():
        raw_idx = word_index.get(word)
        idx = 2 if raw_idx is None else raw_idx + 3
        tokens.append(idx if idx < MAX_WORDS else 2)
    return pad_sequences([tokens], maxlen=MAX_LEN, padding="post", truncating="post")

def predict_text(text: str):
    if not text or not text.strip():
        raise ValueError("Escribe una reseña antes de clasificar.")
    model, _ = _load()
    probability = float(model.predict(encode_review(text), verbose=0)[0][0])
    return {"label":"Positiva" if probability >= 0.5 else "Negativa", "confidence":probability if probability >= 0.5 else 1-probability, "positive_probability":probability}
