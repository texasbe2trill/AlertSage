from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib

from .preprocess import clean_description

_VECTORIZER = None
_MODEL = None


def _get_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def load_vectorizer_and_model():
    global _VECTORIZER, _MODEL

    if _VECTORIZER is not None and _MODEL is not None:
        return _VECTORIZER, _MODEL

    models_dir = _get_models_dir()
    vectorizer_path = models_dir / "vectorizer.joblib"
    model_path = models_dir / "enhanced_logreg.joblib"

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found at: {vectorizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    _VECTORIZER = joblib.load(vectorizer_path)
    _MODEL = joblib.load(model_path)

    return _VECTORIZER, _MODEL


def predict_event_type(
    raw_text: str,
    top_k: int = 5,
) -> Tuple[str, Optional[Dict[str, float]]]:
    vectorizer, clf = load_vectorizer_and_model()

    clean = clean_description(raw_text)
    X = vectorizer.transform([clean])

    label = clf.predict(X)[0]

    proba_dict: Optional[Dict[str, float]] = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        proba_dict = dict(zip(clf.classes_, proba))
        if top_k is not None:
            proba_dict = dict(
                sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            )

    return label, proba_dict
