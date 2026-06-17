from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models"

ABSA_MODEL_PATH = MODEL_ROOT / "bert_absa_model"
OVERALL_MODEL_PATH = MODEL_ROOT / "bert_sentiment_model"