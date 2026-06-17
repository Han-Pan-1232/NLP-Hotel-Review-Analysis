from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .OverallSentimentAnalyzer import OverallSentimentAnalyzer
from .AspectSentimentAnalyzer import AspectSentimentAnalyzer
from .settings import ABSA_MODEL_PATH, OVERALL_MODEL_PATH


class ReviewAnalyzer:
    def __init__(self):
        self._ensure_models()
        self.overall_analyzer = OverallSentimentAnalyzer()
        self.aspect_analyzer = AspectSentimentAnalyzer()

    def _ensure_models(self) -> None:
        self._ensure_single_model(
            model_path=ABSA_MODEL_PATH,
            model_name="yangheng/deberta-v3-base-absa-v1.1"
        )
        self._ensure_single_model(
            model_path=OVERALL_MODEL_PATH,
            model_name="nlptown/bert-base-multilingual-uncased-sentiment"
        )

    def _ensure_single_model(self, model_path, model_name: str) -> None:
        required_files = [
            "config.json",
            "tokenizer_config.json"
        ]

        has_required_files = model_path.exists() and all(
            (model_path / file_name).exists()
            for file_name in required_files
        )

        has_weight_file = model_path.exists() and (
            (model_path / "pytorch_model.bin").exists()
            or (model_path / "model.safetensors").exists()
        )

        if has_required_files and has_weight_file:
            return

        print(f"Downloading model: {model_name}")
        model_path.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)

    def analyze(self, text: str) -> dict:
        overall_result = self.overall_analyzer.analyze(text)
        aspect_result = self.aspect_analyzer.analyze(text)

        return {
            "text": text,
            "overall_sentiment": {
                "label": overall_result["label"],
                "score": overall_result["score"],
                "confidence": overall_result["confidence"],
                "needs_review": overall_result["needs_review"]
            },
            "aspect_sentiment": aspect_result["aspect_sentiment"],
            "needs_review": (
                overall_result["needs_review"]
                or aspect_result["needs_review"]
            )
        }