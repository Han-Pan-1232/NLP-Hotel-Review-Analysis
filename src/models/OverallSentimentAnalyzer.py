from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from .settings import OVERALL_MODEL_PATH


class OverallSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(OVERALL_MODEL_PATH),
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(OVERALL_MODEL_PATH),
            local_files_only=True
        )
        self.model.eval()

    def _compute_score(self, probs: torch.Tensor) -> float:
        """
        Compute a sentiment score on a 1-5 scale from class probabilities.

        Args:
            probs (torch.Tensor): Probability tensor of shape (5,)
                corresponding to 1-star to 5-star classes.

        Returns:
            float: Weighted sentiment score between 1 and 5.
        """
        weights = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        return torch.sum(probs * weights).item()

    def _map_label(self, predicted_class: int) -> str:
        """
        Map star-class prediction to sentiment label.

        Args:
            predicted_class (int): Predicted class index in range [0, 4].

        Returns:
            str: Sentiment label.
        """
        star = predicted_class + 1

        if star <= 2:
            return "negative"
        elif star == 3:
            return "neutral"
        return "positive"

    def analyze(self, text: str) -> dict:
        """
        Perform overall sentiment analysis on a review.

        Args:
            text (str): Input review text.

        Returns:
            dict: Structured overall sentiment result.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)[0]
        predicted_class = torch.argmax(probs).item()
        score = self._compute_score(probs)

        sorted_probs, _ = torch.sort(probs, descending=True)
        margin = (sorted_probs[0] - sorted_probs[1]).item()

        confidence = round(probs.max().item(), 3)
        score_rounded = round(score, 2)

        return {
            "text": text,
            "label": self._map_label(predicted_class),
            "score": score_rounded,
            "confidence": confidence,
            "needs_review": confidence < 0.65 or margin < 0.15 or (2.7 <= score_rounded <= 3.3),
            "probabilities": {
                "1_star": round(probs[0].item(), 3),
                "2_star": round(probs[1].item(), 3),
                "3_star": round(probs[2].item(), 3),
                "4_star": round(probs[3].item(), 3),
                "5_star": round(probs[4].item(), 3)
            }
        }