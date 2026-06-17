from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re
from .settings import ABSA_MODEL_PATH


class AspectSentimentAnalyzer:
    def __init__(self):
        self.aspect_keywords = {
            "room": {
                "space_condition": [
                    "suite", "suites",
                    "spacious", "compact", "cramped"
                ],
                "bed_comfort": [
                    "bed", "beds",
                    "mattress", "mattresses",
                    "pillow", "pillows",
                    "blanket", "blankets",
                    "linen", "linens",
                    "sheet", "sheets"
                ],
                "bathroom_facility": [
                    "bathroom", "bathrooms",
                    "toilet", "toilets",
                    "sink", "sinks",
                    "bathtub", "bathtubs",
                    "shower", "showers",
                    "washroom"
                ],
                "in_room_facilities": [
                    "air conditioning", "aircon",
                    "tv", "flat panel",
                    "fridge", "microwave",
                    "mini bar"
                ],
                "noise": [
                    "noise", "noisy",
                    "quiet", "soundproof", "soundproofing"
                ]
            },

            "cleanliness": {
                "room_cleanliness": [
                    "clean room", "dirty room",
                    "dust", "dusty",
                    "streaked",
                    "smell in the room", "smelly room",
                    "unclean room"
                ],
                "bathroom_cleanliness": [
                    "dirty bathroom", "dirty toilet",
                    "dirty sink", "dirty shower",
                    "smelly bathroom",
                    "mold", "mould",
                    "stained towel", "stained towels"
                ],
                "restaurant_cleanliness": [
                    "dirty table", "dirty dishes",
                    "unclean restaurant",
                    "sticky table",
                    "unclean plate", "unclean plates"
                ],
                "public_area_cleanliness": [
                    "dirty lobby", "dirty hallway",
                    "dirty elevator",
                    "unclean corridor",
                    "messy public area"
                ]
            },

            "service": {
                "staff_attitude": [
                    "staff",
                    "attendant", "attendants",
                    "friendly staff", "helpful staff",
                    "rude staff", "polite staff",
                    "welcoming staff", "attentive staff"
                ],
                "front_desk": [
                    "front desk", "reception",
                    "check-in", "check in",
                    "check-out", "check out",
                    "concierge", "doormen", "clerk"
                ],
                "housekeeping": [
                    "housekeeping",
                    "cleaner", "cleaners",
                    "maid", "maids",
                    "towel service"
                ],
                "responsiveness": [
                    "slow service", "quick service",
                    "responsive", "unresponsive",
                    "waiting time", "delay", "delayed",
                    "late response"
                ]
            },

            "food": {
                "breakfast": [
                    "breakfast",
                    "buffet breakfast",
                    "continental breakfast",
                    "coffee"
                ],
                "restaurant_food_quality": [
                    "restaurant food",
                    "meal", "meals",
                    "buffet",
                    "dinner"
                ],
                "restaurant_service": [
                    "waiter", "waiters",
                    "waitress", "waitresses",
                    "table service",
                    "restaurant staff"
                ]
            },

            "facilities": {
                "pool": [
                    "pool", "swimming pool",
                    "poolside", "cabanas"
                ],
                "gym": [
                    "gym",
                    "fitness center", "fitness centre"
                ],
                "parking": [
                    "parking",
                    "car park",
                    "garage",
                    "underground parking"
                ],
                "business_facilities": [
                    "business center", "business centre",
                    "conference room", "meeting room"
                ],
                "internet": [
                    "internet", "wifi", "wi-fi"
                ]
            },

            "value": {
                "price_value": [
                    "price", "prices", "priced",
                    "rate", "rates",
                    "cost", "costs",
                    "value",
                    "worth",
                    "overpriced",
                    "affordable"
                ]
            }
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ABSA_MODEL_PATH),
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(ABSA_MODEL_PATH),
            local_files_only=True
        )
        self.model.eval()

        self.labels = ["negative", "neutral", "positive"]

    def _keyword_match(self, keyword: str, text: str) -> bool:
        """
        Match a keyword or phrase using word-boundary-aware regex.
        """
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        return re.search(pattern, text.lower()) is not None

    def _split_clauses(self, text: str) -> list[str]:
        """
        Split review text into smaller clauses for finer-grained analysis.
        """
        pattern = re.compile(
            r",|;|:|\.|!|\?|\bbut\b|\band\b|\bhowever\b|\balthough\b|\bthough\b",
            flags=re.IGNORECASE
        )
        return [clause.strip() for clause in pattern.split(text) if clause.strip()]

    def extract_aspects(self, text: str) -> dict:
        """
        Extract matched aspects and sub-aspects using hierarchical keyword matching.
        """
        matches = {}

        for aspect, sub_aspects in self.aspect_keywords.items():
            aspect_matches = {}

            for sub_aspect, keywords in sub_aspects.items():
                matched_keywords = [
                    keyword for keyword in keywords
                    if self._keyword_match(keyword, text)
                ]

                if matched_keywords:
                    aspect_matches[sub_aspect] = matched_keywords

            if aspect_matches:
                matches[aspect] = aspect_matches

        return matches

    def extract_aspect_clauses(self, text: str, keywords: list[str]) -> list[str]:
        """
        Extract clauses containing any keyword of the target sub-aspect.
        """
        clauses = self._split_clauses(text)
        matched_clauses = []

        for clause in clauses:
            if any(self._keyword_match(keyword, clause) for keyword in keywords):
                matched_clauses.append(clause)

        return matched_clauses

    def _compute_score(self, probs: torch.Tensor) -> float:
        """
        Compute a sentiment score on a 1-5 scale from class probabilities.
        """
        return (
            probs[0] * 1 +
            probs[1] * 3 +
            probs[2] * 5
        ).item()

    def _get_aspect_sentiment(self, aspect_text: str, aspect: str, sub_aspect: str) -> dict:
        """
        Predict sentiment for a specific aspect/sub-aspect text span.
        """
        input_text = f"{aspect}: {aspect_text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)[0]
        score = self._compute_score(probs)

        sorted_probs, _ = torch.sort(probs, descending=True)
        margin = (sorted_probs[0] - sorted_probs[1]).item()

        confidence = round(probs.max().item(), 3)
        score_rounded = round(score, 2)

        return {
            "aspect": aspect,
            "sub_aspect": sub_aspect,
            "text": aspect_text,
            "label": self.labels[probs.argmax().item()],
            "score": score_rounded,
            "confidence": confidence,
            "needs_review": confidence < 0.65 or margin < 0.15 or (2.7 <= score_rounded <= 3.3),
            "probabilities": {
                "negative": round(probs[0].item(), 3),
                "neutral": round(probs[1].item(), 3),
                "positive": round(probs[2].item(), 3)
            }
        }

    def _infer_sub_aspect_sentiments(self, text: str, detected_aspects: dict) -> list[dict]:
        """
        Run clause-level sentiment inference for all detected sub-aspects.
        """
        sub_aspect_results = []

        for aspect, sub_aspects in detected_aspects.items():
            for sub_aspect in sub_aspects.keys():
                keywords = self.aspect_keywords[aspect][sub_aspect]
                matched_clauses = self.extract_aspect_clauses(text, keywords)

                if not matched_clauses:
                    continue

                for clause in matched_clauses:
                    result = self._get_aspect_sentiment(
                        aspect_text=clause,
                        aspect=aspect,
                        sub_aspect=sub_aspect
                    )
                    sub_aspect_results.append(result)

        return sub_aspect_results

    def _aggregate_results(self, results: list[dict]) -> list[dict]:
        """
        Aggregate sub-aspect results into aspect-level results.
        """
        grouped = {}

        for result in results:
            aspect = result["aspect"]

            if aspect not in grouped:
                grouped[aspect] = {
                    "aspect": aspect,
                    "sub_aspects": [],
                    "scores": [],
                    "negative_probs": [],
                    "neutral_probs": [],
                    "positive_probs": []
                }

            grouped[aspect]["sub_aspects"].append(result)
            grouped[aspect]["scores"].append(result["score"])
            grouped[aspect]["negative_probs"].append(result["probabilities"]["negative"])
            grouped[aspect]["neutral_probs"].append(result["probabilities"]["neutral"])
            grouped[aspect]["positive_probs"].append(result["probabilities"]["positive"])

        aggregated = []

        for aspect, data in grouped.items():
            avg_negative = sum(data["negative_probs"]) / len(data["negative_probs"])
            avg_neutral = sum(data["neutral_probs"]) / len(data["neutral_probs"])
            avg_positive = sum(data["positive_probs"]) / len(data["positive_probs"])

            avg_probs = torch.tensor(
                [avg_negative, avg_neutral, avg_positive],
                dtype=torch.float32
            )
            avg_score = self._compute_score(avg_probs)

            sorted_probs, _ = torch.sort(avg_probs, descending=True)
            margin = (sorted_probs[0] - sorted_probs[1]).item()

            confidence = round(float(avg_probs.max().item()), 3)
            score_rounded = round(avg_score, 2)

            aggregated.append({
                "aspect": aspect,
                "label": self.labels[avg_probs.argmax().item()],
                "score": score_rounded,
                "confidence": confidence,
                "needs_review": confidence < 0.65 or margin < 0.15 or (2.7 <= score_rounded <= 3.3),
                "probabilities": {
                    "negative": round(avg_negative, 3),
                    "neutral": round(avg_neutral, 3),
                    "positive": round(avg_positive, 3)
                },
                "sub_aspects": data["sub_aspects"]
            })

        return aggregated

    def analyze(self, text: str) -> dict:
        detected_aspects = self.extract_aspects(text)
        sub_aspect_results = self._infer_sub_aspect_sentiments(text, detected_aspects)
        aspect_results = self._aggregate_results(sub_aspect_results)

        aspect_sentiment = []
        for aspect_item in aspect_results:
            aspect_sentiment.append({
                "aspect": aspect_item["aspect"],
                "label": aspect_item["label"],
                "score": aspect_item["score"],
                "confidence": aspect_item["confidence"],
                "needs_review": aspect_item["needs_review"],
                "sub_aspects": [
                    {
                        "sub_aspect": sub["sub_aspect"],
                        "label": sub["label"],
                        "score": sub["score"],
                        "confidence": sub["confidence"],
                        "text": sub["text"]
                    }
                    for sub in aspect_item["sub_aspects"]
                ]
            })

        return {
            "text": text,
            "aspect_sentiment": aspect_sentiment,
            "needs_review": any(item["needs_review"] for item in aspect_sentiment)
        }