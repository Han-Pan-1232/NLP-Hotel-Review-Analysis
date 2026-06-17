
import asyncio
import concurrent.futures
import json
import os
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Annotated, Dict, List, Literal, Optional
from dotenv import load_dotenv
from fastapi import (BackgroundTasks, Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status,)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prompts import (get_reply_generation_system_prompt, get_reply_generation_user_prompt,)

load_dotenv()


# --- 1. Models

# User
class UserBase(BaseModel):
    username: str
    email: Optional[str] = None

class UserCreate(UserBase):
    role: Literal["hotel_user", "admin", "guest"] = "hotel_user"

class UserInDB(UserBase):
    id: str
    role: Literal["hotel_user", "admin", "guest"]
    hotel_ids: List[str] = Field(default_factory=list)


# Hotel
class HotelBase(BaseModel):
    name: str
    location: str
    description: str

class HotelInDB(HotelBase):
    id: str
    owner_id: str


# Review
class ReviewBase(BaseModel):
    hotel_id: str
    platform_source: str
    reviewer_name: str
    rating: int = Field(..., ge=1, le=5)
    text: str
    review_date: str

class ReviewInDB(ReviewBase):
    id: str
    status: Literal["pending", "processing", "processed", "ai_processing_failed"] = "pending"
    overall_sentiment: Optional[Literal["positive", "neutral", "negative"]] = None
    aspects: Optional[List[Dict[str, str]]] = None
    processed_at: Optional[datetime] = None

class AIResponseDraft(BaseModel):
    id: str
    review_id: str
    draft_text: str
    style: str
    generated_at: str


# Batch processing
class BatchReviewSubmit(BaseModel):
    reviews: List[ReviewBase] = Field(..., min_length=1, max_length=200)

class BatchSubmitResponse(BaseModel):
    batch_id: str
    submitted_count: int
    review_ids: List[str]
    message: str

class BatchStatus(BaseModel):
    batch_id: str
    total: int
    pending: int
    processing: int
    processed: int
    failed: int
    progress_percent: float
    is_complete: bool


# Staff reply
class ReplyStatus(BaseModel):
    review_id: str
    selected_draft_id: Optional[str] = None
    final_text: Optional[str] = None
    state: Literal["draft_pending", "selected", "approved", "rejected", "sent"] = "draft_pending"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    sent_at: Optional[str] = None
    edited: bool = False

class SelectDraftRequest(BaseModel):
    draft_id: str
    edited_text: Optional[str] = None

class ApproveRequest(BaseModel):
    approve: bool = True
    rejection_reason: Optional[str] = None

class GenerateDraftRequest(BaseModel):
    style: Literal["formal", "friendly", "empathetic"]


# Analytics
class SentimentDistribution(BaseModel):
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    total: int = 0

class AspectSummary(BaseModel):
    aspect: str
    count: int
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0

class PlatformReviewCount(BaseModel):
    platform: str
    count: int

class SentimentTrendPoint(BaseModel):
    date: str
    positive: int = 0
    negative: int = 0
    neutral: int = 0


# Dashboard
class HotelKPI(BaseModel):
    hotel_id: str
    hotel_name: str
    total_reviews: int = 0
    processed_reviews: int = 0
    avg_rating: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    positive_rate: float = 0.0
    top_positive_aspect: Optional[str] = None
    top_negative_aspect: Optional[str] = None
    response_draft_coverage: float = 0.0

class DashboardOverview(BaseModel):
    total_reviews: int = 0
    processed_reviews: int = 0
    pending_reviews: int = 0
    failed_reviews: int = 0
    avg_rating: float = 0.0
    sentiment_distribution: SentimentDistribution = Field(default_factory=SentimentDistribution)
    top_negative_aspects: List[AspectSummary] = Field(default_factory=list)
    platform_breakdown: List[PlatformReviewCount] = Field(default_factory=list)
    hotel_kpis: List[HotelKPI] = Field(default_factory=list)
    recent_trend: List[SentimentTrendPoint] = Field(default_factory=list)

class HotelComparison(BaseModel):
    hotels: List[HotelKPI]
    compared_at: str

class TopIssue(BaseModel):
    rank: int
    aspect: str
    negative_count: int
    negative_rate: float
    affected_hotel_ids: List[str] = Field(default_factory=list)

class FeaturedReview(BaseModel):
    review_id: str
    hotel_id: str
    reviewer_name: str
    rating: int
    text: str
    review_date: str
    platform_source: str
    aspects: Optional[List[Dict[str, str]]] = None
    quality_score: float

class RoutedIssue(BaseModel):
    review_id: str
    hotel_id: str
    aspect: str
    department: str
    review_excerpt: str
    rating: int
    review_date: str
    reply_state: str = "draft_pending"

class DepartmentWorkload(BaseModel):
    department: str
    open_issue_count: int
    aspects: List[str]
    hotel_ids: List[str]

# --- 2. In-memory database and constants

db_users: Dict[str, Dict] = {}
db_hotels: Dict[str, Dict] = {}
db_reviews: Dict[str, Dict] = {}
db_ai_response_drafts: Dict[str, List[Dict]] = {}
db_batches: Dict[str, List[str]] = {}
db_reply_status: Dict[str, Dict] = {}

# Maps aspect names from the BERT ReviewAnalyzer to the responsible operational department.
ASPECT_TO_DEPARTMENT: Dict[str, str] = {
    "room": "housekeeping",
    "cleanliness": "housekeeping",
    "service": "guest_relations",
    "food": "dining",
    "facilities": "facilities",
    "value": "management",}

# --- 3. Data store
# Data is stored in a single JSON file next to main.py.
DATA_FILE = Path(os.getenv("DATA_FILE", str(Path(__file__).parent / "data.json")))
def save_db() -> None:
    """Persist all in-memory state to disk. Called after every write."""
    payload = {
        "users": db_users,
        "hotels": db_hotels,
        "reviews": db_reviews,
        "ai_response_drafts": db_ai_response_drafts,
        "batches": db_batches,
        "reply_status": db_reply_status,
    }
    DATA_FILE.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

def load_db() -> None:
    """Restore all in-memory state from disk on startup."""
    if not DATA_FILE.exists():
        return
    try:
        payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    db_users.update(payload.get("users", {}))
    db_hotels.update(payload.get("hotels", {}))
    db_reviews.update(payload.get("reviews", {}))
    db_ai_response_drafts.update(payload.get("ai_response_drafts", {}))
    db_batches.update(payload.get("batches", {}))
    db_reply_status.update(payload.get("reply_status", {}))

# --- 4. App
app = FastAPI(title="HOTALAR",)

# CORS — once deployed, the Streamlit frontend will run on a different
# host from this API, so cross-origin requests would otherwise be blocked
# by the browser. ALLOWED_ORIGINS can be a comma-separated list, defaults
# to "*" so local dev just works.
_allowed = os.getenv("ALLOWED_ORIGINS", "*").strip()
if _allowed == "*":
    _allowed_list = ["*"]
else:
    _allowed_list = [o.strip() for o in _allowed.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_current_user(
    x_user_id: Annotated[str, Header(description="User ID from POST /users/")],
) -> UserInDB:
    """Identify the caller by the X-User-Id header. No password, no token — demo only."""
    user = db_users.get(x_user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown user_id. Register via POST /users/ first.",
        )
    return UserInDB(**user)


async def check_hotel_owner(
    hotel_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
) -> Dict:
    hotel = db_hotels.get(hotel_id)
    if not hotel:
        raise HTTPException(status_code=404, detail="Hotel not found")
    if hotel["owner_id"] == current_user.id or hotel_id in current_user.hotel_ids:
        return hotel
    raise HTTPException(status_code=403, detail="Not authorized to access this hotel")

# --- 5. Parallel processing
process_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

# max_workers=2
@app.on_event("startup")
async def startup_event():
    global process_executor
    load_db()
    workers = int(os.getenv("MAX_WORKERS", "2"))
    process_executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)

@app.on_event("shutdown")
async def shutdown_event():
    if process_executor:
        process_executor.shutdown(wait=True)

# --- 6. WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        payload = json.dumps(message, default=str)
        for ws in list(self.active_connections):
            try:
                await ws.send_text(payload)
            except Exception:
                self.disconnect(ws)
manager = ConnectionManager()

# --- 7. NLP
# Using the team’s BERT sentiment and aspect analysis.
def perform_local_nlp_analysis(review_text: str) -> Dict:
    from review_analyzer import analyze_review
    return analyze_review(review_text)

# Generate 3 reply drafts via OpenAI's chat completion API.
#
# Set OPENAI_API_KEY in .env to enable real LLM calls. If the key is
# missing OR the call fails (network, rate limit, malformed JSON, etc.),
# we fall back to deterministic English templates so the demo never
# breaks just because the API is down.
async def generate_reply_drafts_from_openai(
    review_text: str, overall_sentiment: str, aspects: Optional[List[Dict]],
) -> List[Dict]:
    system_prompt = get_reply_generation_system_prompt(overall_sentiment, aspects or [])
    user_prompt = get_reply_generation_user_prompt(review_text)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            return await _call_openai_for_drafts(system_prompt, user_prompt, api_key)
        except Exception as e:
            print(f"[OpenAI] Falling back to template drafts: {e}")

    return _generate_template_drafts(overall_sentiment, aspects)


async def _call_openai_for_drafts(
    system_prompt: str, user_prompt: str, api_key: str,
) -> List[Dict]:
    """
    Async OpenAI call. Validates the response shape strictly so
    upstream code can rely on `[{draft_text, style}, ...]`.

    Async here matters: it lets us run the call in the FastAPI event
    loop instead of in a worker subprocess, which is critical on Windows
    where ProcessPoolExecutor children don't inherit the .env-loaded
    environment variables.
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "[]"

    # When response_format is json_object the model wraps the array in an
    # object — handle both shapes defensively.
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        for key in ("drafts", "replies", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            for value in parsed.values():
                if isinstance(value, list):
                    parsed = value
                    break

    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"OpenAI response was not a usable list: {raw[:200]}")

    valid_styles = {"formal", "friendly", "empathetic"}
    cleaned = []
    for d in parsed:
        if not isinstance(d, dict):
            continue
        text = d.get("draft_text") or d.get("text")
        style = d.get("style")
        if text and style in valid_styles:
            cleaned.append({"draft_text": text.strip(), "style": style})

    if not cleaned:
        raise ValueError("OpenAI response had no valid drafts")
    return cleaned


def _generate_template_drafts(
    overall_sentiment: str, aspects: Optional[List[Dict]],
) -> List[Dict]:
    """Deterministic fallback when no OPENAI_API_KEY is configured."""
    if overall_sentiment == "positive":
        opening = "Thank you so much for your wonderful review! We're delighted you enjoyed your stay."
    elif overall_sentiment == "negative":
        opening = "Thank you for taking the time to share your feedback. We sincerely apologise for the experience you described."
    else:
        opening = "Thank you for your feedback — we appreciate you sharing your thoughts with us."

    if aspects:
        parts = [f"{a['aspect']} ({a['sentiment']})" for a in aspects]
        aspect_line = f" We've noted your comments on {', '.join(parts)}."
    else:
        aspect_line = ""

    return [
        {
            "draft_text": (
                f"{opening}{aspect_line} We greatly value your patronage and "
                "look forward to welcoming you back to our hotel."
            ),
            "style": "formal",
        },
        {
            "draft_text": (
                f"{opening}{aspect_line} It would be lovely to see you again soon — "
                "safe travels until then!"
            ),
            "style": "friendly",
        },
        {
            "draft_text": (
                f"{opening}{aspect_line} Your experience matters deeply to us, "
                "and we're committed to learning from your feedback to improve."
            ),
            "style": "empathetic",
        },
    ]


async def generate_single_draft(
    review_text: str, overall_sentiment: str,
    aspects: Optional[List[Dict]], style: str,
) -> Dict:
    """
    Generate a single reply draft of the requested style.

    Note: currently this internally generates all 3 styles via the LLM and
    picks the one we want — that's still cheaper than calling the LLM
    three times across the whole flow, because once one style is generated
    the others get cached on disk and never regenerated.
    """
    all_drafts = await generate_reply_drafts_from_openai(review_text, overall_sentiment, aspects)
    matching = next((d for d in all_drafts if d["style"] == style), None)
    if not matching:
        raise ValueError(f"Unsupported style: {style}")
    return matching

def _sync_perform_ai_analysis_logic(review_id: str, review_text: str) -> Dict:
    """
    NLP only. Reply drafts are generated lazily per-style via the
    /reviews/{id}/generate_draft endpoint, so we don't burn LLM tokens
    on drafts the user may never look at.
    """
    nlp = perform_local_nlp_analysis(review_text)
    return {
        "overall_sentiment": nlp["overall_sentiment"],
        "aspects": nlp["aspects"],
    }

# Process one review
async def _perform_ai_analysis_for_review(review_id: str, batch_id: Optional[str] = None):
    review = db_reviews.get(review_id)
    if not review:
        return

    review["status"] = "processing"
    await manager.broadcast({"type": "review_update", "review_id": review_id, "status": "processing"})

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            process_executor,
            partial(_sync_perform_ai_analysis_logic, review_id, review["text"]),)
        review["overall_sentiment"] = result["overall_sentiment"]
        review["aspects"] = result["aspects"]
        review["status"] = "processed"
        review["processed_at"] = datetime.now(timezone.utc)
        # No drafts generated here — keep the bucket initialized but empty.
        db_ai_response_drafts.setdefault(review_id, [])
        save_db()
        await manager.broadcast({
            "type": "review_update", "review_id": review_id,
            "status": "processed", "sentiment": review["overall_sentiment"],})
    except Exception as e:
        review["status"] = "ai_processing_failed"
        save_db()
        await manager.broadcast({
            "type": "review_update", "review_id": review_id,
            "status": "ai_processing_failed", "error": str(e),})

    if batch_id:
        await _broadcast_batch_progress(batch_id)

async def _broadcast_batch_progress(batch_id: str):
    review_ids = db_batches.get(batch_id, [])
    counts = {"pending": 0, "processing": 0, "processed": 0, "ai_processing_failed": 0}
    for rid in review_ids:
        s = db_reviews.get(rid, {}).get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
    total = len(review_ids)
    done = counts["processed"] + counts["ai_processing_failed"]
    await manager.broadcast({
        "type": "batch_progress", "batch_id": batch_id, "total": total,
        "pending": counts["pending"], "processing": counts["processing"],
        "processed": counts["processed"], "failed": counts["ai_processing_failed"],
        "progress_percent": round(done / total * 100, 1) if total else 0.0,
        "is_complete": done == total,})

# Run all reviews in a batch
async def _perform_batch_ai_analysis(batch_id: str, review_ids: List[str]):
    await asyncio.gather(
        *[_perform_ai_analysis_for_review(rid, batch_id) for rid in review_ids],
        return_exceptions=True,)

def _get_accessible_hotel_ids(current_user: UserInDB) -> set:
    if current_user.role == "admin":
        return set(db_hotels.keys())
    ids = set(current_user.hotel_ids)
    for hid, h in db_hotels.items():
        if h["owner_id"] == current_user.id:
            ids.add(hid)
    return ids

def _filter_reviews(
    current_user: UserInDB,
    hotel_id: Optional[str] = None,
    platform_source: Optional[str] = None,
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None,
    overall_sentiment: Optional[Literal["positive", "negative", "neutral"]] = None,
    status_filter: Literal["pending", "processing", "processed", "ai_processing_failed", "all"] = "processed",
) -> List[Dict]:
    accessible = _get_accessible_hotel_ids(current_user)
    if hotel_id:
        if hotel_id not in accessible:
            return []
        accessible = {hotel_id}

    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1) if end_date_str else None
    out = []
    for r in db_reviews.values():
        if r["hotel_id"] not in accessible:
            continue
        if status_filter != "all" and r.get("status") != status_filter:
            continue
        if platform_source and r.get("platform_source") != platform_source:
            continue
        if overall_sentiment and r.get("overall_sentiment") != overall_sentiment:
            continue
        if start_dt or end_dt:
            try:
                rd = datetime.strptime(r["review_date"], "%Y-%m-%d")
            except (ValueError, KeyError):
                continue
            if start_dt and rd < start_dt:
                continue
            if end_dt and rd >= end_dt:
                continue
        out.append(r)
    return out

def _compute_hotel_kpi(hotel_id: str, hotel_name: str) -> HotelKPI:
    all_r = [r for r in db_reviews.values() if r["hotel_id"] == hotel_id]
    proc = [r for r in all_r if r.get("status") == "processed"]
    ratings = [r["rating"] for r in all_r if r.get("rating")]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

    pos = sum(1 for r in proc if r.get("overall_sentiment") == "positive")
    neg = sum(1 for r in proc if r.get("overall_sentiment") == "negative")
    neu = len(proc) - pos - neg
    pos_rate = round(pos / len(proc) * 100, 1) if proc else 0.0

    aspect_pos: Dict[str, int] = {}
    aspect_neg: Dict[str, int] = {}
    for r in proc:
        s = r.get("overall_sentiment", "neutral")
        for entry in (r.get("aspects") or []):
            a = entry["aspect"]
            if s == "positive":
                aspect_pos[a] = aspect_pos.get(a, 0) + 1
            elif s == "negative":
                aspect_neg[a] = aspect_neg.get(a, 0) + 1
    top_pos = max(aspect_pos, key=aspect_pos.get, default=None) if aspect_pos else None
    top_neg = max(aspect_neg, key=aspect_neg.get, default=None) if aspect_neg else None

    with_drafts = sum(
        1 for rid, r in db_reviews.items()
        if r["hotel_id"] == hotel_id and r.get("status") == "processed" and rid in db_ai_response_drafts)
    draft_coverage = round(with_drafts / len(proc) * 100, 1) if proc else 0.0

    return HotelKPI(
        hotel_id=hotel_id, hotel_name=hotel_name,
        total_reviews=len(all_r), processed_reviews=len(proc),
        avg_rating=avg_rating,
        positive_count=pos, negative_count=neg, neutral_count=neu,
        positive_rate=pos_rate,
        top_positive_aspect=top_pos, top_negative_aspect=top_neg,
        response_draft_coverage=draft_coverage,)

def _get_or_init_reply_status(review_id: str) -> Dict:
    if review_id not in db_reply_status:
        db_reply_status[review_id] = ReplyStatus(review_id=review_id).model_dump()
    return db_reply_status[review_id]

# --- 8. Users
@app.post("/users/", response_model=UserInDB, status_code=status.HTTP_201_CREATED, summary="Register a new user")
# Register a new user.
async def create_user(user: UserCreate):
    if user.username in [u["username"] for u in db_users.values()]:
        raise HTTPException(status_code=400, detail="Username already registered")
    new_user = UserInDB(
        id=str(uuid.uuid4()),
        username=user.username,
        email=user.email,
        role=user.role,)
    db_users[new_user.id] = new_user.model_dump()
    save_db()
    return new_user

@app.get("/users/me", response_model=UserInDB, summary="Get Current User")
async def get_current_active_user(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    return current_user

@app.get("/users/{user_id}", response_model=UserInDB, summary="Get User by ID")
async def get_user(user_id: str, current_user: Annotated[UserInDB, Depends(get_current_user)]):
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view this user's profile")
    user = db_users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInDB(**user)

# --- 9. Hotels
@app.post("/hotels/", response_model=HotelInDB, status_code=status.HTTP_201_CREATED, summary="Create Hotel")
async def create_hotel(
    hotel: HotelBase,
    current_user: Annotated[UserInDB, Depends(get_current_user)],):
    if current_user.role not in ("hotel_user", "admin"):
        raise HTTPException(status_code=403, detail="Only hotel users or admin can create hotels")
    new_id = str(uuid.uuid4())
    new_hotel = HotelInDB(id=new_id, owner_id=current_user.id, **hotel.model_dump())
    db_hotels[new_id] = new_hotel.model_dump()
    db_users[current_user.id]["hotel_ids"].append(new_id)
    save_db()
    return new_hotel

@app.get("/hotels/", response_model=List[HotelInDB], summary="List Hotels")
async def list_hotels(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    if current_user.role == "admin":
        return [HotelInDB(**h) for h in db_hotels.values()]
    accessible = _get_accessible_hotel_ids(current_user)
    return [HotelInDB(**db_hotels[hid]) for hid in accessible if hid in db_hotels]

@app.get("/hotels/{hotel_id}", response_model=HotelInDB, summary="Get Hotel by ID")
async def get_hotel(hotel_data: Annotated[Dict, Depends(check_hotel_owner)]):
    return HotelInDB(**hotel_data)

# --- 10. Reviews
@app.post(
    "/reviews/",
    response_model=ReviewInDB,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Review for Processing",
    tags=["Reviews"],)

async def submit_review(
    review: ReviewBase,
    background_tasks: BackgroundTasks,
    hotel_data: Annotated[Dict, Depends(check_hotel_owner)],):

    if review.hotel_id != hotel_data["id"]:
        raise HTTPException(status_code=403, detail="Review hotel_id does not match accessible hotel.")
    new_id = str(uuid.uuid4())
    new_review = ReviewInDB(id=new_id, **review.model_dump())
    db_reviews[new_id] = new_review.model_dump()
    save_db()
    background_tasks.add_task(_perform_ai_analysis_for_review, new_id)
    return new_review

@app.post(
    "/reviews/batch",
    response_model=BatchSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a dataset of reviews for PARALLEL AI processing",
    tags=["Reviews"],)

async def submit_reviews_batch(
    payload: BatchReviewSubmit,
    background_tasks: BackgroundTasks,
    current_user: Annotated[UserInDB, Depends(get_current_user)],):
    accessible = _get_accessible_hotel_ids(current_user)
    invalid = list({r.hotel_id for r in payload.reviews if r.hotel_id not in accessible})
    if invalid:
        raise HTTPException(status_code=403, detail=f"Not authorized for hotel(s): {invalid}")

    batch_id = str(uuid.uuid4())
    review_ids: List[str] = []
    for review in payload.reviews:
        new_id = str(uuid.uuid4())
        db_reviews[new_id] = ReviewInDB(id=new_id, **review.model_dump()).model_dump()
        review_ids.append(new_id)
    db_batches[batch_id] = review_ids
    save_db()
    background_tasks.add_task(_perform_batch_ai_analysis, batch_id, review_ids)
    return BatchSubmitResponse(
        batch_id=batch_id,
        submitted_count=len(review_ids),
        review_ids=review_ids,
        message=f"{len(review_ids)} reviews queued. Track via /reviews/batch/{batch_id} or WebSocket.",)

@app.get(
    "/reviews/batch/{batch_id}",
    response_model=BatchStatus,
    summary="Get processing status of a batch",
    tags=["Reviews"],)

async def get_batch_status(
    batch_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],):
    review_ids = db_batches.get(batch_id)
    if review_ids is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    counts = {"pending": 0, "processing": 0, "processed": 0, "ai_processing_failed": 0}
    for rid in review_ids:
        s = db_reviews.get(rid, {}).get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
    total = len(review_ids)
    done = counts["processed"] + counts["ai_processing_failed"]
    return BatchStatus(
        batch_id=batch_id, total=total,
        pending=counts["pending"], processing=counts["processing"],
        processed=counts["processed"], failed=counts["ai_processing_failed"],
        progress_percent=round(done / total * 100, 1) if total else 0.0,
        is_complete=done == total,)

@app.get("/reviews/{review_id}", response_model=ReviewInDB, summary="Get Review Details")
async def get_review(review_id: str, current_user: Annotated[UserInDB, Depends(get_current_user)]):
    review = db_reviews.get(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    await check_hotel_owner(review["hotel_id"], current_user)
    return ReviewInDB(**review)

@app.get(
    "/reviews/hotel/{hotel_id}",
    response_model=List[ReviewInDB],
    summary="List Reviews for a Specific Hotel",)
async def list_reviews_for_hotel(
    hotel_id: str,
    hotel_data: Annotated[Dict, Depends(check_hotel_owner)],):
    return [ReviewInDB(**r) for r in db_reviews.values() if r["hotel_id"] == hotel_id]

@app.get(
    "/reviews/{review_id}/drafts",
    response_model=List[AIResponseDraft],
    summary="Get AI Reply Drafts for a Review",)
async def get_review_drafts(
    review_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],):
    review = db_reviews.get(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    await check_hotel_owner(review["hotel_id"], current_user)
    return [AIResponseDraft(**d) for d in db_ai_response_drafts.get(review_id, [])]


@app.post(
    "/reviews/{review_id}/generate_draft",
    response_model=AIResponseDraft,
    summary="Generate a single reply draft of the requested style on demand",)
async def generate_review_draft(
    review_id: str,
    payload: GenerateDraftRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """
    Lazily generate one draft of the requested style. Used by the frontend
    when the user clicks 'Generate' on a specific tone tab — keeps LLM
    token spend proportional to actual user demand.

    Each call always triggers a fresh LLM generation and overwrites any
    previously cached draft of the same style. Without this, an older
    template-fallback draft (or any earlier generation) would stick
    around forever and the user would never see fresh model output.
    """
    review = db_reviews.get(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    await check_hotel_owner(review["hotel_id"], current_user)
    if review.get("status") != "processed":
        raise HTTPException(
            status_code=400,
            detail="Review must finish AI processing before drafts can be generated",
        )

    # OpenAI is IO-bound (network call), so we await it directly in the
    # event loop rather than dispatching to a worker process. This is
    # faster *and* it preserves environment variables — ProcessPoolExecutor
    # children on Windows don't inherit OPENAI_API_KEY from .env.
    draft = await generate_single_draft(
        review["text"],
        review.get("overall_sentiment") or "neutral",
        review.get("aspects"),
        payload.style,
    )

    new_draft = AIResponseDraft(
        id=str(uuid.uuid4()),
        review_id=review_id,
        draft_text=draft["draft_text"],
        style=draft["style"],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    # Replace the existing draft of this style if one was cached, so the
    # frontend (and any later 'Reply' click) sees the freshest text.
    existing = db_ai_response_drafts.setdefault(review_id, [])
    db_ai_response_drafts[review_id] = [
        d for d in existing if d["style"] != payload.style
    ] + [new_draft.model_dump()]
    save_db()
    return new_draft

# --- 11. Staff reply workflow
@app.post(
    "/reviews/{review_id}/select_draft",
    response_model=ReplyStatus,
    summary="[Staff Workflow] Select (and optionally edit) one of the AI drafts",
    tags=["Staff Review"],)
async def select_draft(
    review_id: str,
    payload: SelectDraftRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)],):
    review = db_reviews.get(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    await check_hotel_owner(review["hotel_id"], current_user)

    drafts = db_ai_response_drafts.get(review_id, [])
    matching = next((d for d in drafts if d["id"] == payload.draft_id), None)
    if not matching:
        raise HTTPException(status_code=404, detail="Draft not found for this review")

    final_text = payload.edited_text or matching["draft_text"]
    edited = bool(payload.edited_text and payload.edited_text.strip() != matching["draft_text"].strip())

    s = _get_or_init_reply_status(review_id)
    s.update({
        "selected_draft_id": payload.draft_id,
        "final_text": final_text,
        "state": "selected",
        "reviewed_by": current_user.id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "edited": edited,
        "rejection_reason": None,})
    save_db()
    return ReplyStatus(**s)

@app.post(
    "/reviews/{review_id}/approve",
    response_model=ReplyStatus,
    summary="[Staff Workflow] Approve or reject the selected reply",
    tags=["Staff Review"],)
async def approve_reply(
    review_id: str,
    payload: ApproveRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)],):
    review = db_reviews.get(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    await check_hotel_owner(review["hotel_id"], current_user)

    s = _get_or_init_reply_status(review_id)
    if s["state"] not in ("selected", "rejected"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve from state '{s['state']}'. Select a draft first.",
        )
    if not payload.approve and not payload.rejection_reason:
        raise HTTPException(status_code=400, detail="rejection_reason is required when approve=False")

    s.update({
        "state": "approved" if payload.approve else "rejected",
        "reviewed_by": current_user.id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "rejection_reason": None if payload.approve else payload.rejection_reason,})
    save_db()
    return ReplyStatus(**s)

@app.post(
    "/reviews/{review_id}/mark_sent",
    response_model=ReplyStatus,
    summary="[Staff Workflow] Mark approved reply as sent / published",
    tags=["Staff Review"],)
async def mark_reply_sent(
    review_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],):
    review = db_reviews.get(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    await check_hotel_owner(review["hotel_id"], current_user)

    s = _get_or_init_reply_status(review_id)
    if s["state"] != "approved":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot send from state '{s['state']}'. Approve the reply first.",)
    s["state"] = "sent"
    s["sent_at"] = datetime.now(timezone.utc).isoformat()
    save_db()
    return ReplyStatus(**s)

@app.get(
    "/reviews/{review_id}/reply_status",
    response_model=ReplyStatus,
    summary="[Staff Workflow] Get the current reply approval status",
    tags=["Staff Review"],)
async def get_reply_status(
    review_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],):
    review = db_reviews.get(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    await check_hotel_owner(review["hotel_id"], current_user)
    return ReplyStatus(**_get_or_init_reply_status(review_id))


@app.get(
    "/reviews/hotel/{hotel_id}/reply_statuses",
    response_model=List[ReplyStatus],
    summary="[Staff Workflow] Get reply statuses for all reviews of a hotel",
    tags=["Staff Review"],)
async def get_hotel_reply_statuses(
    hotel_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """
    Bulk version of /reply_status. Returns one ReplyStatus per review in
    the hotel, used by the inbox to render the 'Replied' badge correctly
    after a fresh page load — without making N separate API calls.
    """
    await check_hotel_owner(hotel_id, current_user)
    review_ids = [r["id"] for r in db_reviews.values() if r["hotel_id"] == hotel_id]
    return [
        ReplyStatus(**_get_or_init_reply_status(rid))
        for rid in review_ids
    ]

# --- 12. Analytics
@app.get(
    "/analytics/sentiment_distribution",
    response_model=SentimentDistribution,
    summary="Get overall sentiment distribution",
    tags=["Analytics (Dashboard)"],)
async def get_sentiment_distribution(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,
    platform_source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,):
    reviews = _filter_reviews(current_user, hotel_id, platform_source, start_date, end_date)
    d = SentimentDistribution()
    for r in reviews:
        s = r.get("overall_sentiment")
        if s == "positive":
            d.positive += 1
        elif s == "negative":
            d.negative += 1
        else:
            d.neutral += 1
        d.total += 1
    return d

@app.get(
    "/analytics/aspect_summary",
    response_model=List[AspectSummary],
    summary="Get summary of aspects mentioned and their sentiment",
    tags=["Analytics (Dashboard)"],)
async def get_aspect_summary(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,
    platform_source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,):
    reviews = _filter_reviews(current_user, hotel_id, platform_source, start_date, end_date)
    data: Dict[str, Dict[str, int]] = {}
    for r in reviews:
        s = r.get("overall_sentiment")
        for entry in (r.get("aspects") or []):
            a = entry["aspect"]
            if a not in data:
                data[a] = {"count": 0, "positive": 0, "negative": 0, "neutral": 0}
            data[a]["count"] += 1
            if s in ("positive", "negative", "neutral"):
                data[a][s] += 1
    summary = [
        AspectSummary(
            aspect=a, count=d["count"],
            positive_count=d["positive"], negative_count=d["negative"], neutral_count=d["neutral"],)
        for a, d in data.items()]
    return sorted(summary, key=lambda x: x.count, reverse=True)

@app.get(
    "/analytics/reviews_by_platform",
    response_model=List[PlatformReviewCount],
    summary="Get count of reviews per source platform",
    tags=["Analytics (Dashboard)"],)
async def get_reviews_by_platform(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,):
    reviews = _filter_reviews(current_user, hotel_id, None, start_date, end_date)
    counts: Dict[str, int] = {}
    for r in reviews:
        p = r.get("platform_source") or "unknown"
        counts[p] = counts.get(p, 0) + 1
    return [PlatformReviewCount(platform=p, count=c) for p, c in counts.items()]

@app.get(
    "/analytics/sentiment_over_time",
    response_model=List[SentimentTrendPoint],
    summary="Get sentiment trends over time",
    tags=["Analytics (Dashboard)"],)
async def get_sentiment_over_time(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,
    platform_source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,):
    reviews = _filter_reviews(current_user, hotel_id, platform_source, start_date, end_date)
    daily: Dict[str, Dict[str, int]] = {}
    for r in reviews:
        s = r.get("overall_sentiment")
        d = r.get("review_date")
        if not s or not d:
            continue
        if d not in daily:
            daily[d] = {"positive": 0, "negative": 0, "neutral": 0}
        daily[d][s] += 1
    return sorted(
        [SentimentTrendPoint(date=d, **v) for d, v in daily.items()],
        key=lambda x: x.date,)

# --- 13. Dashboard aggregation
@app.get(
    "/analytics/dashboard/overview",
    response_model=DashboardOverview,
    summary="[Dashboard] All KPI cards in a single call",
    tags=["Analytics (Dashboard)"],)
async def get_dashboard_overview(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_issues_limit: int = 5,):
    all_r = _filter_reviews(current_user, hotel_id, None, start_date, end_date, status_filter="all")
    proc = [r for r in all_r if r.get("status") == "processed"]
    pend = [r for r in all_r if r.get("status") == "pending"]
    fail = [r for r in all_r if r.get("status") == "ai_processing_failed"]
    ratings = [r["rating"] for r in all_r if r.get("rating")]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0
    pos = sum(1 for r in proc if r.get("overall_sentiment") == "positive")
    neg = sum(1 for r in proc if r.get("overall_sentiment") == "negative")
    neu = len(proc) - pos - neg
    sentiment = SentimentDistribution(positive=pos, negative=neg, neutral=neu, total=len(proc))
    aspect_data: Dict[str, Dict[str, int]] = {}
    for r in proc:
        s = r.get("overall_sentiment")
        for entry in (r.get("aspects") or []):
            a = entry["aspect"]
            if a not in aspect_data:
                aspect_data[a] = {"count": 0, "positive": 0, "negative": 0, "neutral": 0}
            aspect_data[a]["count"] += 1
            if s in ("positive", "negative", "neutral"):
                aspect_data[a][s] += 1
    top_negative = sorted(
        [
            AspectSummary(
                aspect=a, count=d["count"],
                positive_count=d["positive"], negative_count=d["negative"], neutral_count=d["neutral"],
            )
            for a, d in aspect_data.items()
        ],
        key=lambda x: x.negative_count, reverse=True,
    )[:top_issues_limit]
    platform_counts: Dict[str, int] = {}
    for r in proc:
        p = r.get("platform_source") or "unknown"
        platform_counts[p] = platform_counts.get(p, 0) + 1
    platforms = [PlatformReviewCount(platform=p, count=c) for p, c in platform_counts.items()]

    targets = {hotel_id} if hotel_id else _get_accessible_hotel_ids(current_user)
    hotel_kpis = [_compute_hotel_kpi(hid, db_hotels[hid]["name"]) for hid in targets if hid in db_hotels]

    trend_start = start_date or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    trend_reviews = _filter_reviews(current_user, hotel_id, None, trend_start, end_date)
    daily: Dict[str, Dict[str, int]] = {}
    for r in trend_reviews:
        d = r.get("review_date")
        s = r.get("overall_sentiment")
        if not d or not s:
            continue
        if d not in daily:
            daily[d] = {"positive": 0, "negative": 0, "neutral": 0}
        daily[d][s] += 1
    recent_trend = sorted(
        [SentimentTrendPoint(date=d, **v) for d, v in daily.items()],
        key=lambda x: x.date,)
    return DashboardOverview(
        total_reviews=len(all_r),
        processed_reviews=len(proc),
        pending_reviews=len(pend),
        failed_reviews=len(fail),
        avg_rating=avg_rating,
        sentiment_distribution=sentiment,
        top_negative_aspects=top_negative,
        platform_breakdown=platforms,
        hotel_kpis=hotel_kpis,
        recent_trend=recent_trend,)

@app.get(
    "/analytics/dashboard/hotel_comparison",
    response_model=HotelComparison,
    summary="[Dashboard] Side-by-side KPI comparison across hotels",
    tags=["Analytics (Dashboard)"],)
async def get_hotel_comparison(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_ids: Optional[str] = None,):
    accessible = _get_accessible_hotel_ids(current_user)
    if hotel_ids:
        requested = [h.strip() for h in hotel_ids.split(",") if h.strip()]
        unauthorized = [h for h in requested if h not in accessible]
        if unauthorized:
            raise HTTPException(status_code=403, detail=f"Not authorized for: {unauthorized}")
        targets = requested
    else:
        targets = list(accessible)
    return HotelComparison(
        hotels=[_compute_hotel_kpi(hid, db_hotels[hid]["name"]) for hid in targets if hid in db_hotels],
        compared_at=datetime.now(timezone.utc).isoformat(),)

@app.get(
    "/analytics/dashboard/top_issues",
    response_model=List[TopIssue],
    summary="[Dashboard] Top service issues ranked by negative mention count",
    tags=["Analytics (Dashboard)"],)
async def get_top_issues(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,):
    reviews = _filter_reviews(current_user, hotel_id, None, start_date, end_date)
    data: Dict[str, Dict] = {}
    for r in reviews:
        for entry in (r.get("aspects") or []):
            a = entry["aspect"]
            if a not in data:
                data[a] = {"count": 0, "negative": 0, "hotel_ids": set()}
            data[a]["count"] += 1
            if r.get("overall_sentiment") == "negative":
                data[a]["negative"] += 1
                data[a]["hotel_ids"].add(r["hotel_id"])

    ranked = sorted([{
                "aspect": a,
                "negative_count": d["negative"],
                "negative_rate": round(d["negative"] / d["count"] * 100, 1) if d["count"] else 0.0,
                "hotel_ids": list(d["hotel_ids"]),}
            for a, d in data.items() if d["negative"] > 0],
        key=lambda x: (x["negative_count"], x["negative_rate"]),
        reverse=True,)[:limit]
    return [
        TopIssue(
            rank=i + 1, aspect=item["aspect"],
            negative_count=item["negative_count"],
            negative_rate=item["negative_rate"],
            affected_hotel_ids=item["hotel_ids"],)
        for i, item in enumerate(ranked)]

@app.get(
    "/analytics/dashboard/avg_rating_trend",
    response_model=List[Dict],
    summary="[Dashboard] Average star rating trend (day / week / month)",
    tags=["Analytics (Dashboard)"],)
async def get_avg_rating_trend(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: Literal["day", "week", "month"] = "day",):
    reviews = _filter_reviews(current_user, hotel_id, None, start_date, end_date, status_filter="all")
    buckets: Dict[str, Dict] = {}
    for r in reviews:
        if not r.get("rating") or not r.get("review_date"):
            continue
        try:
            dt = datetime.strptime(r["review_date"], "%Y-%m-%d")
        except ValueError:
            continue
        if granularity == "month":
            key = dt.strftime("%Y-%m")
        elif granularity == "week":
            key = dt.strftime("%Y-W%V")
        else:
            key = dt.strftime("%Y-%m-%d")
        if key not in buckets:
            buckets[key] = {"rating_sum": 0, "count": 0}
        buckets[key]["rating_sum"] += r["rating"]
        buckets[key]["count"] += 1

    return sorted(
        [{"period": k, "avg_rating": round(v["rating_sum"] / v["count"], 2), "review_count": v["count"]}
            for k, v in buckets.items()],
        key=lambda x: x["period"],)

@app.get(
    "/analytics/featured_reviews",
    response_model=List[FeaturedReview],
    summary="[Dashboard] Top high-quality positive reviews for public display",
    tags=["Analytics (Dashboard)"],)
async def get_featured_reviews(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,
    min_rating: int = 4,
    min_length: int = 50,
    limit: int = 10,):
    reviews = _filter_reviews(current_user, hotel_id, overall_sentiment="positive")
    candidates = []
    for r in reviews:
        if r.get("rating", 0) < min_rating or len(r.get("text", "")) < min_length:
            continue
        score = r["rating"] * 10 + len(r["text"]) / 20
        candidates.append(FeaturedReview(
            review_id=r["id"], hotel_id=r["hotel_id"],
            reviewer_name=r["reviewer_name"], rating=r["rating"],
            text=r["text"], review_date=r["review_date"],
            platform_source=r["platform_source"], aspects=r.get("aspects"),
            quality_score=round(score, 2),))
    candidates.sort(key=lambda x: x.quality_score, reverse=True)
    return candidates[:limit]

@app.get(
    "/analytics/dashboard/department_routing",
    response_model=List[RoutedIssue],
    summary="[Dashboard] Negative reviews routed to responsible departments",
    tags=["Analytics (Dashboard)"],)
async def get_department_routing(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,
    department: Optional[str] = None,
    only_open: bool = True,
    limit: int = 100,):
    reviews = _filter_reviews(current_user, hotel_id, overall_sentiment="negative")
    routed: List[RoutedIssue] = []
    for r in reviews:
        for entry in (r.get("aspects") or []):
            aspect = entry["aspect"]
            dept = ASPECT_TO_DEPARTMENT.get(aspect, "general")
            if department and dept != department:
                continue
            reply_state = db_reply_status.get(r["id"], {}).get("state", "draft_pending")
            if only_open and reply_state == "sent":
                continue
            text = r.get("text", "")
            excerpt = text if len(text) <= 200 else text[:200] + "..."
            routed.append(RoutedIssue(
                review_id=r["id"], hotel_id=r["hotel_id"],
                aspect=aspect, department=dept, review_excerpt=excerpt,
                rating=r.get("rating", 0), review_date=r.get("review_date", ""),
                reply_state=reply_state,))
    return routed[:limit]

@app.get(
    "/analytics/dashboard/department_workload",
    response_model=List[DepartmentWorkload],
    summary="[Dashboard] Issue count per department (workload overview)",
    tags=["Analytics (Dashboard)"],)
async def get_department_workload(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    hotel_id: Optional[str] = None,):
    reviews = _filter_reviews(current_user, hotel_id, overall_sentiment="negative")
    by_dept: Dict[str, Dict] = {}
    for r in reviews:
        if db_reply_status.get(r["id"], {}).get("state") == "sent":
            continue
        for entry in (r.get("aspects") or []):
            aspect = entry["aspect"]
            dept = ASPECT_TO_DEPARTMENT.get(aspect, "general")
            if dept not in by_dept:
                by_dept[dept] = {"count": 0, "aspects": set(), "hotel_ids": set()}
            by_dept[dept]["count"] += 1
            by_dept[dept]["aspects"].add(aspect)
            by_dept[dept]["hotel_ids"].add(r["hotel_id"])
    return sorted([DepartmentWorkload(department=d, open_issue_count=v["count"], aspects=sorted(v["aspects"]), hotel_ids=sorted(v["hotel_ids"]),) for d, v in by_dept.items()], key=lambda x: x.open_issue_count, reverse=True,)

@app.get(
    "/analytics/aspect_department_map",
    response_model=Dict[str, str],
    summary="[Config] Get the aspect -> department mapping",
    tags=["Analytics (Dashboard)"],)
async def get_aspect_department_map():
    return ASPECT_TO_DEPARTMENT

# --- 14. Webstock
@app.websocket("/ws/review_updates")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)