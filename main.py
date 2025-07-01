import os
import time
import logging
import uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, Request, HTTPException, Header, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import openai
from collections import defaultdict
from datetime import datetime
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Load environment variables ---
load_dotenv()

# --- Configuration ---
class Config:
    AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful, knowledgeable assistant.")
    MAX_TOKENS = min(int(os.getenv("MAX_TOKENS", 200)), 4000)  # Hard cap at 4000
    ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    VALID_API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
    RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))  # requests
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))  # seconds

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    @classmethod
    def validate(cls):
        if not cls.VALID_API_KEYS:
            logger.warning("No API_KEYS set in environment, using default key (insecure for production!)")
            cls.VALID_API_KEYS = {"default-dev-key"}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("secure-chatbot")

Config.validate()

# Track API start time for uptime reporting
API_START_TIME = time.time()

# --- OpenAI Client ---
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=Config.OPENAI_TIMEOUT
)

# --- FastAPI Setup ---
app = FastAPI(
    title="Enterprise Chatbot API",
    description="Secure, production-ready API for AI-powered chatbot with rate limiting, monitoring, and analytics.",
    version="1.2.0",
    docs_url="/docs",
    redoc_url=None
)

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Rate Limiting Storage ---
rate_limits = defaultdict(list)

# --- Models ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, 
                         description="The user's question to the AI assistant")
    context: Optional[str] = Field(None, max_length=1000, 
                                 description="Additional context to help answer the question")
    conversation_id: Optional[str] = Field(None, description="Unique ID for conversation tracking")

    @validator('question')
    def validate_question(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Question must be at least 3 characters long")
        return v

class AskResponse(BaseModel):
    answer: str
    model: str
    tokens_used: int
    request_id: str
    timestamp: str
    conversation_id: Optional[str]

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    uptime: float

class ErrorResponse(BaseModel):
    error: str
    code: int
    details: Optional[Dict]

# --- Middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    logger.info(f"Request {request_id} completed in {process_time:.2f}s")
    return response

# --- Dependencies ---
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    if x_api_key not in Config.VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt: {x_api_key}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Rate limiting
    now = time.time()
    window_start = now - Config.RATE_LIMIT_WINDOW
    requests_in_window = [ts for ts in rate_limits[x_api_key] if ts > window_start]
    
    if len(requests_in_window) >= Config.RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for key: {x_api_key}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({Config.RATE_LIMIT} requests per minute)"
        )
    
    rate_limits[x_api_key].append(now)
    return x_api_key

# --- Exception Handlers ---
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "Rate limit exceeded", "detail": str(exc)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "code": exc.status_code}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# --- Endpoints ---
@app.post(
    "/ask", 
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
@limiter.limit(f"{Config.RATE_LIMIT}/minute")
async def ask_question(
    request: Request,
    payload: AskRequest,
    api_key: str = Depends(verify_api_key)
) -> AskResponse:
    """
    Submit a question to the AI assistant and receive a response.
    
    - **question**: Required, 3-500 characters
    - **context**: Optional additional context
    - **conversation_id**: Optional ID for conversation tracking
    """
    logger.info(
        f"New request from {api_key[:4]}... | "
        f"Q: {payload.question[:50]}... | "
        f"Conversation: {payload.conversation_id or 'new'}"
    )
    
    try:
        messages = [
            {"role": "system", "content": Config.SYSTEM_PROMPT},
            {"role": "user", "content": payload.question}
        ]
        
        if payload.context:
            messages.append({"role": "user", "content": f"Additional context: {payload.context}"})
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=Config.AI_MODEL,
            messages=messages,
            max_tokens=Config.MAX_TOKENS,
            temperature=0.7
        )
        processing_time = time.time() - start_time
        
        answer = response.choices[0].message.content.strip()
        usage = response.usage
        
        logger.info(
            f"Request completed in {processing_time:.2f}s | "
            f"Tokens: {usage.total_tokens} | "
            f"Model: {Config.AI_MODEL}"
        )
        
        return AskResponse(
            answer=answer,
            model=Config.AI_MODEL,
            tokens_used=usage.total_tokens,
            request_id=request.state.request_id,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=payload.conversation_id
        )
        
    except openai.RateLimitError:
        logger.error("OpenAI rate limit exceeded")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI service rate limit exceeded"
        )
    except openai.APITimeoutError:
        logger.error("OpenAI API timeout")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="OpenAI service timeout"
        )
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI service error: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint that reports API status and version information.
    """
    return HealthResponse(
        status="healthy",
        version=app.version,
        model=Config.AI_MODEL,
        uptime=time.time() - API_START_TIME
    )

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Enterprise Chatbot API - See /docs for API documentation"}