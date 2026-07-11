import os
import logging
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import models
import llm_service
from dotenv import load_dotenv

load_dotenv()

# Global Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

app = FastAPI(title="Adaptive English Learning API")

@app.on_event("startup")
async def startup_event():
    llm_service.embedding_sanity_check()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
APP_TOKEN = os.getenv("X_APP_TOKEN", "super-secret-token")

async def verify_token(x_app_token: str = Header(None)):
    if x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-App-Token")
    return x_app_token

# --- Adventure Setup (Launchpad) ---

@app.post("/adventure/setup", response_model=models.AdventureSetupResponse)
async def adventure_setup(request: models.AdventureSetupRequest, token: str = Depends(verify_token)):
    return llm_service.generate_story_options(request)

@app.post("/adventure/generate-dag", response_model=models.StoryDAG)
async def generate_dag(request: models.GenerateDAGRequest, token: str = Depends(verify_token)):
    return llm_service.generate_story_dag(request)

@app.post("/adventure/apply-guardrails", response_model=models.AdventureSetupResponse)
async def apply_guardrails(request: models.GuardrailRequest, token: str = Depends(verify_token)):
    return llm_service.apply_adventure_guardrails(request.data, request.target_rank)

# --- Pedagogical Wizard ---

@app.post("/interview-chat", response_model=models.InterviewChatResponse)
async def interview_chat(request: models.InterviewChatRequest, token: str = Depends(verify_token)):
    return llm_service.generate_interview_response(request)

# --- Hierarchical Story Pipeline ---

@app.post("/story/onboarding-decision", response_model=models.PedagogicalDecision)
async def onboarding_decision(request: models.GenerateArcRequest, token: str = Depends(verify_token)):
    # This might be used by the wizard to finalize the arc and pedagogical state
    return llm_service.onboarding_final_decision(request)

@app.post("/story/generate-act-content", response_model=models.ActContentResponse)
async def generate_act_content(request: models.ActContentRequest, token: str = Depends(verify_token)):
    try:
        return llm_service.generate_act_content(request)
    except Exception as e:
        logging.exception("Error in generate_act_content")
        raise HTTPException(status_code=500, detail=str(e))

# --- Assessment & Evaluation ---

@app.post("/evaluate-assessment", response_model=models.AssessmentFeedback)
async def evaluate_assessment(submission: models.AssessmentSubmission, token: str = Depends(verify_token)):
    return llm_service.evaluate_assessment_performance(submission)

@app.post("/generate-exam", response_model=models.ExamResponse)
async def generate_exam(request: models.GenerateExamRequest, token: str = Depends(verify_token)):
    return llm_service.generate_cefr_exam(request)

# Sandbox UI
@app.get("/sandbox")
async def get_sandbox():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
