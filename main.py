import os
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import models
import llm_service
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Adaptive English Learning API")

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

# --- Pedagogical Wizard ---

@app.post("/interview-chat", response_model=models.InterviewChatResponse)
async def interview_chat(request: models.InterviewChatRequest, token: str = Depends(verify_token)):
    return llm_service.generate_interview_response(request)

# --- Hierarchical Story Pipeline ---

@app.post("/story/generate-arc", response_model=models.StoryArc)
async def generate_arc(request: models.GenerateArcRequest, token: str = Depends(verify_token)):
    return llm_service.generate_story_arc(request)

@app.post("/story/generate-act-content", response_model=models.ActContentResponse)
async def generate_act_content(request: models.ActContentRequest, token: str = Depends(verify_token)):
    return llm_service.generate_act_content(request)

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
