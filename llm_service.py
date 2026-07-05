import os
import json
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import models

load_dotenv()

SCENE_MODEL_ID = os.getenv("SCENE_MODEL_ID", "gemini-3.1-flash-lite")
EXAM_MODEL_ID = os.getenv("EXAM_MODEL_ID", "gemini-2.5-pro")

def get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_interview_response(request: models.InterviewChatRequest) -> models.InterviewChatResponse:
    client = get_client()
    system_prompt = f"""
    Expert ESL Pedagogical Assistant. Build a story with a child in HEBREW and assess their English level.
    CONTEXT: Level: {request.student_state.current_estimated_level}
    INSTRUCTIONS:
    1. If Level is 'unknown': Ask 1 simple English check question.
    2. Narrow elements: Hero, Setting, Goal.
    3. Conclude after 3-4 turns with JSON.
    """

    contents = [types.Content(role="user", parts=[types.Part(text=system_prompt)])]
    for msg in request.history:
        role = "model" if msg.role == "model" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=request.message)]))
    
    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=contents
    )

    text = response.text.strip()
    try:
        json_text = text
        if "```json" in text:
            json_text = text.split("```json")[1].split("```")[0].strip()
        elif "{" in text:
            json_text = text[text.find("{"):text.rfind("}")+1]
        data = json.loads(json_text)
        decision_data = data.get("pedagogical_decision", data)
        if "category_name" in decision_data:
            return models.InterviewChatResponse(
                pedagogical_decision=models.PedagogicalDecision(
                    category_name=decision_data["category_name"],
                    target_words=decision_data["target_words"],
                    updated_level=decision_data["updated_level"],
                    story_elements=models.StoryElements(**decision_data["story_elements"])
                ),
                is_final_turn=True
            )
    except: pass
    return models.InterviewChatResponse(chat_response=text, is_final_turn=False)

def generate_story_arc(request: models.GenerateArcRequest) -> models.StoryArc:
    """Step 1: Generate the high-level 5-act story blueprint."""
    client = get_client()
    system_prompt = f"""
    ESL Story Architect. Generate a 5-act dramatic story arc blueprint for a child.

    ELEMENTS:
    - Hero: {request.story_elements.hero_name}
    - Setting: {request.story_elements.setting}
    - Goal: {request.story_elements.goal}
    - CEFR Level: {request.student_state.current_estimated_level}
    - Genre/Theme: {request.genre_theme or "Adventure"}

    OUTPUT:
    A strict JSON matching the StoryArc schema.
    Define 5 acts: Introduction, Inciting Incident, Rising Action, Climax, Resolution.
    Each act needs a title and a brief description.
    """

    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=system_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.StoryArc,
        )
    )
    return response.parsed

def generate_act_content(request: models.ActContentRequest) -> models.ActContentResponse:
    """Step 2: Generate the prose and assessment for a specific act."""
    client = get_client()
    act = next((a for a in request.story_arc.acts if a.act_number == request.act_number), None)

    system_prompt = f"""
    ESL Story Writer. Write the prose for Act {request.act_number}: {act.title if act else 'Next Act'}.

    STORY ARC CONTEXT:
    - Overall Arc: {request.story_arc.story_title}
    - Current Act Goal: {act.description if act else ''}
    - CEFR Level: {request.student_state.current_estimated_level}
    - Target Words to weave in: {", ".join(request.target_words)}

    REQUIREMENTS:
    1. scene_text: Simple English prose for the act.
    2. remedial_scene_text: Hebrew translation.
    3. vocabulary_definitions: Dictionary of target words and their Hebrew meaning.
    4. assessment_tasks: 1 comprehension (HE), 1 cloze (EN).
    5. story_branches: 2-3 branches for how the story might transition slightly.
    """

    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=system_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.ActContentResponse,
        )
    )
    return response.parsed

def evaluate_assessment_performance(submission: models.AssessmentSubmission) -> models.AssessmentFeedback:
    client = get_client()
    num_correct = sum(1 for k in submission.answers if submission.answers[k] == submission.correct_answers.get(k))
    prompt = f"Score: {num_correct}/{len(submission.correct_answers)}. Give Hebrew feedback (JSON)."
    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.AssessmentFeedback,
        )
    )
    return response.parsed

def generate_cefr_exam(request: models.GenerateExamRequest) -> models.ExamResponse:
    client = get_client()
    prompt = f"Generate CEFR {request.cefr_level} exam in Hebrew. JSON."
    response = client.models.generate_content(
        model=EXAM_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.ExamResponse,
        )
    )
    return response.parsed
