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
    """
    Handles 'Story Setup' & Assessment.
    If level is unknown, AI MUST assess English level with 1-2 simple questions.
    """
    
    system_prompt = f"""
    Expert ESL Pedagogical Assistant.
    Task: Build a story with a child in HEBREW and assess their English level.
    
    CONTEXT:
    - Level: {request.student_state.current_estimated_level}
    - History: {len(request.history)} turns.
    
    INSTRUCTIONS:
    1. If Level is 'unknown': You MUST ask 1-2 very simple English questions (e.g. 'Do you know what "Dog" means?') to gauge proficiency.
    2. Narrow elements: Hero, Setting, Goal.
    3. Conclude after 3-4 turns with JSON.
    4. Be brief (token-efficient).
    
    JSON FORMAT:
    {{
        "pedagogical_decision": {{
            "category_name": "string",
            "target_words": ["w1", "w2", "w3", "w4", "w5", "w6"],
            "updated_level": "A1-Sub1/2/3/4",
            "story_elements": {{"hero_name": "str", "setting": "str", "initial_plot_point": "str"}}
        }}
    }}
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
    except:
        pass
        
    return models.InterviewChatResponse(chat_response=text, is_final_turn=False)

def evaluate_assessment_performance(submission: models.AssessmentSubmission) -> models.AssessmentFeedback:
    client = get_client()
    num_correct = sum(1 for k in submission.answers if submission.answers[k] == submission.correct_answers.get(k))
    total_questions = len(submission.correct_answers)

    prompt = f"Score: {num_correct}/{total_questions}. Level: {submission.level}. Briefly give Hebrew feedback and level update (JSON)."

    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.AssessmentFeedback,
        )
    )
    return response.parsed

def generate_story_scene(request: models.GenerateSceneRequest) -> models.SceneResponse:
    client = get_client()
    
    system_prompt = f"""
    ESL Storyteller. Level: {request.student_state.current_estimated_level}.
    Hero: {request.story_elements.hero_name if request.story_elements else 'Hero'}
    Setting: {request.story_elements.setting if request.story_elements else 'Setting'}
    
    HISTORY: {json.dumps(request.plot_history)}
    CHOICE MADE: Branch {request.selected_branch_id if request.selected_branch_id is not None else 'None (Start)'}
    
    REQUIREMENTS:
    1. Continue the story based on the choice.
    2. Include words: {", ".join(request.target_words)}
    3. Output strict JSON (SceneResponse).
    4. scene_text (EN), remedial_scene_text (HE).
    5. assessment_tasks: 1 comprehension (HE), 1 cloze (EN).
    6. story_branches: 2-3 choices (EN/HE).
    """
    
    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=system_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.SceneResponse,
        )
    )
    return response.parsed

def generate_cefr_exam(request: models.GenerateExamRequest) -> models.ExamResponse:
    client = get_client()
    prompt = f"Generate CEFR {request.cefr_level} exam (5 questions) in Hebrew based on: {json.dumps(request.scenes_data)}. JSON output."
    
    response = client.models.generate_content(
        model=EXAM_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.ExamResponse,
        )
    )
    return response.parsed
