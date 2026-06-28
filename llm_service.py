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

CATALOG_PATH = "cefr_catalog.json"
STORY_LOGIC_PATH = "CYOA_story_logic.txt"

def get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_catalog_words(rank_index: int, x: int = 100, pct_above: float = 0.1) -> List[str]:
    """
    Selects x words from cefr_catalog.json around rank_index.
    10% above index, 90% below index (including index).
    Only nouns (n.) and adjectives (adj.).
    """
    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)

    # Filter for nouns and adjectives
    filtered = [w for w in catalog if w.get("pos") in ["n.", "adj."]]

    # Sort by rank
    filtered.sort(key=lambda x: x["rank"])

    num_above = int(x * pct_above)
    num_below = x - num_above

    # Find words below or equal to rank_index
    below_words = [w for w in filtered if w["rank"] <= rank_index]
    # Find words above rank_index
    above_words = [w for w in filtered if w["rank"] > rank_index]

    selected_below = below_words[-num_below:] if len(below_words) >= num_below else below_words
    selected_above = above_words[:num_above] if len(above_words) >= num_above else above_words

    all_selected = selected_below + selected_above
    return [w["w"] for w in all_selected]

def generate_adventure_setup(request: models.AdventureSetupRequest) -> models.AdventureSetupResponse:
    """
    Step 0: Initial story build mechanism.
    Generates 3 heroes, 3 settings, 3 catalysts and 9 story arcs.
    """
    client = get_client()
    words = get_catalog_words(request.rank_index)

    with open(STORY_LOGIC_PATH, "r") as f:
        story_logic = f.read()

    prompt = f"""
    You are an expert ESL Story Architect. Your goal is to build an adventure launchpad for a child.

    GENRE: {request.genre}
    INDEX: {request.rank_index}
    VOCABULARY TO USE: {", ".join(words)}

    STORY LOGIC CONTEXT:
    {story_logic}

    TASK:
    1. Generate 3 distinct HEROES (ID, name/text, description).
    2. Generate 3 distinct SETTINGS (ID, text, description).
    3. Generate 3 distinct CATALYSTS (ID, text, description).
    4. Generate 9 High-Level STORY ARCS (blueprints) that meet the potential selections.
       Each arc should be a conceptual blueprint mapping to the CYOA methodology.

    OUTPUT:
    Strict JSON matching the AdventureSetupResponse schema.
    """

    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.AdventureSetupResponse,
        )
    )
    data = response.parsed
    data.selected_vocabulary = words
    return data

def generate_interview_response(request: models.InterviewChatRequest) -> models.InterviewChatResponse:
    client = get_client()
    system_prompt = f"""
    Expert ESL Pedagogical Assistant. Conduct an interview in HEBREW to assess the child.

    STUDENT STATE:
    - Current Level: {request.student_state.current_estimated_level}

    GUIDELINES:
    1. Be friendly, magical, and brief.
    2. If level is 'unknown', ask exactly ONE very simple English check question.
    3. Otherwise, guide them toward choosing story elements.
    4. Conclude after 3-4 turns with JSON.
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
    """Step 1: The Blueprint."""
    client = get_client()
    system_prompt = f"""
    Expert ESL Story Architect. Generate a complete 5-act story arc blueprint.

    ELEMENTS:
    - Hero: {request.story_elements.hero_name}
    - Setting: {request.story_elements.setting}
    - Goal: {request.story_elements.goal}
    - CEFR: {request.student_state.current_estimated_level}

    REQUIREMENTS:
    - Define 5 acts (Intro, Inciting Incident, Rising Action, Climax, Resolution).
    - Provide a structured JSON matching models.StoryArc.
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
    """Step 2: The Prose (Act content)."""
    client = get_client()
    act = next((a for a in request.story_arc.acts if a.act_number == request.act_number), None)

    system_prompt = f"""
    Expert ESL Story Writer. Write Act {request.act_number} for an adventure.
    
    CONTEXT:
    - Story Arc Title: {request.story_arc.story_title}
    - Act Info: {act.title}: {act.description if act else ''}
    - CEFR Level: {request.student_state.current_estimated_level}
    - Target Vocabulary: {", ".join(request.target_words)}
    
    REQUIREMENTS:
    1. scene_text: Simple English prose. Weave in target words naturally.
    2. remedial_scene_text: Simplified Hebrew translation of the scene.
    3. vocabulary_definitions: Dictionary (Word -> Hebrew Meaning).
    4. assessment_tasks:
        - 1 Comprehension Question (Hebrew)
        - 1 Cloze Task (English sentence, options in English, translation in Hebrew)
    5. story_branches: 2 choices for local flavor (English/Hebrew).

    Output MUST be a strict JSON matching ActContentResponse schema.
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
    total = len(submission.correct_answers)
    prompt = f"Student got {num_correct}/{total} on their tasks. Provide warm Hebrew feedback and pedagogical strategy (JSON)."

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
    prompt = f"Generate CEFR {request.cefr_level} exam in Hebrew based on previous scenes. JSON output."
    response = client.models.generate_content(
        model=EXAM_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.ExamResponse,
        )
    )
    return response.parsed
