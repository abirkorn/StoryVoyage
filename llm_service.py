import os
import json
import logging
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import models

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENE_MODEL_ID = os.getenv("SCENE_MODEL_ID", "gemini-3.1-flash-lite")
EXAM_MODEL_ID = os.getenv("EXAM_MODEL_ID", "gemini-2.5-pro")

CATALOG_PATH = "cefr_catalog.json"
STORY_LOGIC_PATH = "CYOA_story_logic.txt"

def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        logger.warning("GEMINI_API_KEY not found in environment!")
    return genai.Client(api_key=key)

def get_catalog_words(rank_index: int, x: int = 100, pct_above: float = 0.1) -> List[str]:
    """
    Selects x words from cefr_catalog.json around rank_index.
    """
    try:
        if not os.path.exists(CATALOG_PATH):
            logger.error(f"Catalog file missing: {CATALOG_PATH}")
            return ["friend", "adventure", "magic", "quest", "hero"]

        with open(CATALOG_PATH, "r") as f:
            catalog = json.load(f)

        filtered = [w for w in catalog if w.get("pos") in ["n.", "adj."]]
        filtered.sort(key=lambda x: x["rank"])

        num_above = int(x * pct_above)
        num_below = x - num_above

        below_words = [w for w in filtered if w["rank"] <= rank_index]
        above_words = [w for w in filtered if w["rank"] > rank_index]

        selected_below = below_words[-num_below:] if len(below_words) >= num_below else below_words
        selected_above = above_words[:num_above] if len(above_words) >= num_above else above_words

        all_selected = selected_below + selected_above
        logger.info(f"Selected {len(all_selected)} words for rank {rank_index}")
        return [w["w"] for w in all_selected]
    except Exception as e:
        logger.error(f"Error in get_catalog_words: {e}")
        return ["water", "tree", "friend", "happy", "big"]

def generate_adventure_setup(request: models.AdventureSetupRequest) -> models.AdventureSetupResponse:
    logger.info(f"Generating Adventure Setup. Genre: {request.genre}, Rank: {request.rank_index}, Words: {request.num_words}")
    client = get_client()
    words = get_catalog_words(request.rank_index, request.num_words, request.pct_above)

    try:
        if not os.path.exists(STORY_LOGIC_PATH):
            logger.error(f"Story logic file missing: {STORY_LOGIC_PATH}")
            story_logic = "Use CYOA branching logic."
        else:
            with open(STORY_LOGIC_PATH, "r") as f:
                story_logic = f.read()

        prompt = f"""
        You are an expert ESL Story Architect. build an adventure setup for a child.

        GENRE: {request.genre}
        VOCABULARY TO INTEGRATE: {", ".join(words)}

        STORY LOGIC CONTEXT:
        {story_logic}

        TASK:
        1. Generate EXACTLY 3 distinct HEROES (ID, text, description).
        2. Generate EXACTLY 3 distinct SETTINGS (ID, text, description).
        3. Generate EXACTLY 3 distinct CATALYSTS (ID, text, description).
        4. Generate EXACTLY 9 STORY ARCS mapping to selections.

        Output MUST be a strict JSON matching AdventureSetupResponse schema.
        """

        response = client.models.generate_content(
            model=SCENE_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=models.AdventureSetupResponse,
            )
        )

        if not response or not response.parsed:
            logger.error("LLM returned empty or unparseable response")
            raise ValueError("Invalid LLM response")

        data = response.parsed
        data.selected_vocabulary = words
        return data
    except Exception as e:
        logger.error(f"Failed to generate adventure setup: {e}")
        # Provide a minimal valid response as emergency fallback to avoid 500
        return models.AdventureSetupResponse(
            heroes=[models.LaunchpadAnchor(id="1", text="Hero", description="A brave hero")],
            settings=[models.LaunchpadAnchor(id="1", text="Forest", description="A deep forest")],
            catalysts=[models.LaunchpadAnchor(id="1", text="Lost Key", description="A key was lost")],
            potential_story_arcs=[{"title": "Default Arc"}],
            selected_vocabulary=words
        )

def generate_interview_response(request: models.InterviewChatRequest) -> models.InterviewChatResponse:
    client = get_client()
    system_prompt = f"Pedagogical Assistant. Hebrew interview. Level: {request.student_state.current_estimated_level}. Goal: Story build + Level check."
    contents = [types.Content(role="user", parts=[types.Part(text=system_prompt)])]
    for msg in request.history:
        role = "model" if msg.role == "model" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=request.message)]))
    
    response = client.models.generate_content(model=SCENE_MODEL_ID, contents=contents)
    text = response.text.strip()
    try:
        json_text = text
        if "```json" in text:
            json_text = text.split("```json")[1].split("```")[0].strip()
        elif "{" in text:
            json_text = text[text.find("{"):text.rfind("}")+1]
        data = json.loads(json_text)
        decision_data = data.get("pedagogical_decision", data)
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
    client = get_client()
    prompt = f"Generate 5-act story arc for {request.story_elements.hero_name}. JSON."
    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.StoryArc,
        )
    )
    return response.parsed

def generate_act_content(request: models.ActContentRequest) -> models.ActContentResponse:
    client = get_client()
    act = next((a for a in request.story_arc.acts if a.act_number == request.act_number), None)
    prompt = f"Write Act {request.act_number}. Title: {act.title if act else 'Next'}. CEFR: {request.student_state.current_estimated_level}. Vocab: {', '.join(request.target_words)}. JSON output."
    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.ActContentResponse,
        )
    )
    return response.parsed

def evaluate_assessment_performance(submission: models.AssessmentSubmission) -> models.AssessmentFeedback:
    client = get_client()
    num_correct = sum(1 for k in submission.answers if submission.answers[k] == submission.correct_answers.get(k))
    prompt = f"Score {num_correct}. Hebrew feedback (JSON)."
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
    prompt = "Generate CEFR exam. JSON."
    response = client.models.generate_content(
        model=EXAM_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.ExamResponse,
        )
    )
    return response.parsed
