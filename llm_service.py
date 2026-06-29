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
        logger.error("GEMINI_API_KEY not found in environment!")
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

    # Validation to prevent logic errors with pct_above
    if request.pct_above > 1.0:
        logger.warning(f"pct_above {request.pct_above} is > 1.0, likely a UI error. Normalizing to {request.pct_above/100}")
        request.pct_above /= 100.0

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

        logger.info(f"Calling LLM ({SCENE_MODEL_ID}) for adventure setup...")
        response = client.models.generate_content(
            model=SCENE_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=models.AdventureSetupResponse,
            )
        )

        if not response or not response.parsed:
            logger.error(f"LLM returned empty or unparseable response. Raw: {response.text if response else 'None'}")
            raise ValueError("Invalid LLM response")

        data = response.parsed
        data.selected_vocabulary = words
        logger.info("Adventure setup generated successfully.")
        return data
    except Exception as e:
        logger.exception("CRITICAL: Failed to generate adventure setup")
        raise e

def generate_interview_response(request: models.InterviewChatRequest) -> models.InterviewChatResponse:
    logger.info("Generating Interview Response...")
    client = get_client()
    system_prompt = f"Pedagogical Assistant. Hebrew interview. Level: {request.student_state.current_estimated_level}. Goal: Story build + Level check."
    contents = [types.Content(role="user", parts=[types.Part(text=system_prompt)])]
    for msg in request.history:
        role = "model" if msg.role == "model" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=request.message)]))
    
    try:
        response = client.models.generate_content(model=SCENE_MODEL_ID, contents=contents)
        text = response.text.strip()
        # If the LLM indicates a final decision with JSON
        if "{" in text:
            try:
                json_text = text
                if "```json" in text:
                    json_text = text.split("```json")[1].split("```")[0].strip()
                elif "{" in text:
                    json_text = text[text.find("{"):text.rfind("}")+1]
                data = json.loads(json_text)
                decision_data = data.get("pedagogical_decision", data)
                logger.info("Interview concluded with pedagogical decision.")
                return models.InterviewChatResponse(
                    pedagogical_decision=models.PedagogicalDecision(
                        category_name=decision_data["category_name"],
                        target_words=decision_data["target_words"],
                        updated_level=decision_data["updated_level"],
                        story_elements=models.StoryElements(**decision_data["story_elements"])
                    ),
                    is_final_turn=True
                )
            except Exception as json_err:
                logger.warning(f"Failed to parse interview decision JSON: {json_err}. Falling back to chat.")

        return models.InterviewChatResponse(chat_response=text, is_final_turn=False)
    except Exception as e:
        logger.exception("CRITICAL: Failed generate_interview_response")
        raise e

def generate_story_arc(request: models.GenerateArcRequest) -> models.StoryArc:
    logger.info(f"Generating Story Arc for {request.story_elements.hero_name}...")
    client = get_client()
    prompt = f"""
    You are an expert Story Architect for an ESL learning platform.
    Create a 5-act branching story blueprint based on these elements:
    HERO: {request.story_elements.hero_name}
    SETTING: {request.story_elements.setting}
    GOAL: {request.story_elements.goal}

    The story should follow a 5-act structure:
    Act 1: Introduction
    Act 2: Inciting Incident
    Act 3: Rising Action
    Act 4: Climax
    Act 5: Resolution

    Output MUST be a strict JSON matching StoryArc schema.
    """
    try:
        response = client.models.generate_content(
            model=SCENE_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=models.StoryArc,
            )
        )
        if not response or not response.parsed:
            logger.error(f"Failed to generate story arc. Raw: {response.text if response else 'None'}")
            raise ValueError("Invalid Story Arc response")
        return response.parsed
    except Exception as e:
        logger.exception("CRITICAL: Failed generate_story_arc")
        raise e

def generate_act_content(request: models.ActContentRequest) -> models.ActContentResponse:
    logger.info(f"Generating Act {request.act_number} content...")
    client = get_client()
    act_blueprint = next((a for a in request.story_arc.acts if a.act_number == request.act_number), None)

    prompt = f"""
    You are an ESL content creator. Write the content for Act {request.act_number} of a story.

    STORY ARC TITLE: {request.story_arc.story_title}
    ACT BLUEPRINT: {act_blueprint.title if act_blueprint else 'N/A'} - {act_blueprint.description if act_blueprint else 'N/A'}
    CEFR LEVEL: {request.student_state.current_estimated_level}
    TARGET VOCABULARY: {", ".join(request.target_words)}

    REQUIREMENTS:
    1. 'scene_text': Write in SIMPLE ENGLISH suitable for the CEFR level. Integrate target vocabulary naturally.
    2. 'remedial_scene_text': Provide a HEBREW translation of the scene text.
    3. 'vocabulary_definitions': Provide HEBREW definitions for the target words.
    4. 'assessment_tasks':
       - 'comprehension_question': A question in HEBREW about the scene.
       - 'cloze_task': An English sentence from the scene with one word missing (the blank).
    5. 'story_branches': Two choices for what happens next. Provide both HEBREW and ENGLISH text for choices.

    Output MUST be strict JSON matching ActContentResponse schema.
    """
    try:
        response = client.models.generate_content(
            model=SCENE_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=models.ActContentResponse,
            )
        )
        if not response or not response.parsed:
            logger.error(f"Failed to generate act content. Raw: {response.text if response else 'None'}")
            raise ValueError("Invalid Act Content response")
        return response.parsed
    except Exception as e:
        logger.exception("CRITICAL: Failed generate_act_content")
        raise e

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
