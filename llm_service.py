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

_catalog_data = None
_pos_lookup = {}

def _load_catalog():
    global _catalog_data, _pos_lookup
    if _catalog_data is not None:
        return
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r") as f:
            _catalog_data = json.load(f)
        for w in _catalog_data:
            word = w["w"].lower()
            if word not in _pos_lookup:
                _pos_lookup[word] = []
            _pos_lookup[word].append({"pos": w["pos"], "rank": w["rank"]})

def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        logger.error("GEMINI_API_KEY not found in environment!")
    return genai.Client(api_key=key)

def get_catalog_words(rank_index: int, x: int = 100, pct_above: float = 0.1) -> Dict[str, Any]:
    """
    Selects x words from cefr_catalog.json around rank_index.
    Returns dict with words and rank bounds.
    """
    try:
        _load_catalog()
        if not _catalog_data:
            return {"words": ["friend", "adventure", "magic", "quest", "hero"], "min_rank": 0, "max_rank": 0}

        filtered = [w for w in _catalog_data if w.get("pos") in ["n.", "adj."]]
        filtered.sort(key=lambda x: x["rank"])

        num_above = int(x * pct_above)
        num_below = x - num_above

        below_words = [w for w in filtered if w["rank"] <= rank_index]
        above_words = [w for w in filtered if w["rank"] > rank_index]

        selected_below = below_words[-num_below:] if len(below_words) >= num_below else below_words
        selected_above = above_words[:num_above] if len(above_words) >= num_above else above_words

        all_selected = selected_below + selected_above
        ranks = [w["rank"] for w in all_selected]

        logger.info(f"Selected {len(all_selected)} words for rank {rank_index}. Range: {min(ranks) if ranks else 0}-{max(ranks) if ranks else 0}")
        return {
            "words": [w["w"] for w in all_selected],
            "min_rank": min(ranks) if ranks else 0,
            "max_rank": max(ranks) if ranks else 0
        }
    except Exception as e:
        logger.error(f"Error in get_catalog_words: {e}")
        return {"words": ["water", "tree", "friend", "happy", "big"], "min_rank": 0, "max_rank": 0}

def generate_adventure_setup(request: models.AdventureSetupRequest) -> models.AdventureSetupResponse:
    logger.info(f"Generating Adventure Setup. Genre: {request.genre}, Rank: {request.rank_index}, Words: {request.num_words}")

    # Validation to prevent logic errors with pct_above
    if request.pct_above > 1.0:
        logger.warning(f"pct_above {request.pct_above} is > 1.0, likely a UI error. Normalizing to {request.pct_above/100}")
        request.pct_above /= 100.0

    client = get_client()
    vocab_data = get_catalog_words(request.rank_index, request.num_words, request.pct_above)
    words = vocab_data["words"]

    try:
        if not os.path.exists(STORY_LOGIC_PATH):
            logger.error(f"Story logic file missing: {STORY_LOGIC_PATH}")
            story_logic = "Use CYOA branching logic."
        else:
            with open(STORY_LOGIC_PATH, "r") as f:
                story_logic = f.read()

        # Derive name anchor letters from first few words
        anchor_letters = "".join([w[0].upper() for w in words[:4]])

        prompt = f"""
        You are an expert ESL Story Architect. Build a modular adventure setup for a child.

        GENRE: {request.genre}
        VOCABULARY BANK: {", ".join(words)}
        NAME ANCHOR LETTERS: {anchor_letters}

        STORY LOGIC CONTEXT:
        {story_logic}

        CONSTRAINTS:
        - HERO LABELS: For heroes, use the pattern "Name - Short Vocabulary Description".
          - Example: "Didi - a small monkey" (if 'monkey' is in vocab).
          - NAMES: Ensure high variability. Use unique names that incorporate or rhyme with the anchor letters '{anchor_letters}' where possible.
        - SETTING/CATALYST LABELS: The 'text' field MUST primarily use words from the VOCABULARY BANK.
        - DESCRIPTIONS: The 'description' field can use richer, more descriptive free text to provide context.
        - STORY ARCS:
          - Generate 9 DETAILED STORY ARCS.
          - IMPORTANT: Each arc MUST explicitly embed and reference the specific Hero, Setting, and Catalyst combination it represents.
          - CYOA LOGIC: Each act's 'ending_point' must be a moment of choice.
          - BRANCH OPTIONS: Provide 2-3 distinct options for what the player can do next.

        TASK:
        1. Generate EXACTLY 3 distinct HEROES (ID, text, description).
        2. Generate EXACTLY 3 distinct SETTINGS (ID, text, description).
        3. Generate EXACTLY 3 distinct CATALYSTS (ID, text, description).
        4. Generate EXACTLY 9 DETAILED STORY ARCS following the constraints above.

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
        data.vocabulary_min_rank = vocab_data["min_rank"]
        data.vocabulary_max_rank = vocab_data["max_rank"]
        logger.info("Adventure setup generated successfully.")
        return data
    except Exception as e:
        logger.exception("CRITICAL: Failed to generate adventure setup")
        raise e

def onboarding_final_decision(request: models.GenerateArcRequest) -> models.PedagogicalDecision:
    # Logic to finalize elements based on wizard interview
    # This would call LLM to synthesize elements into a DetailedStoryArc eventually
    # For now, it returns the standard decision structure
    return models.PedagogicalDecision(
        category_name="Adventure",
        target_words=["lion", "brave", "forest", "map", "magic", "sun"],
        updated_level="A1-Sub1",
        story_elements=request.story_elements
    )

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
    logger.info(f"Generating Act {request.act_blueprint.act_number} content for {request.story_arc_title}...")
    client = get_client()
    bp = request.act_blueprint

    prompt = f"""
    You are an expert Children's Book Author specializing in interactive fiction ("Choose Your Own Adventure"). Write Act {bp.act_number} of a captivating, engaging story for young readers. 

    The primary goal is literary quality and emotional resonance. The story must NEVER feel like a language test.

    STORY CONTEXT:
    - TITLE: {request.story_arc_title}
    - GENRE: {request.genre}
    - HERO: {request.hero_description}
    - SETTING: {request.setting_description}
    - FLOW: {request.previous_act_title or 'Start'} -> **{bp.title}** -> {request.next_act_title or 'End'}

    ACT DETAILS:
    - DESCRIPTION: {bp.description}
    - STARTING POINT: {bp.starting_point}
    - ENDING POINT: {bp.ending_point}
    - CHOICE OPTIONS TO BUILD TOWARDS: {", ".join(bp.branch_options)}

    LITERARY & VOCABULARY GUIDELINES (CEFR LEVEL: {request.student_state.current_estimated_level}):
    - TARGET WORD COUNT: ~{request.word_count_target} words.
    - MANDATORY VOCABULARY: {", ".join(request.target_words)}
    
    REQUIREMENTS:
    1. 'scene_text': Write engaging, descriptive prose in SIMPLE, AGE-APPROPRIATE ENGLISH. 
       - Seamlessly bridge the STARTING POINT to the ENDING POINT.
       - You MUST include ALL mandatory vocabulary words, but they must be integrated ORGANICALLY. Do not break narrative flow to force a word. Use context clues, sensory details, and natural sentence structures. The words can be morphed (e.g., plurals, past tense) to fit the grammar perfectly.
       - The scene must build natural tension and end smoothly at a crossroad that makes the CHOICE OPTIONS obvious and compelling.
    2. 'story_branches': Match the CHOICE OPTIONS exactly. Ensure choices represent active, concrete deeds or decisions. Provide both HEBREW and ENGLISH text for each.
    3. 'remedial_scene_text': Provide a natural, storytelling HEBREW translation of the scene text.
    4. 'vocabulary_definitions': Provide HEBREW definitions accurately reflecting how the mandatory words were used in the context of the scene.
    5. 'assessment_tasks':
       - 'comprehension_question': A question in HEBREW about the scene.
       - 'cloze_task': An English sentence from the scene with one word missing (the blank).

    Output MUST be strictly valid JSON matching the ActContentResponse schema. Do not include markdown formatting or prose outside the JSON.
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

def apply_adventure_guardrails(data: models.AdventureSetupResponse, target_rank: int) -> models.AdventureSetupResponse:
    """
    Identifies high-rank nouns/adjectives in LLM output and replaces them with
    simpler alternatives from the catalog.
    """
    _load_catalog()
    if not _catalog_data:
        return data

    import re
    # Simple word-by-word replacement logic
    def simplify_text(text: str, max_rank: int) -> str:
        words = re.findall(r"\w+", text)
        new_text = text
        for w in words:
            lw = w.lower()
            if lw in _pos_lookup:
                # Check if any sense of this word is too hard
                hardest_rank = max(p["rank"] for p in _pos_lookup[lw])
                if hardest_rank > max_rank + 200: # Threshold for 'too far'
                    # Find replacement
                    target_pos = _pos_lookup[lw][0]["pos"]
                    replacements = [c for c in _catalog_data if c["pos"] == target_pos and c["rank"] <= max_rank]
                    if replacements:
                        # Pick one reasonably close to target rank
                        best_rep = replacements[-1]["w"]
                        new_text = re.sub(rf"\b{w}\b", best_rep, new_text)
        return new_text

    # Apply ONLY to labels (the 'text' field), preserving 'description'
    for h in data.heroes:
        h.text = simplify_text(h.text, target_rank)
    for s in data.settings:
        s.text = simplify_text(s.text, target_rank)
    for c in data.catalysts:
        c.text = simplify_text(c.text, target_rank)

    return data

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
