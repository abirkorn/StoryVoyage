import os
import json
import logging
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import models

load_dotenv()

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
            raise FileNotFoundError(f"Linguistic catalog not found at {CATALOG_PATH}")

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
        - LABELS: In the 'text' field for Heroes, Settings, and Catalysts, prioritize natural, creative names. Use the VOCABULARY BANK for inspiration, but don't force words if they make the label nonsensical.
        - HERO LABELS: Use the pattern "Name - Description".
        - DATA LINKING: Every Story Arc MUST explicitly link to a Hero, Setting, and Catalyst using the 'hero_id', 'setting_id', and 'catalyst_id' fields.
        - STRUCTURE: Each arc should have 3 Acts.
        - ACT DETAILS: For every act in 'acts', you MUST provide:
            - 'act_number': (1, 2, or 3)
            - 'title': A short, catchy title.
            - 'description': A high-level overview of the act.
            - 'plot_beats': A list of 4-5 specific chronological events (beats) that happen during this act. This is the skeleton for the actual prose.
            - 'starting_point': Where the character begins this act.
            - 'ending_point': Where the act ends, leading to a choice.
            - 'branch_options': A list of 2-3 short, active choice strings.

        TASK:
        1. Generate EXACTLY 3 HEROES (IDs: h1, h2, h3).
        2. Generate EXACTLY 3 SETTINGS (IDs: s1, s2, s3).
        3. Generate EXACTLY 3 CATALYSTS (IDs: c1, c2, c3).
        4. Generate EXACTLY 9 STORY ARCS covering unique combinations.
           Each arc object MUST have: 'hero_id', 'setting_id', 'catalyst_id', 'title', and 'acts' (with 3 full Act objects).

        Output MUST be strict JSON matching the AdventureSetupResponse schema.
        """

        logger.info(f"Calling LLM ({SCENE_MODEL_ID}) for adventure setup...")
        try:
            response = client.models.generate_content(
                model=SCENE_MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
        except Exception as api_err:
            logger.error(f"Gemini API Call Failed for /adventure/setup: {api_err}")
            raise api_err

        if not response or not response.parsed:
            raw_text = getattr(response, 'text', 'None')
            logger.error(f"LLM returned empty or unparseable response for /adventure/setup. Raw text: {raw_text}")

            # Fallback for Pydantic parsing issues if raw text exists
            if raw_text and raw_text != 'None':
                try:
                    # Strip markdown if present
                    clean_json = raw_text
                    if "```json" in raw_text:
                        clean_json = raw_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in raw_text:
                        clean_json = raw_text.split("```")[1].split("```")[0].strip()

                    # Handle extra data at the end (like an extra })
                    try:
                        raw_json = json.loads(clean_json)
                    except json.JSONDecodeError:
                        # Try to find the last valid }
                        last_brace = clean_json.rfind("}")
                        if last_brace != -1:
                            raw_json = json.loads(clean_json[:last_brace+1])
                        else:
                            raise

                    # Normalize keys
                    if "story_arcs" in raw_json and "potential_story_arcs" not in raw_json:
                        raw_json["potential_story_arcs"] = raw_json.pop("story_arcs")

                    # Normalize hero/setting/catalyst fields
                    for key in ["heroes", "settings", "catalysts"]:
                        if key in raw_json and isinstance(raw_json[key], list):
                            for item in raw_json[key]:
                                if not isinstance(item, dict): continue
                                if "name" in item and "text" not in item:
                                    item["text"] = item.pop("name")
                                if "id" not in item:
                                    item["id"] = f"{key[0]}{raw_json[key].index(item)+1}"

                    if "potential_story_arcs" in raw_json and isinstance(raw_json["potential_story_arcs"], list):
                        for arc in raw_json["potential_story_arcs"]:
                            if not isinstance(arc, dict): continue
                            if "acts" in arc and isinstance(arc["acts"], list):
                                new_acts = []
                                for i, act in enumerate(arc["acts"]):
                                    if isinstance(act, str):
                                        new_acts.append({
                                            "act_number": i + 1,
                                            "title": f"Act {i+1}",
                                            "description": act,
                                            "starting_point": "Start",
                                            "ending_point": "End",
                                            "branch_options": []
                                        })
                                    elif isinstance(act, dict):
                                        if "act" in act and "act_number" not in act:
                                            act["act_number"] = act.pop("act")
                                        if "act_number" not in act:
                                            act["act_number"] = i + 1
                                        for field in ["title", "starting_point", "ending_point", "description"]:
                                            if field not in act: act[field] = "..."
                                        if "plot_beats" not in act: act["plot_beats"] = []
                                        if "branch_options" not in act: act["branch_options"] = []
                                        new_acts.append(act)
                                arc["acts"] = new_acts
                            else:
                                arc["acts"] = []

                    data = models.AdventureSetupResponse(**raw_json)
                    logger.info("Manual parse fallback succeeded for adventure setup.")
                except Exception as parse_err:
                    logger.error(f"Manual parse fallback failed for adventure setup: {parse_err}")
                    raise ValueError(f"Invalid LLM response and fallback failed: {raw_text}")
            else:
                raise ValueError("Invalid LLM response: Empty body or blocked")
        else:
            data = response.parsed

        if data.potential_story_arcs:
            logger.info(f"Arc 0 keys: {data.potential_story_arcs[0].model_dump().keys()}")

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
            )
        )

        raw_text = getattr(response, 'text', 'None')
        if not raw_text or raw_text == 'None':
             raise ValueError("Invalid Story Arc response: Empty body")

        try:
            clean_json = raw_text
            if "```json" in raw_text:
                clean_json = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                clean_json = raw_text.split("```")[1].split("```")[0].strip()

            try:
                raw_json = json.loads(clean_json)
            except json.JSONDecodeError:
                last_brace = clean_json.rfind("}")
                if last_brace != -1:
                    raw_json = json.loads(clean_json[:last_brace+1])
                else:
                    raise

            # Normalize acts if they are called something else or have different structure
            if "acts" in raw_json and isinstance(raw_json["acts"], list):
                new_acts = []
                for i, act in enumerate(raw_json["acts"]):
                    if isinstance(act, str):
                        new_acts.append({
                            "act_number": i + 1,
                            "title": f"Act {i+1}",
                            "description": act,
                            "starting_point": "...",
                            "ending_point": "..."
                        })
                    elif isinstance(act, dict):
                        if "act" in act and "act_number" not in act:
                            act["act_number"] = act.pop("act")
                        for field in ["title", "starting_point", "ending_point", "description"]:
                            if field not in act: act[field] = "..."
                        if "plot_beats" not in act: act["plot_beats"] = []
                        new_acts.append(act)
                raw_json["acts"] = new_acts

            return models.StoryArc(**raw_json)
        except Exception as parse_err:
            logger.error(f"Manual parse failed for story arc: {parse_err}")
            raise ValueError(f"Invalid Story Arc response: {raw_text}")
    except Exception as e:
        logger.exception("CRITICAL: Failed generate_story_arc")
        raise e

def generate_act_content(request: models.ActContentRequest) -> models.ActContentResponse:
    logger.info(f"Generating Act {request.act_blueprint.act_number} content for {request.story_arc_title}...")
    client = get_client()
    bp = request.act_blueprint

    prompt = f"""
    You are an expert, award-winning Children's Book Author specializing in interactive fiction ("Choose Your Own Adventure"). Write Act {bp.act_number} of a captivating, deeply engaging story for young readers.

    The primary goal is **exceptional literary quality and emotional resonance**. The story must feel like a cherished piece of children's literature, NEVER like a dry language exercise or a collection of forced sentences.

    STORY CONTEXT:
    - TITLE: {request.story_arc_title}
    - GENRE: {request.genre}
    - HERO: {request.hero_description}
    - SETTING: {request.setting_description}
    - CATALYST: {request.catalyst_description}
    - FLOW: {request.previous_act_title or 'Start'} -> **{bp.title}** -> {request.next_act_title or 'End'}

    ACT DETAILS:
    - DESCRIPTION: {bp.description}
    - PLOT BEATS: {", ".join(bp.plot_beats)}
    - STARTING POINT: {bp.starting_point}
    - ENDING POINT: {bp.ending_point}
    - CHOICE OPTIONS TO BUILD TOWARDS: {", ".join(bp.branch_options)}

    LITERARY & VOCABULARY GUIDELINES (CEFR LEVEL: {request.student_state.current_estimated_level}):
    - NARRATIVE VOICE: Use a warm, vivid, and age-appropriate voice. Even with limited vocabulary, strive for "Show, Don't Tell." Focus on sensory details (sounds, colors, smells) and the protagonist's internal feelings (excitement, hesitation, curiosity).
    - STRUCTURE: Write EXACTLY {request.num_paragraphs} detailed paragraphs. Each paragraph MUST contain EXACTLY {request.sentences_per_paragraph} sentences. This is a strict constraint.
    - VOCABULARY POOL: {", ".join(request.target_words)}
    - SELECTION: Naturally weave in at least 10 words from the VOCABULARY POOL above.

    REQUIREMENTS:
    1. 'scene_text': Write engaging, descriptive prose in VIVID, SIMPLE ENGLISH.
       - Seamlessly bridge the STARTING POINT to the ENDING POINT while expanding on the provided DESCRIPTION.
       - Integration: The words from the pool MUST feel like a natural part of the author's voice. You may adapt the words (e.g., "glowing" instead of "glow") to maintain grammatical excellence.
       - Pacing: Build mystery, wonder, or excitement. The scene must end exactly at the moment of decision, making the BRANCH OPTIONS feel like urgent, meaningful crossroads.
    2. 'used_vocabulary': List the words from the pool that you actually incorporated.
    3. 'remedial_scene_text': Provide a natural, storytelling HEBREW translation of the scene text.
    4. 'vocabulary_definitions': Provide HEBREW definitions accurately reflecting how the words were used in the context of the scene.
    5. 'assessment_tasks':
       - 'comprehension_question': A question in HEBREW about the scene. Provide 3 likely options in HEBREW and the correct index.
       - 'cloze_task': An English sentence from the scene with one word missing (the blank). Provide 3 options in ENGLISH and the correct index.
    6. 'story_branches': Match the CHOICE OPTIONS exactly. Provide both HEBREW and ENGLISH text for each.

    Output MUST be strictly valid JSON matching the ActContentResponse schema. Do not include markdown formatting or prose outside the JSON.
    """
    try:
        response = client.models.generate_content(
            model=SCENE_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )

        if not response or not response.parsed:
            raw_text = getattr(response, 'text', 'None')
            logger.error(f"LLM returned empty or unparseable response for /story/generate-act-content. Raw text: {raw_text}")

            # Fallback for Pydantic parsing issues if raw text exists
            if raw_text and raw_text != 'None':
                try:
                    # Strip markdown if present
                    clean_json = raw_text
                    if "```json" in raw_text:
                        clean_json = raw_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in raw_text:
                        clean_json = raw_text.split("```")[1].split("```")[0].strip()

                    try:
                        raw_json = json.loads(clean_json)
                    except json.JSONDecodeError:
                        last_brace = clean_json.rfind("}")
                        if last_brace != -1:
                            raw_json = json.loads(clean_json[:last_brace+1])
                        else:
                            raise

                    # Normalize vocabulary_definitions (if it's a dict instead of a list)
                    if "vocabulary_definitions" in raw_json and isinstance(raw_json["vocabulary_definitions"], dict):
                        raw_json["vocabulary_definitions"] = [
                            {"word": k, "definition_hebrew": v} for k, v in raw_json["vocabulary_definitions"].items()
                        ]

                    # Normalize assessment_tasks
                    if "assessment_tasks" in raw_json and isinstance(raw_json["assessment_tasks"], dict):
                        at = raw_json["assessment_tasks"]
                        if "comprehension_question" in at:
                            cq = at["comprehension_question"]
                            if isinstance(cq, str):
                                at["comprehension_question"] = {
                                    "question_text_hebrew": cq,
                                    "options_hebrew": ["...", "...", "..."],
                                    "correct_option_index": 0,
                                    "explanation_hebrew": "..."
                                }
                            elif isinstance(cq, dict):
                                if "question" in cq and "question_text_hebrew" not in cq:
                                    cq["question_text_hebrew"] = cq.pop("question")
                                if "question_text" in cq and "question_text_hebrew" not in cq:
                                    cq["question_text_hebrew"] = cq.pop("question_text")
                                if "options" in cq and "options_hebrew" not in cq:
                                    cq["options_hebrew"] = cq.pop("options")
                                if "correct_index" in cq and "correct_option_index" not in cq:
                                    cq["correct_option_index"] = cq.pop("correct_index")

                        if "cloze_task" in at:
                            ct = at["cloze_task"]
                            if isinstance(ct, str):
                                at["cloze_task"] = {
                                    "sentence_with_blank": ct,
                                    "options": ["...", "...", "..."],
                                    "correct_option_index": 0,
                                    "translation_of_blank_word_hebrew": "..."
                                }
                            elif isinstance(ct, dict):
                                if "sentence" in ct and "sentence_with_blank" not in ct:
                                    ct["sentence_with_blank"] = ct.pop("sentence")
                                if "sentence_text" in ct and "sentence_with_blank" not in ct:
                                    ct["sentence_with_blank"] = ct.pop("sentence_text")
                                if "correct_index" in ct and "correct_option_index" not in ct:
                                    ct["correct_option_index"] = ct.pop("correct_index")

                    # Normalize story_branches
                    if "branches" in raw_json and "story_branches" not in raw_json:
                        raw_json["story_branches"] = raw_json.pop("branches")
                    if "story_branches" in raw_json and isinstance(raw_json["story_branches"], list):
                        for i, branch in enumerate(raw_json["story_branches"]):
                            if not isinstance(branch, dict): continue
                            if "english" in branch and "text_english" not in branch:
                                branch["text_english"] = branch.pop("english")
                            if "hebrew" in branch and "text_hebrew" not in branch:
                                branch["text_hebrew"] = branch.pop("hebrew")
                            if "choice_id" not in branch:
                                branch["choice_id"] = i + 1

                    data = models.ActContentResponse(**raw_json)
                    logger.info("Manual parse fallback succeeded for act content.")
                    return data
                except Exception as parse_err:
                    logger.error(f"Manual parse fallback failed for act content: {parse_err}")
                    raise ValueError(f"Invalid LLM response and fallback failed: {raw_text}")
            else:
                raise ValueError("Invalid LLM response: Empty body or blocked")
        else:
            return response.parsed
    except Exception as e:
        logger.exception("CRITICAL: Failed generate_act_content")
        raise e

def evaluate_assessment_performance(submission: models.AssessmentSubmission) -> models.AssessmentFeedback:
    client = get_client()
    num_correct = sum(1 for k in submission.answers if submission.answers[k] == submission.correct_answers.get(k))
    prompt = f"""
    The student got {num_correct} questions correct out of {len(submission.correct_answers)}.
    Provide feedback in Hebrew.

    Output MUST be a strict JSON matching AssessmentFeedback schema:
    - is_correct: bool
    - explanation_hebrew: str
    - suggested_state_updates: str
    - encouragement_message_hebrew: str
    """
    try:
        response = client.models.generate_content(
            model=SCENE_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )

        raw_text = getattr(response, 'text', 'None')
        try:
            # Strip markdown if present
            clean_json = raw_text
            if "```json" in raw_text:
                clean_json = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                clean_json = raw_text.split("```")[1].split("```")[0].strip()

            raw_json = json.loads(clean_json)
            # Ensure all fields are present for Pydantic
            if "suggested_state_updates" not in raw_json:
                raw_json["suggested_state_updates"] = ""

            return models.AssessmentFeedback(**raw_json)
        except Exception as parse_err:
            logger.error(f"Manual parse failed for assessment feedback: {parse_err}")
            # Minimum viable fallback
            return models.AssessmentFeedback(
                is_correct=num_correct > 0,
                explanation_hebrew="כל הכבוד על המאמץ!",
                suggested_state_updates="",
                encouragement_message_hebrew="תמשיך ככה!"
            )
    except Exception as e:
        logger.exception("CRITICAL: Failed evaluate_assessment_performance")
        raise e

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
    prompt = f"Generate a {request.cefr_level} CEFR exam for the student. Output as strict JSON matching ExamResponse schema."
    try:
        response = client.models.generate_content(
            model=EXAM_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        raw_text = getattr(response, 'text', 'None')
        try:
            # Strip markdown if present
            clean_json = raw_text
            if "```json" in raw_text:
                clean_json = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                clean_json = raw_text.split("```")[1].split("```")[0].strip()

            raw_json = json.loads(clean_json)
            return models.ExamResponse(**raw_json)
        except Exception as parse_err:
            logger.error(f"Manual parse failed for CEFR exam: {parse_err}")
            raise ValueError(f"Invalid Exam Response: {raw_text}")
    except Exception as e:
        logger.exception("CRITICAL: Failed generate_cefr_exam")
        raise e
