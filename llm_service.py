import os
import json
import logging
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import models
from vocabulary_engine import VocabularyEngine

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

def get_target_structure(level: str) -> Dict[str, int]:
    """
    Determines the target narrative structure based on CEFR level.
    """
    lvl = level.upper()
    if any(p in lvl for p in ["PRE-A1", "A1"]):
        return {"num_paragraphs": 2, "sentences_per_paragraph": 3}
    if "A2" in lvl:
        return {"num_paragraphs": 3, "sentences_per_paragraph": 4}
    if any(p in lvl for p in ["B1", "B2", "C1", "C2"]):
        return {"num_paragraphs": 4, "sentences_per_paragraph": 5}

    # Default fallback
    return {"num_paragraphs": 3, "sentences_per_paragraph": 4}

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

def generate_story_options(request: models.AdventureSetupRequest) -> models.AdventureSetupResponse:
    logger.info(f"Generating Story Options. Genre: {request.genre}, Rank: {request.rank_index}")

    engine = VocabularyEngine()
    words = engine.fetch_vocabulary(
        target_rank=request.rank_index,
        semantic_query=request.semantic_query or request.genre,
        pos_distribution={"n.": 0.5, "adj.": 0.3, "v.": 0.2},
        state_distribution={models.WordState.UNSEEN: 0.9, models.WordState.LEARNING: 0.1},
        word_count=40
    )

    client = get_client()
    prompt = f"""
    You are an expert Story Architect. Generate EXACTLY 3 distinct story premises for a child.

    THEME/GENRE: {request.genre}
    INSPIRATION VOCABULARY: {", ".join(words)}

    TASK:
    Generate a list of 3 StoryPremise objects.
    Each Premise MUST include:
    - 'id': 'p1', 'p2', or 'p3'
    - 'title': A catchy name for the story.
    - 'hero': A brief description (e.g., "Coco - a brave cat who fears water").
    - 'setting': A vivid location (e.g., "The Whispering Woods where trees move").
    - 'catalyst': The event that starts the adventure (e.g., "Finding a map that changes its route").

    Output MUST be strict JSON matching models.AdventureSetupResponse schema (specifically the 'premises' field).
    """

    try:
        response = client.models.generate_content(
            model=SCENE_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )

        raw_text = response.text
        # Apply normalization fallback if needed
        try:
            clean_json = raw_text
            if "```json" in raw_text:
                clean_json = raw_text.split("```json")[1].split("```")[0].strip()
            raw_json = json.loads(clean_json)
            data = models.AdventureSetupResponse(**raw_json)
        except Exception:
            # Simple manual parse if response is a raw list
            data = models.AdventureSetupResponse(premises=json.loads(clean_json), selected_vocabulary=words)

        data.selected_vocabulary = words
        return data
    except Exception as e:
        logger.exception("Failed to generate story options")
        raise e

def generate_story_dag(request: models.GenerateDAGRequest) -> models.StoryDAG:
    logger.info(f"Generating Story DAG for premise: {request.premise.title}")

    # Fresh list of 25 words from VocabularyEngine
    engine = VocabularyEngine()
    words = engine.fetch_vocabulary(
        target_rank=request.student_state.current_rank_index,
        semantic_query=f"{request.premise.title} {request.premise.setting}",
        pos_distribution={"n.": 0.4, "adj.": 0.4, "v.": 0.2},
        state_distribution={models.WordState.UNSEEN: 0.8, models.WordState.LEARNING: 0.2},
        word_count=25
    )

    client = get_client()

    if not os.path.exists(STORY_LOGIC_PATH):
        story_logic = "Use Directed Acyclic Graph (DAG) for Acts (Levels 0-3)."
    else:
        with open(STORY_LOGIC_PATH, "r") as f:
            story_logic = f.read()

    prompt = f"""
    You are an expert interactive fiction designer. Create a Directed Acyclic Graph (DAG) for Act Blueprints.

    PREMISE: {request.premise.model_dump_json()}
    VOCABULARY POOL: {", ".join(words)}

    CYOA STRUCTURE LAWS:
    {story_logic}

    TASK:
    1. Create a Directed Acyclic Graph (DAG) of nodes (Levels 0 to 3).
    2. Level 0: The 'entry_node'.
    3. Level 1 & 2: Internal nodes. Each MUST branch into 2 next_node_ids.
    4. Level 3: Terminal nodes (ending_point reached, next_node_ids empty).
    5. Structural consistency: Ensure no dead ends before Level 3. Every node at Level L must point to nodes at Level L+1.

    SCHEMA for each node in 'nodes' (Dict[id, object]):
    - 'node_id': unique string (e.g. 'n0', 'n1_1', 'n1_2', 'n2_1'...)
    - 'act_number': (integer 1-5+)
    - 'level': (integer 0-3)
    - 'title': short title
    - 'description': high-level overview
    - 'plot_beats': List of 4-5 specific chronological events
    - 'starting_point': text
    - 'ending_point': text
    - 'branch_options': List of 2 options (except level 3)
    - 'next_node_ids': List of ids

    Output MUST be a strict JSON matching models.StoryDAG.
    """

    try:
        response = client.models.generate_content(
            model=SCENE_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )

        raw_text = response.text
        clean_json = raw_text
        if "```json" in raw_text:
            clean_json = raw_text.split("```json")[1].split("```")[0].strip()

        raw_json = json.loads(clean_json)

        # Ensure premise_id is present
        if "premise_id" not in raw_json:
            raw_json["premise_id"] = request.premise.id

        # Robust normalization for 'nodes' which might be a list or dict
        if "nodes" in raw_json and isinstance(raw_json["nodes"], list):
            new_nodes = {}
            for node in raw_json["nodes"]:
                if isinstance(node, dict) and "node_id" in node:
                    new_nodes[node["node_id"]] = node
            raw_json["nodes"] = new_nodes

        # Simple normalization if 'entry_node_id' is missing
        if "entry_node_id" not in raw_json and "nodes" in raw_json:
            # Pick the level 0 node
            for nid, node in raw_json["nodes"].items():
                if node.get("level") == 0:
                    raw_json["entry_node_id"] = nid
                    break

            # Fallback if no level 0 found
            if "entry_node_id" not in raw_json and raw_json["nodes"]:
                raw_json["entry_node_id"] = list(raw_json["nodes"].keys())[0]

        return models.StoryDAG(**raw_json)
    except Exception as e:
        logger.exception("Failed to generate story DAG")
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


def generate_act_content(request: models.ActContentRequest) -> models.ActContentResponse:
    logger.info(f"Generating Act {request.node.act_number} content for {request.story_title}...")
    client = get_client()
    node = request.node

    # Dynamic structure logic
    struct = get_target_structure(request.student_state.current_estimated_level)
    num_paragraphs = request.num_paragraphs or struct["num_paragraphs"]
    sentences_per_paragraph = request.sentences_per_paragraph or struct["sentences_per_paragraph"]

    prompt = f"""
    You are an expert Children's Book Author. Write Act {node.act_number} of an interactive story.

    STORY CONTEXT:
    - TITLE: {request.story_title}
    - HERO: {request.hero_description}
    - SETTING: {request.setting_description}
    - CATALYST: {request.catalyst_description}
    - PLOT BEATS: {", ".join(node.plot_beats)}
    - BRIDGE: Start at '{node.starting_point}' and end exactly at '{node.ending_point}'.

    LITERARY CONSTRAINTS (LEVEL: {request.student_state.current_estimated_level}):
    - VOICE: Warm and vivid. Use sensory details and internal feelings.
    - STRUCTURE: You MUST output 'scene_paragraphs' as an array of arrays: [[sentence1, sentence2...], [sentence1...]].
    - CONSTRAINT: EXACTLY {num_paragraphs} paragraphs, each with EXACTLY {sentences_per_paragraph} sentences.
    - VOCABULARY POOL: {", ".join(request.target_words)}.
    - REQUIREMENT: Weave in at least 10 words from the VOCABULARY POOL.
    - ENDING: The very last sentence of the final paragraph must perfectly set up these choices: {", ".join(node.branch_options)}.

    JSON OUTPUT REQUIREMENTS:
    1. 'scene_paragraphs': List of lists of sentences as defined above.
    2. 'used_vocabulary': List of words used from the pool.
    3. 'remedial_scene_text': Storytelling Hebrew translation.
    4. 'vocabulary_definitions': Hebrew definitions for used words (List of {{"word": "...", "definition_hebrew": "..."}}).
    5. 'assessment_tasks':
       - 'comprehension_question': Hebrew question + 3 options + correct_option_index.
       - 'cloze_task': English sentence from your text with one word replaced by a blank + 3 options + correct_option_index.
    6. 'story_branches': List of {{"choice_id": 1, "text_english": "...", "text_hebrew": "..."}} matching the options provided.

    Strict valid JSON matching ActContentResponse. No markdown.
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
