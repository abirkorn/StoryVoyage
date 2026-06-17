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
    Handles the interview/evaluation chat with the child.
    Determines if it's the final turn to produce a pedagogical decision.
    """
    
    system_prompt = f"""
    You are an expert ESL pedagogical assistant for kids. 
    Your goal is to interview a child to evaluate their English level and interest to decide on the next learning category.
    
    Student Current Level: {request.student_state.current_estimated_level}
    Previously Covered Categories: {json.dumps(request.student_state.covered_categories)}
    
    Guidelines:
    1. Keep the conversation friendly, simple, and encouraging for a child.
    2. Conduct the interview in a mix of simple English and Hebrew if needed (but primarily English for the child to respond to).
    3. You have a maximum of 3-4 turns total in the history.
    4. On the final turn, you MUST provide a structured JSON decision.
    
    If you are NOT ready to make a decision, return a plain text message for the child.
    If you ARE ready (usually after 3-4 turns), return a JSON object with the following structure:
    {{
        "final_decision": {{
            "category_name": "string",
            "target_words": ["word1", "word2", "word3", "word4", "word5", "word6"],
            "updated_level": "string"
        }}
    }}
    """
    
    contents = [types.Content(role="user", parts=[types.Part(text=system_prompt)])]
    for msg in request.history:
        contents.append(types.Content(role=msg.role, parts=[types.Part(text=msg.content)]))
    
    contents.append(types.Content(role="user", parts=[types.Part(text=request.message)]))
    
    response = client.models.generate_content(
        model=SCENE_MODEL_ID,
        contents=contents
    )
    
    text = response.text.strip()
    
    # Try to extract JSON if it's present
    try:
        json_text = text
        if "```json" in text:
            json_text = text.split("```json")[1].split("```")[0].strip()
        elif "{" in text:
            json_text = text[text.find("{"):text.rfind("}")+1]
            
        data = json.loads(json_text)
        if "final_decision" in data or "category_name" in data:
            decision_data = data.get("final_decision", data)
            return models.InterviewChatResponse(
                pedagogical_decision=models.PedagogicalDecision(
                    category_name=decision_data["category_name"],
                    target_words=decision_data["target_words"],
                    updated_level=decision_data["updated_level"]
                ),
                is_final_turn=True
            )
    except:
        pass
        
    return models.InterviewChatResponse(chat_response=text, is_final_turn=False)

def generate_story_scene(request: models.GenerateSceneRequest) -> models.SceneResponse:
    """
    Generates a story scene based on pedagogical constraints and plot history.
    """
    client = get_client()
    system_prompt = f"""
    You are a creative ESL storyteller for kids. Generate the next scene in an ongoing story.
    
    PEDAGOGICAL CONSTRAINTS:
    - Level: {request.student_state.current_estimated_level}
    - Category: {request.category}
    - Target Words (MUST include these): {", ".join(request.target_words)}
    
    STORY CONTEXT:
    - Plot History: {json.dumps(request.plot_history)}
    
    REQUIREMENTS:
    - scene_text: Strictly in English. Simple, engaging, and appropriate for the level.
    - remedial_scene_text: A simplified version of the scene_text in Hebrew.
    - assessment_tasks: 
        - comprehension_question: In Hebrew, testing understanding of the scene.
        - cloze_task: A sentence in English with one of the target words missing. Options and translation in Hebrew.
    - story_branches: 2-3 choices for the child to continue the story. text_english and text_hebrew required.
    
    Output MUST be a strict JSON matching the SceneResponse schema.
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
    """
    Generates a CEFR level-up exam.
    """
    client = get_client()
    system_prompt = f"""
    You are an expert ESL examiner. Generate a CEFR level-up exam for level: {request.cefr_level}.
    
    Context from previous scenes: {json.dumps(request.scenes_data)}
    
    REQUIREMENTS:
    - Generate 5 questions.
    - Question types: comprehension, vocabulary, grammar, cause/effect, and inference.
    - All text (titles, instructions, questions, options, explanations) MUST be in Hebrew, except where testing English specific terms.
    - Passing score is typically 4 out of 5.
    
    Output MUST be a strict JSON matching the ExamResponse schema.
    """
    
    response = client.models.generate_content(
        model=EXAM_MODEL_ID,
        contents=system_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=models.ExamResponse,
        )
    )
    
    return response.parsed
