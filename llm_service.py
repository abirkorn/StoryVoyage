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
    Handles the "Story Setup" chat with the child (Hebrew/English mix).
    Aims to build the story context while assessing the initial English level.
    """
    
    system_prompt = f"""
    You are an expert ESL pedagogical assistant for kids. 
    Your goal is to help a child "build" their next story adventure.
    
    STUDENT STATE:
    - Current Level: {request.student_state.current_estimated_level}
    - Assessment History: {json.dumps([r.dict() for r in request.student_state.assessment_history])}
    - Covered Categories: {json.dumps(request.student_state.covered_categories)}
    - Total Scenes: {request.student_state.total_scenes_completed}
    
    GUIDELINES:
    1. Conduct the conversation primarily in HEBREW, as the child might not speak English well yet. Use simple English occasionally.
    2. Be extremely encouraging, magical, and friendly.
    3. Ask about:
       - Who is the hero? (Name, what are they like?)
       - Where does it take place? (Space, jungle, magic school, etc.)
       - What is the main theme or interest the child wants to explore?
    4. Use this interaction to subtly assess if the child's level should be adjusted (though usually, we stick to the current level unless they seem very advanced/struggling).
    5. After 3-4 turns, when you have enough info, conclude the setup.
    
    OUTPUT FORMAT:
    - If continuing the chat: Return a warm response in Hebrew.
    - If concluding: Return a JSON object with:
    {{
        "pedagogical_decision": {{
            "category_name": "string",
            "target_words": ["word1", "word2", "word3", "word4", "word5", "word6"],
            "updated_level": "string",
            "story_elements": {{
                "hero_name": "string",
                "setting": "string",
                "initial_plot_point": "string"
            }}
        }}
    }}
    """
    
    contents = [types.Content(role="user", parts=[types.Part(text=system_prompt)])]
    for msg in request.history:
        # Map 'model' role to 'model' for Gemini
        role = "model" if msg.role == "model" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
    
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
    """
    Evaluates the student's performance on a scene's assessment tasks and determines pedagogical adjustments.
    """
    client = get_client()

    # Simple logic combined with LLM insight for the "Pedagogical Strategy"
    num_correct = sum(1 for k in submission.answers if submission.answers[k] == submission.correct_answers.get(k))
    total_questions = len(submission.correct_answers)
    score = num_correct / total_questions if total_questions > 0 else 0

    prompt = f"""
    Evaluate the child's performance on their recent ESL tasks.

    Score: {num_correct}/{total_questions}
    Current Level: {submission.level}
    Category: {submission.category}

    Student State Context: {submission.student_state.json()}

    Task:
    1. Provide a warm, encouraging explanation in Hebrew about how they did.
    2. Decide on a pedagogical update:
       - Should they 'Level Up' (e.g., A1-Sub1 -> A1-Sub2)?
       - Should they 'Stay' at the current level?
       - Should we change the 'Explore/Exploit' strategy?

    Output a JSON matching the AssessmentFeedback model.
    """

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
    """
    Generates a story scene based on pedagogical constraints, plot history, and story elements.
    """
    client = get_client()

    story_context = ""
    if request.story_elements:
        story_context = f"""
        STORY ELEMENTS:
        - Hero: {request.story_elements.hero_name}
        - Setting: {request.story_elements.setting}
        - Initial Plot: {request.story_elements.initial_plot_point}
        """

    system_prompt = f"""
    You are a creative ESL storyteller for kids. Generate the next scene in an ongoing story.
    
    PEDAGOGICAL CONSTRAINTS:
    - Level: {request.student_state.current_estimated_level}
    - Category: {request.category}
    - Target Words (MUST include these): {", ".join(request.target_words)}
    
    {story_context}

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
    Student State: {request.student_state.json()}
    
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
