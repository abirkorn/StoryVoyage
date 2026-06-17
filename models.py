from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# --- Common Models ---

class StudentState(BaseModel):
    current_estimated_level: str = Field(..., example="A1-Sub1")
    covered_categories: Dict[str, str] = Field(default_factory=dict, description="e.g., {'space_objects': 'completed'}")

# --- Interview Chat Models ---

class ChatMessage(BaseModel):
    role: str # 'user' or 'model'
    content: str

class InterviewChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    student_state: StudentState

class PedagogicalDecision(BaseModel):
    category_name: str
    target_words: List[str] = Field(..., min_items=6, max_items=6)
    updated_level: str

class InterviewChatResponse(BaseModel):
    chat_response: Optional[str] = None
    pedagogical_decision: Optional[PedagogicalDecision] = None
    is_final_turn: bool = False

# --- Scene Generation Models ---

class GenerateSceneRequest(BaseModel):
    category: str
    target_words: List[str]
    plot_history: List[str] = Field(default_factory=list)
    student_state: StudentState

class ComprehensionQuestion(BaseModel):
    question_text_hebrew: str
    options_hebrew: List[str]
    correct_option_index: int
    explanation_hebrew: str

class ClozeTask(BaseModel):
    sentence_with_blank: str
    options: List[str]
    correct_option_index: int
    translation_of_blank_word_hebrew: str

class AssessmentTasks(BaseModel):
    comprehension_question: ComprehensionQuestion
    cloze_task: ClozeTask

class StoryBranch(BaseModel):
    choice_id: int
    text_hebrew: str
    text_english: str

class SceneMetadata(BaseModel):
    scene_id: int
    adaptive_level_applied: str

class SceneResponse(BaseModel):
    metadata: SceneMetadata
    scene_text: str
    remedial_scene_text: str
    target_words: List[str]
    assessment_tasks: AssessmentTasks
    story_branches: List[StoryBranch]

# --- CEFR Exam Models ---

class GenerateExamRequest(BaseModel):
    cefr_level: str
    scenes_data: List[Dict[str, Any]] = Field(default_factory=list)

class ExamQuestion(BaseModel):
    question_number: int
    type: str # comprehension, vocabulary, grammar, cause/effect, inference
    question_text_hebrew: str
    options_hebrew: List[str]
    correct_option_index: int
    explanation_hebrew: str
    difficulty: Optional[str] = None

class ExamResponse(BaseModel):
    exam_title_hebrew: str
    cefr_level: str
    instructions_hebrew: str
    questions: List[ExamQuestion]
    passing_score: int
