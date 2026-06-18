from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# --- Common Models ---

class AssessmentRecord(BaseModel):
    category: str
    level: str
    score: float # 0.0 to 1.0
    completed_at: str # ISO timestamp or simple date
    assessment_type: str # "scene" or "exam"

class StoryPreferences(BaseModel):
    hero_name: Optional[str] = None
    hero_description: Optional[str] = None
    theme: Optional[str] = None # e.g., "Space", "Fantasy", "Dinosaurs"
    favorite_topics: List[str] = Field(default_factory=list)
    avoid_topics: List[str] = Field(default_factory=list)

class StudentState(BaseModel):
    current_estimated_level: str = Field(..., example="A1-Sub1")
    covered_categories: Dict[str, str] = Field(default_factory=dict, description="e.g., {'space_objects': 'completed'}")
    assessment_history: List[AssessmentRecord] = Field(default_factory=list)
    story_preferences: StoryPreferences = Field(default_factory=StoryPreferences)
    total_scenes_completed: int = 0

# --- Interview Chat Models ---

class ChatMessage(BaseModel):
    role: str # 'user' or 'model'
    content: str

class InterviewChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    student_state: StudentState

class StoryElements(BaseModel):
    hero_name: str
    setting: str
    initial_plot_point: str

class PedagogicalDecision(BaseModel):
    category_name: str
    target_words: List[str] = Field(..., min_items=6, max_items=6)
    updated_level: str
    story_elements: Optional[StoryElements] = None

class InterviewChatResponse(BaseModel):
    chat_response: Optional[str] = None
    pedagogical_decision: Optional[PedagogicalDecision] = None
    is_final_turn: bool = False

# --- Assessment Evaluation Models ---

class AssessmentSubmission(BaseModel):
    student_state: StudentState
    category: str
    level: str
    answers: Dict[str, Any] # choice_id or text
    correct_answers: Dict[str, Any]

class AssessmentFeedback(BaseModel):
    is_correct: bool
    explanation_hebrew: str
    suggested_state_updates: Dict[str, Any]
    encouragement_message_hebrew: str

# --- Scene Generation Models ---

class GenerateSceneRequest(BaseModel):
    category: str
    target_words: List[str]
    plot_history: List[str] = Field(default_factory=list)
    student_state: StudentState
    story_elements: Optional[StoryElements] = None

class ComprehensionQuestion(BaseModel):
    question_id: str
    question_text_hebrew: str
    options_hebrew: List[str]
    correct_option_index: int
    explanation_hebrew: str

class ClozeTask(BaseModel):
    task_id: str
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
    student_state: StudentState

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
