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
    current_estimated_level: str = Field(..., example="A1-Sub1", description="Level like A1-Sub1 or 'unknown'")
    covered_categories: Dict[str, str] = Field(default_factory=dict, description="e.g., {'space_objects': 'completed'}")
    assessment_history: List[AssessmentRecord] = Field(default_factory=list)
    story_preferences: StoryPreferences = Field(default_factory=StoryPreferences)
    total_scenes_completed: int = 0
    current_rank_index: int = 100 # New field for CEFR rank index

# --- Adventure Setup Models (The Launchpad) ---

class AdventureSetupRequest(BaseModel):
    rank_index: int
    genre: str
    num_words: int = 100
    pct_above: float = 0.1

class LaunchpadAnchor(BaseModel):
    id: str
    text: str
    description: str

class ActBlueprint(BaseModel):
    act_number: int
    title: str
    starting_point: str
    ending_point: str
    description: str
    branch_options: List[str] = Field(default_factory=list)

class StoryArc(BaseModel):
    title: str
    acts: List[ActBlueprint]

class AdventureSetupResponse(BaseModel):
    heroes: List[LaunchpadAnchor] = Field(default_factory=list)
    settings: List[LaunchpadAnchor] = Field(default_factory=list)
    catalysts: List[LaunchpadAnchor] = Field(default_factory=list)
    potential_story_arcs: List[StoryArc] = Field(default_factory=list)
    selected_vocabulary: List[str] = Field(default_factory=list)
    vocabulary_min_rank: int = 0
    vocabulary_max_rank: int = 0

class GuardrailRequest(BaseModel):
    data: AdventureSetupResponse
    target_rank: int

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
    goal: str

class PedagogicalDecision(BaseModel):
    category_name: str
    target_words: List[str] = Field(..., min_items=6, max_items=6)
    updated_level: str
    story_elements: Optional[StoryElements] = None

class InterviewChatResponse(BaseModel):
    chat_response: Optional[str] = None
    pedagogical_decision: Optional[PedagogicalDecision] = None
    is_final_turn: bool = False

# --- Hierarchical Story Models ---

class GenerateArcRequest(BaseModel):
    story_elements: StoryElements
    student_state: StudentState
    genre_theme: Optional[str] = None

class ActContentRequest(BaseModel):
    story_arc_title: str
    act_blueprint: ActBlueprint
    target_words: List[str] = Field(default_factory=list)
    student_state: StudentState
    word_count_target: int = 150
    hero_description: Optional[str] = None
    setting_description: Optional[str] = None
    genre: Optional[str] = None
    previous_act_title: Optional[str] = None
    next_act_title: Optional[str] = None

# --- Assessment & Scene Models ---

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

class VocabularyDefinition(BaseModel):
    word: str
    definition_hebrew: str

class ActContentResponse(BaseModel):
    act_number: int
    scene_text: str
    remedial_scene_text: str
    vocabulary_definitions: List[VocabularyDefinition]
    used_vocabulary: List[str] = Field(default_factory=list)
    assessment_tasks: AssessmentTasks
    story_branches: List[StoryBranch]

# --- Assessment Evaluation Models ---

class AssessmentSubmission(BaseModel):
    student_state: StudentState
    category: str
    level: str
    answers: Dict[str, Any]
    correct_answers: Dict[str, Any]

class AssessmentFeedback(BaseModel):
    is_correct: bool
    explanation_hebrew: str
    suggested_state_updates: Dict[str, Any]
    encouragement_message_hebrew: str

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
