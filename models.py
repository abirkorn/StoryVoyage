from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# --- Common Models ---

class AssessmentRecord(BaseModel):
    category: str = ""
    level: str = ""
    score: float = 0.0
    completed_at: str = ""
    assessment_type: str = "" # "scene" or "exam"

class StoryPreferences(BaseModel):
    hero_name: Optional[str] = None
    hero_description: Optional[str] = None
    theme: Optional[str] = None
    favorite_topics: List[str] = Field(default_factory=list)
    avoid_topics: List[str] = Field(default_factory=list)

class StudentState(BaseModel):
    current_estimated_level: str = "unknown"
    covered_categories: Dict[str, str] = Field(default_factory=dict)
    assessment_history: List[AssessmentRecord] = Field(default_factory=list)
    story_preferences: StoryPreferences = Field(default_factory=StoryPreferences)
    total_scenes_completed: int = 0
    current_rank_index: int = 100

# --- Adventure Setup Models (The Launchpad) ---

class AdventureSetupRequest(BaseModel):
    rank_index: int
    genre: str
    num_words: int = 100
    pct_above: float = 0.1

class LaunchpadAnchor(BaseModel):
    id: str = ""
    text: str = ""
    description: str = ""

class ActBlueprint(BaseModel):
    act_number: int = 0
    title: str = ""
    starting_point: str = ""
    ending_point: str = ""
    description: str = ""
    branch_options: List[str] = Field(default_factory=list)

class StoryArc(BaseModel):
    title: str = ""
    hero_id: str = ""
    setting_id: str = ""
    catalyst_id: str = ""
    acts: List[ActBlueprint] = Field(default_factory=list)

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
    hero_name: str = ""
    setting: str = ""
    goal: str = ""

class PedagogicalDecision(BaseModel):
    category_name: str = ""
    target_words: List[str] = Field(default_factory=list)
    updated_level: str = ""
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
    question_id: str = ""
    question_text_hebrew: str = ""
    options_hebrew: List[str] = Field(default_factory=list)
    correct_option_index: int = 0
    explanation_hebrew: str = ""

class ClozeTask(BaseModel):
    task_id: str = ""
    sentence_with_blank: str = ""
    options: List[str] = Field(default_factory=list)
    correct_option_index: int = 0
    translation_of_blank_word_hebrew: str = ""

class AssessmentTasks(BaseModel):
    comprehension_question: Optional[ComprehensionQuestion] = None
    cloze_task: Optional[ClozeTask] = None

class StoryBranch(BaseModel):
    choice_id: int = 0
    text_hebrew: str = ""
    text_english: str = ""

class VocabularyDefinition(BaseModel):
    word: str = ""
    definition_hebrew: str = ""

class ActContentResponse(BaseModel):
    act_number: int = 0
    scene_text: str = ""
    remedial_scene_text: str = ""
    vocabulary_definitions: List[VocabularyDefinition] = Field(default_factory=list)
    used_vocabulary: List[str] = Field(default_factory=list)
    assessment_tasks: Optional[AssessmentTasks] = None
    story_branches: List[StoryBranch] = Field(default_factory=list)

# --- Assessment Evaluation Models ---

class AssessmentSubmission(BaseModel):
    student_state: StudentState
    category: str
    level: str
    answers: Dict[str, Any]
    correct_answers: Dict[str, Any]

class AssessmentFeedback(BaseModel):
    is_correct: bool = False
    explanation_hebrew: str = ""
    suggested_state_updates: str = ""
    encouragement_message_hebrew: str = ""

# --- CEFR Exam Models ---

class GenerateExamRequest(BaseModel):
    cefr_level: str
    scenes_data: List[Dict[str, Any]] = Field(default_factory=list)
    student_state: StudentState

class ExamQuestion(BaseModel):
    question_number: int = 0
    type: str = "" # comprehension, vocabulary, grammar, cause/effect, inference
    question_text_hebrew: str = ""
    options_hebrew: List[str] = Field(default_factory=list)
    correct_option_index: int = 0
    explanation_hebrew: str = ""
    difficulty: str = "medium"

class ExamResponse(BaseModel):
    exam_title_hebrew: str = ""
    cefr_level: str = ""
    instructions_hebrew: str = ""
    questions: List[ExamQuestion] = Field(default_factory=list)
    passing_score: int = 70
