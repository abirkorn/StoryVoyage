from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum

# --- Common Models ---

class WordState(str, Enum):
    MASTERED = "MASTERED"
    LEARNING = "LEARNING"
    UNSEEN = "UNSEEN"

class VocabularyWord(BaseModel):
    w: str
    pos: str
    rank: int
    state: WordState = WordState.UNSEEN
    theme: Optional[str] = None

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
    num_words: int = 40
    semantic_query: Optional[str] = None

class StoryPremise(BaseModel):
    id: str
    hero: str
    setting: str
    catalyst: str
    title: str

class AdventureSetupResponse(BaseModel):
    premises: List[StoryPremise] = Field(default_factory=list)
    selected_vocabulary: List[str] = Field(default_factory=list)

class DAGNode(BaseModel):
    node_id: str
    act_number: int
    level: int # Depth in the DAG (0-3 as requested)
    title: str
    description: str
    plot_beats: List[str] = Field(default_factory=list)
    starting_point: str
    ending_point: str
    branch_options: List[str] = Field(default_factory=list)
    next_node_ids: List[str] = Field(default_factory=list)

class StoryDAG(BaseModel):
    premise_id: str
    nodes: Dict[str, DAGNode] = Field(default_factory=dict)
    entry_node_id: str

class GenerateDAGRequest(BaseModel):
    premise: StoryPremise
    target_words: List[str] = Field(default_factory=list)
    student_state: StudentState

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
    story_title: str
    node: DAGNode
    target_words: List[str] = Field(default_factory=list)
    student_state: StudentState
    num_paragraphs: Optional[int] = None
    sentences_per_paragraph: Optional[int] = None
    hero_description: str
    setting_description: str
    catalyst_description: str
    genre: Optional[str] = None

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
    scene_paragraphs: List[List[str]] = Field(default_factory=list)
    scene_text: str = ""
    remedial_scene_text: str = ""
    vocabulary_definitions: List[VocabularyDefinition] = Field(default_factory=list)
    used_vocabulary: List[str] = Field(default_factory=list)
    assessment_tasks: Optional[AssessmentTasks] = None
    story_branches: List[StoryBranch] = Field(default_factory=list)

    @model_validator(mode='after')
    def flatten_scene_text(self) -> 'ActContentResponse':
        if self.scene_paragraphs:
            # Join sentences with spaces, and paragraphs with \n\n
            p_strings = [" ".join(sentences) for sentences in self.scene_paragraphs]
            self.scene_text = "\n\n".join(p_strings)
        return self

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
