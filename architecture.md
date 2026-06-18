StoryVoyage Backend Architecture
1. Core Philosophy: The Stateless Backend
The StoryVoyage backend is designed as a completely stateless pipe. It does not maintain a database (SQLite, PostgreSQL, etc.).

State Ownership: The "source of truth" for the student's progress is held by the Base44 Frontend User Entity.
State Passing: In every API request (/interview-chat, /generate-scene), the frontend must include the current StudentState (current level, covered categories, etc.) in the request body.
State Mutation: The backend processes the state, performs LLM operations, and returns any updated state (e.g., a new level or category) back to the frontend. The frontend is responsible for persisting these changes via base44.auth.updateMe.
2. Split Endpoint Design
The backend logic is decoupled into two primary workflows to maintain a clean separation of concerns:

A. The Pedagogical Engine (/interview-chat)
Responsibility: Conducts a multi-turn dialogue with the child to evaluate their English level and interests.
Workflow:
Turns 1–3: Returns a conversational string (chat) to the child.
Turn 4 (Final): Returns a structured PedagogicalDecision JSON containing the newly generated category and 6 specific target words.
Goal: Determine what the child should learn next.
B. The Narrative Execution Engine (/generate-scene)
Responsibility: Takes the pedagogical constraints (category, target words) and generates a narrative segment.
Workflow: Receives the chosen category, words, and plot_history to generate a scene that matches the child's proficiency level.
Schema Enforcement: Generates a complex JSON object containing:
scene_text (English)
remedial_scene_text (Hebrew)
assessment_tasks (Comprehension question + Cloze task in Hebrew)
story_branches (Branching choices in English and Hebrew)
3. LLM-First Approach (Generative Curriculum)
Unlike traditional learning apps with static levels, StoryVoyage uses a Fully Generative Curriculum.

No Hardcoded Paths: Complexity mappings, vocabulary lists, and story arcs are NOT hardcoded in the Python logic.
Prompt-Driven Logic: The pedagogical logic resides in the system prompts within llm_service.py.
Schema Safety: Pydantic models in models.py act as the "contract" between the LLM and the frontend, ensuring the AI output always matches the expected UI structure.
4. Technical Stack & Deployment
Framework: FastAPI (Python 3.11+) for high-performance, asynchronous routing.
LLM Provider: Google GenAI SDK (Gemini models).
gemini-3.1-flash-lite: Used for scenes and chat (optimized for speed/cost).
gemini-2.5-pro: Used for CEFR exams (optimized for accuracy/reasoning).
Developer Sandbox: A built-in UI at /sandbox allows for instant prompt tuning and simulation of the student state.
Security: Simple shared-secret validation using the X-App-Token header.
Environment Variables:
GEMINI_API_KEY: Google API credentials.
X_APP_TOKEN: Shared secret for frontend authentication.
CORS_ORIGINS: Allowed frontend domains.
PORT: Dynamically assigned by the cloud provider (e.g., Render, Cloud Run).
