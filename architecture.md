# StoryVoyage Backend Architecture

*Note: Please read `PRODUCT_CONTEXT.md` first to understand the pedagogical goals and CEFR rules of this project.*

## 1. Core Philosophy: The Stateless Backend
The StoryVoyage backend is a completely stateless "pipe". It does not maintain a database.
* **State Ownership:** The "source of truth" (CEFR level, sub-level, covered categories) is held by the Base44 Frontend User Entity.
* **State Passing:** In every API request, the frontend includes the `StudentState` in the request body.
* **State Mutation:** The backend processes the state via the LLM and returns the pedagogical results. The frontend is responsible for persisting any progress via `base44.auth.updateMe`.

## 2. Split Endpoint Design
The backend logic is decoupled into distinct endpoints mapping to the Product Context:

### A. The Pedagogical Engine (`/interview-chat`)
* **Responsibility:** Conducts a short dialogue to evaluate the child.
* **Goal:** Replaces hardcoded curriculum paths. On the final turn, it outputs a `PedagogicalDecision` JSON (next category + 6 target words).

### B. The Narrative Execution Engine (`/generate-scene`)
* **Responsibility:** Takes the pedagogical constraints and plot history to generate the next story segment.
* **Schema Enforcement:** Generates a strict JSON containing the English `scene_text`, Hebrew `remedial_scene_text`, Hebrew `assessment_tasks`, and bilingual `story_branches`.

### C. The Assessment Engine (`/generate-exam`)
* **Responsibility:** Generates the CEFR level-up exam when the student completes sub-level 4.
* **Workflow:** Takes the aggregated scenes data and outputs a 5-question exam (Comprehension, Vocabulary, Grammar, Inference) in Hebrew.

## 3. LLM-First Approach (Generative Curriculum)
We use a **Fully Generative Curriculum**. 
* **No Hardcoded Paths:** We DO NOT use hardcoded tables mapping levels to "Max Words/Sentence" or "Complexity". 
* **Prompt-Driven:** The intelligence relies entirely on the System Prompts in `llm_service.py` dictating how the LLM should interpret the CEFR level dynamically.
* **Validation:** Pydantic models in `models.py` act as the contract between the LLM and the frontend.

## 4. Technical Stack
* **Framework:** FastAPI (Python 3.11+).
* **LLM:** Google GenAI SDK. `gemini-3.1-flash-lite` for agile scene/chat generation, and `gemini-2.5-pro` for deep reasoning in CEFR exams.
* **Environment Variables:** `GEMINI_API_KEY`, `X_APP_TOKEN` (for basic frontend auth validation), `PORT`, `CORS_ORIGINS`.
