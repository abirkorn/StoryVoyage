# StoryVoyage: Product & Domain Context

## 1. Conceptual Overview
StoryVoyage is a CEFR-based adaptive storytelling platform for ESL (English as a Second Language) learners, specifically tailored for native Hebrew speakers (kids). 
The system uses a continuous narrative model where story scenes are generated dynamically. This keeps the learner engaged in a personalized plot while strictly enforcing pedagogical and linguistic constraints.

## 2. Learning Progression & CEFR Mapping
The application tracks and advances the student's English proficiency based on the Common European Framework of Reference (CEFR):
* **Levels:** Ranges from A1 (Beginner) to C2 (Mastery).
* **Sub-Levels:** Each CEFR level is divided into 4 inner sub-levels (e.g., A1-Sub1, A1-Sub2... A1-Sub4).
* **Incremental Advancement:** Advancement within a CEFR level (moving up sub-levels) is incremental and driven by the child's performance in the daily interactive scenes.
* **Gated Level-Ups:** Moving from one full CEFR level to the next (e.g., A1 to B2) requires passing a comprehensive generative Level-Up Exam.

## 3. The Adaptive Pedagogical Loop
Every interaction in the app serves a pedagogical purpose, masked as a game:
1. **Evaluation (The Interview):** The system periodically chats with the child to gauge their interests and current vocabulary, deciding the next conversational "Category" and selecting 6 Target Words.
2. **Execution (The Scene):** The system generates a story chapter weaving in those target words.
3. **Assessment:** Every scene concludes with assessment tasks (Comprehension & Cloze tests) to verify the child understood the text and the target words.
4. **Branching:** The child chooses how the story continues, restarting the loop.

## 4. Language & Localization Rules
* **The Story:** Strictly in English, adapted dynamically to the current CEFR sub-level complexity.
* **The Remedial Text:** A simplified Hebrew translation of the story scene to help struggling learners.
* **Assessments & UI:** Comprehension questions, multiple-choice options, explanations, and exam instructions must be generated in **Hebrew** to test actual understanding rather than just pattern matching in English.
