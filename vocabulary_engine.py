import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import models

class VocabularyEngine:
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VocabularyEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, catalog_path: str = "cefr_catalog.json"):
        if hasattr(self, 'initialized'):
            return
        self.catalog_path = catalog_path
        self.words: List[models.VocabularyWord] = []
        self._load_catalog()
        self.initialized = True

    def _get_model(self):
        if self._model is None:
            # Local fast embedding model
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def _load_catalog(self):
        if not os.path.exists(self.catalog_path):
            return
        with open(self.catalog_path, "r") as f:
            data = json.load(f)
            for item in data:
                self.words.append(models.VocabularyWord(
                    w=item["w"],
                    pos=item["pos"],
                    rank=item["rank"],
                    state=models.WordState.UNSEEN, # Default
                    theme=item.get("theme")
                ))

        # Pre-compute embeddings for all words to speed up fetching
        # In a real production app, we'd cache these or use a vector DB
        texts = [word.w for word in self.words]
        model = self._get_model()
        self.word_embeddings = model.encode(texts, convert_to_numpy=True)

    def fetch_vocabulary(
        self,
        target_rank: int,
        semantic_query: str,
        pos_distribution: Dict[str, float],
        state_distribution: Dict[models.WordState, float],
        word_count: int,
        diversity_penalty: float = 0.5,
        rank_radius: int = 1000
    ) -> List[str]:
        """
        Fetches words based on rank, semantic relevance, and diversity (MMR).
        """
        model = self._get_model()
        query_embedding = model.encode([semantic_query], convert_to_numpy=True)

        # 1. Initial Filtering
        candidates_indices = []
        for i, word in enumerate(self.words):
            # LEARNING bypasses rank filter
            if word.state == models.WordState.LEARNING:
                candidates_indices.append(i)
                continue

            # UNSEEN must be within radius
            if word.state == models.WordState.UNSEEN:
                if abs(word.rank - target_rank) <= rank_radius:
                    candidates_indices.append(i)
                    continue

            # MASTERED (Optional logic, usually include them for reinforcement)
            if word.state == models.WordState.MASTERED:
                candidates_indices.append(i)

        if not candidates_indices:
            # Fallback if too restrictive
            candidates_indices = list(range(len(self.words)))

        candidate_embeddings = self.word_embeddings[candidates_indices]

        # 2. Semantic Relevance (Cosine Similarity)
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # 3. MMR Selection logic
        # We need to respect pos_distribution and state_distribution
        # To simplify, we'll calculate target counts for each bucket
        buckets = {}
        for pos, p_pct in pos_distribution.items():
            for state, s_pct in state_distribution.items():
                target = int(word_count * p_pct * s_pct)
                if target > 0:
                    buckets[(pos, state)] = target

        # Handle rounding errors to ensure exact word_count
        current_total = sum(buckets.values())
        if current_total < word_count:
            # Add to the largest bucket
            if buckets:
                max_key = max(buckets, key=buckets.get)
                buckets[max_key] += (word_count - current_total)
            else:
                # Distribution was empty? Fallback.
                buckets[("n.", models.WordState.UNSEEN)] = word_count

        selected_indices = []
        final_words = []

        # MMR Algorithm adapted for buckets
        # Note: True MMR is global, but we have bucket constraints.
        # We'll fill buckets one by one using a diverse selection from the filtered set.

        remaining_indices = set(range(len(candidates_indices)))

        for (target_pos, target_state), target_num in buckets.items():
            bucket_selected = 0
            while bucket_selected < target_num and remaining_indices:
                best_score = -np.inf
                best_idx = -1

                # Filter remaining by POS and State for this bucket
                valid_for_bucket = []
                for idx in remaining_indices:
                    orig_idx = candidates_indices[idx]
                    word = self.words[orig_idx]
                    if word.pos == target_pos and word.state == target_state:
                        valid_for_bucket.append(idx)

                if not valid_for_bucket:
                    # If we can't find exact match, try to just match POS or just State?
                    # For now, break and we'll fill leftovers later
                    break

                for idx in valid_for_bucket:
                    rel_score = similarities[idx]

                    if not selected_indices:
                        score = rel_score
                    else:
                        # Diversity score: max similarity to any already selected word
                        div_score = np.max(cosine_similarity(
                            candidate_embeddings[idx:idx+1],
                            self.word_embeddings[selected_indices]
                        ))
                        # MMR = lambda * Rel - (1-lambda) * Div
                        # diversity_penalty maps to (1-lambda)
                        score = (1 - diversity_penalty) * rel_score - diversity_penalty * div_score

                    if score > best_score:
                        best_score = score
                        best_idx = idx

                if best_idx != -1:
                    orig_idx = candidates_indices[best_idx]
                    selected_indices.append(orig_idx)
                    final_words.append(self.words[orig_idx].w)
                    remaining_indices.remove(best_idx)
                    bucket_selected += 1
                else:
                    break

        # Fill any remaining slots if buckets couldn't be satisfied
        while len(final_words) < word_count and remaining_indices:
            # Just pick most similar remaining
            available_sims = [(i, similarities[i]) for i in remaining_indices]
            available_sims.sort(key=lambda x: x[1], reverse=True)
            best_rem_idx = available_sims[0][0]
            orig_idx = candidates_indices[best_rem_idx]
            final_words.append(self.words[orig_idx].w)
            selected_indices.append(orig_idx)
            remaining_indices.remove(best_rem_idx)

        return final_words[:word_count]
