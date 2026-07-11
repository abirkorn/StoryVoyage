import json
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
import models

logger = logging.getLogger(__name__)

class VocabularyEngine:
    def __init__(self, catalog_path: str = "cefr_catalog.json"):
        self.catalog_path = catalog_path
        self.words: List[models.VocabularyWord] = []
        self._load_catalog()

    def _load_catalog(self):
        if not os.path.exists(self.catalog_path):
            logger.error(f"Catalog file not found: {self.catalog_path}")
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

    def get_client(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            logger.error("GEMINI_API_KEY not found!")
        return genai.Client(api_key=key, http_options={"api_version": "v1"})

    def fetch_vocabulary(
        self,
        target_rank: int,
        semantic_query: str,
        pos_distribution: Dict[str, float],
        state_distribution: Dict[models.WordState, float],
        word_count: int,
        diversity_penalty: float = 0.5,
        rank_radius: int = 500
    ) -> List[str]:
        """
        On-the-fly vocabulary selection using Gemini Embeddings and MMR.
        """
        # Step A: Filter subset (~1000 words)
        candidates = []
        for word in self.words:
            if word.state == models.WordState.LEARNING:
                candidates.append(word)
                continue

            if word.state == models.WordState.UNSEEN:
                if abs(word.rank - target_rank) <= rank_radius:
                    candidates.append(word)
                    continue

        if not candidates:
            # Fallback to nearest words if filter is too strict
            sorted_by_rank = sorted(self.words, key=lambda x: abs(x.rank - target_rank))
            candidates = sorted_by_rank[:1000]
        else:
            # Cap at 1000 for batch limit safety
            if len(candidates) > 1000:
                # Prioritize those closer to rank
                candidates.sort(key=lambda x: abs(x.rank - target_rank))
                candidates = candidates[:1000]

        # Step B: Batch Embed
        client = self.get_client()
        texts_to_embed = [semantic_query] + [c.w for c in candidates]

        try:
            res = client.models.embed_content(
                model="models/gemini-embedding-2",
                contents=texts_to_embed,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            embeddings = np.array([e.values for e in res.embeddings])
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            # Fallback: Just return random from candidates if API fails
            import random
            return [c.w for c in random.sample(candidates, min(len(candidates), word_count))]

        query_vec = embeddings[0:1]
        candidate_vecs = embeddings[1:]

        # Step C: Score & Select (MMR)
        similarities = cosine_similarity(query_vec, candidate_vecs)[0]

        # Split candidates into buckets by POS and State
        buckets = {}
        for pos, p_pct in pos_distribution.items():
            for state, s_pct in state_distribution.items():
                target = int(word_count * p_pct * s_pct)
                if target > 0:
                    buckets[(pos, state)] = target

        # MMR Logic adapted for buckets
        selected_indices = []
        final_words = []

        remaining_indices = set(range(len(candidates)))

        # Fill buckets
        for (target_pos, target_state), target_num in buckets.items():
            bucket_selected = 0
            while bucket_selected < target_num and remaining_indices:
                best_score = -np.inf
                best_idx = -1

                # Check remaining for bucket match
                for idx in remaining_indices:
                    word = candidates[idx]
                    if word.pos == target_pos and word.state == target_state:
                        rel_score = similarities[idx]

                        if not selected_indices:
                            score = rel_score
                        else:
                            div_score = np.max(cosine_similarity(
                                candidate_vecs[idx:idx+1],
                                candidate_vecs[selected_indices]
                            ))
                            score = (1 - diversity_penalty) * rel_score - diversity_penalty * div_score

                        if score > best_score:
                            best_score = score
                            best_idx = idx

                if best_idx != -1:
                    final_words.append(candidates[best_idx].w)
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                    bucket_selected += 1
                else:
                    break

        # Final fill if rounding/distributions left gaps
        while len(final_words) < word_count and remaining_indices:
            available = list(remaining_indices)
            best_idx = available[np.argmax(similarities[available])]
            final_words.append(candidates[best_idx].w)
            remaining_indices.remove(best_idx)

        return final_words[:word_count]
