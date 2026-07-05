"""
Cross-Entropy Attack (CEA) search method.

Native TextAttack ``SearchMethod`` implementation of the Cross-Entropy
Optimization procedure (Algorithm 1) from:

    Ni, M., Gong, Y., Liu, W. "Cross-Entropy Attacks to Language Models via
    Rare Event Simulation." Reference implementation:
    https://github.com/MingzeLucasNi/CE-Attack

Implementing it as a SearchMethod (rather than porting the standalone
reference script) lets it plug into the same Transformation/Constraint/
GoalFunction pipeline as PWWS/TextFooler/CLARE/PSO in this repo, so results
are directly comparable.

Algorithm, per position i in the editable set I:
  - S_i: substitution candidates for position i, taken from whatever
    Transformation + constraints are attached to the Attack (here:
    WordNet synonyms UNION RoBERTa masked-LM predictions -- see
    custom_recipes.build_cea).
  - theta_i: a categorical distribution over S_i, initialized uniformly.

At each iteration:
  1. Sample `num_candidates` full-text candidates: for every i in I,
     independently draw a substitution from theta_i (every editable
     position is always substituted, per the paper -- there is no
     "leave unchanged" option once a position is in I).
  2. Score each candidate x~ by f(x~; x) = m(F(x~)) * Sim(x~, x), where
     m(F(x~)) is the goal function's targeted-class score (already
     evaluates the objective in Eq. 5/6 of the paper) and Sim is cosine
     similarity between sentence-transformer embeddings.
  3. Keep the top `rho` fraction of candidates by score ("elite").
  4. Re-estimate each theta_i by maximum likelihood over the elite set
     (Eq. 19): the probability of each candidate word is its frequency
     among the elite samples at that position.
Repeat for `max_iters` iterations (or stop early on a successful attack /
query-budget exhaustion), then return the best-scoring candidate seen.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from textattack.goal_function_results.goal_function_result import (
    GoalFunctionResultStatus,
)
from textattack.search_methods import SearchMethod
from textattack.shared.validators import transformation_consists_of_word_swaps


class CrossEntropySearch(SearchMethod):
    def __init__(
        self,
        num_candidates=100,
        rho=0.5,
        max_iters=50,
        sim_model_name="all-MiniLM-L6-v2",
    ):
        self.num_candidates = num_candidates
        self.rho = rho
        self.max_iters = max_iters
        self._sim_model_name = sim_model_name
        self._sim_model = None
        self._search_over = False

    @property
    def sim_model(self):
        if self._sim_model is None:
            self._sim_model = SentenceTransformer(self._sim_model_name)
        return self._sim_model

    def _similarities(self, original_text, candidate_texts):
        """Cosine similarity between `original_text` and each of
        `candidate_texts`, used as Sim(x~, x) in the attack objective."""
        if not candidate_texts:
            return np.array([])
        embeddings = self.sim_model.encode(
            [original_text] + candidate_texts, convert_to_numpy=True
        )
        orig_emb, cand_embs = embeddings[0], embeddings[1:]
        orig_unit = orig_emb / (np.linalg.norm(orig_emb) + 1e-8)
        cand_unit = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-8)
        return cand_unit @ orig_unit

    def perform_search(self, initial_result):
        self._search_over = False
        original_text = initial_result.attacked_text

        # Build per-position substitution sets S_i (Eq. 8) by collecting every
        # single-position candidate the attached transformation + constraints
        # allow, grouped by which position they change.
        all_candidates = self.get_transformations(original_text, original_text=original_text)
        position_words = {}
        for cand in all_candidates:
            idx = next(iter(cand.attack_attrs["newly_modified_indices"]))
            words_here = position_words.setdefault(idx, [])
            word = cand.words[idx]
            if word not in words_here:
                words_here.append(word)

        editable_positions = list(position_words.keys())
        if not editable_positions:
            return initial_result

        # theta[i]: categorical distribution over position_words[i] (Eq. 12),
        # initialized uniformly (Eq. 20).
        theta = {
            i: np.ones(len(position_words[i])) / len(position_words[i])
            for i in editable_positions
        }

        best_result = initial_result
        best_score = initial_result.score

        for _ in range(self.max_iters):
            choices = []
            candidate_texts = []
            for _ in range(self.num_candidates):
                choice = {}
                indices, words = [], []
                for i in editable_positions:
                    j = int(np.random.choice(len(theta[i]), p=theta[i]))
                    choice[i] = j
                    indices.append(i)
                    words.append(position_words[i][j])
                choices.append(choice)
                candidate_texts.append(original_text.replace_words_at_indices(indices, words))

            results, self._search_over = self.get_goal_results(candidate_texts)
            if not results:
                break
            # `results` can be shorter than `candidate_texts` if the query
            # budget ran out mid-batch; keep `choices` aligned with it.
            choices = choices[: len(results)]

            m_scores = np.array([r.score for r in results])
            sims = self._similarities(
                original_text.text, [r.attacked_text.text for r in results]
            )
            scores = m_scores * sims

            top_i = int(np.argmax(scores))
            if scores[top_i] > best_score:
                best_score = scores[top_i]
                best_result = results[top_i]
            if (
                results[top_i].goal_status == GoalFunctionResultStatus.SUCCEEDED
                or self._search_over
            ):
                return results[top_i]

            # Elite selection: keep the top `rho` fraction by objective score
            # (Eq. 18: gamma is the (1-rho)-quantile threshold).
            n_elite = max(1, int(np.ceil(self.rho * len(scores))))
            elite_idx = np.argsort(-scores)[:n_elite]
            elite_choices = [choices[i] for i in elite_idx]

            # Maximum-likelihood update of each position's distribution
            # (Eq. 19): probability = frequency among elite samples.
            for i in editable_positions:
                counts = np.zeros(len(position_words[i]))
                for c in elite_choices:
                    counts[c[i]] += 1
                if counts.sum() > 0:
                    theta[i] = counts / counts.sum()

        return best_result

    def check_transformation_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["num_candidates", "rho", "max_iters"]
