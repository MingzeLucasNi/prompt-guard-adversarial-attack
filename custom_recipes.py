"""
Custom attack recipes for this project:

- TF-free versions of TextFooler (Jin et al. 2019) and CLARE (Li et al.
  2020): identical to textattack's built-in `TextFoolerJin2019` /
  `CLARE2020` except the tensorflow_hub-based UniversalSentenceEncoder
  constraint is swapped for `SBERTSentenceEncoder` (see sbert_encoder.py)
  because tensorflow itself crashes on import on this machine.
- A WordNet-based PSO (Zang et al. 2020): identical to textattack's
  built-in `PSOZang2020` except the substitution source is WordNet
  synonyms instead of HowNet sememes, which gave PSO a near-empty
  candidate pool for this kind of English text (0/50 evasions -- see
  results/pso/).
- CEA (Ni, Gong, Liu -- Cross-Entropy Attack), ported from the author's
  own reference implementation (github.com/MingzeLucasNi/CE-Attack) into
  a native TextAttack SearchMethod (see cea_search.py) so it runs through
  the same pipeline as the other three attacks. Candidates come from
  WordNet synonyms UNION RoBERTa masked-LM predictions -- unified with
  PSO's substitution source, and a larger pool than the paper's WordNet ∩
  MLM intersection, per project preference for more candidates.
"""
import transformers
from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import (
    GreedySearch,
    GreedyWordSwapWIR,
    ParticleSwarmOptimization,
)
from textattack.transformations import (
    CompositeTransformation,
    WordInsertionMaskedLM,
    WordMergeMaskedLM,
    WordSwapEmbedding,
    WordSwapMaskedLM,
    WordSwapWordNet,
)

from cea_search import CrossEntropySearch
from sbert_encoder import SBERTSentenceEncoder


def build_textfooler(model_wrapper):
    transformation = WordSwapEmbedding(max_candidates=50)
    # fmt: off
    stopwords = set(
        ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
    )
    # fmt: on
    constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
    constraints.append(InputColumnModification(["premise", "hypothesis"], {"premise"}))
    constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
    constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
    constraints.append(
        SBERTSentenceEncoder(
            threshold=0.75,
            metric="cosine",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
    )
    goal_function = UntargetedClassification(model_wrapper)
    search_method = GreedyWordSwapWIR(wir_method="delete")
    return Attack(goal_function, constraints, transformation, search_method)


def build_clare(model_wrapper):
    shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained("distilroberta-base")
    shared_tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
    transformation = CompositeTransformation(
        [
            WordSwapMaskedLM(
                method="bae",
                masked_language_model=shared_masked_lm,
                tokenizer=shared_tokenizer,
                max_candidates=50,
                min_confidence=5e-4,
            ),
            WordInsertionMaskedLM(
                masked_language_model=shared_masked_lm,
                tokenizer=shared_tokenizer,
                max_candidates=50,
                min_confidence=0.0,
            ),
            WordMergeMaskedLM(
                masked_language_model=shared_masked_lm,
                tokenizer=shared_tokenizer,
                max_candidates=50,
                min_confidence=5e-3,
            ),
        ]
    )
    constraints = [RepeatModification(), StopwordModification()]
    constraints.append(
        SBERTSentenceEncoder(
            threshold=0.7,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
    )
    goal_function = UntargetedClassification(model_wrapper)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, transformation, search_method)


def build_pso(model_wrapper):
    transformation = WordSwapWordNet()
    constraints = [RepeatModification(), StopwordModification()]
    constraints.append(InputColumnModification(["premise", "hypothesis"], {"premise"}))
    goal_function = UntargetedClassification(model_wrapper)
    search_method = ParticleSwarmOptimization(pop_size=60, max_iters=20)
    return Attack(goal_function, constraints, transformation, search_method)


def build_cea(model_wrapper):
    shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained("distilroberta-base")
    shared_tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
    transformation = CompositeTransformation(
        [
            WordSwapWordNet(),
            WordSwapMaskedLM(
                method="bae",
                masked_language_model=shared_masked_lm,
                tokenizer=shared_tokenizer,
                max_candidates=50,
                min_confidence=5e-4,
            ),
        ]
    )
    constraints = [RepeatModification(), StopwordModification()]
    goal_function = UntargetedClassification(model_wrapper)
    # N=100 candidates/iteration, rho=0.5 elite fraction, T=50 max iterations:
    # the paper's own defaults (Section 4.1).
    search_method = CrossEntropySearch(num_candidates=100, rho=0.5, max_iters=50)
    return Attack(goal_function, constraints, transformation, search_method)
