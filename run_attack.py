"""
Run a TextAttack recipe against Prompt Guard 86M to measure how often a
known prompt-injection/jailbreak text can be perturbed into evading
detection (i.e. flipped to predicted label BENIGN).

Usage:
    python run_attack.py pwws
    python run_attack.py textfooler
    python run_attack.py clare
    python run_attack.py pso_wordnet
    python run_attack.py cea_wordnet_mlm
    python run_attack.py cea_textfooler

Requires detected_injections.json (produced by baseline.py) to already exist.
"""
import os

# Must be set before `transformers`/`textattack` are imported (see baseline.py).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE_DIR = os.path.join(PROJECT_DIR, ".cache")
os.environ.setdefault("TA_CACHE_DIR", os.path.join(LOCAL_CACHE_DIR, "textattack"))
os.environ.setdefault("HF_HOME", os.path.join(LOCAL_CACHE_DIR, "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", os.path.join(LOCAL_CACHE_DIR, "huggingface", "datasets")
)
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    os.path.join(LOCAL_CACHE_DIR, "sentence-transformers"),
)
os.environ.setdefault("NLTK_DATA", os.path.join(LOCAL_CACHE_DIR, "nltk"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(LOCAL_CACHE_DIR, "matplotlib"))

for cache_dir in (
    os.environ["TA_CACHE_DIR"],
    os.environ["HF_HOME"],
    os.environ["HF_DATASETS_CACHE"],
    os.environ["SENTENCE_TRANSFORMERS_HOME"],
    os.environ["NLTK_DATA"],
    os.environ["MPLCONFIGDIR"],
):
    os.makedirs(cache_dir, exist_ok=True)

import argparse
import json
import random

import nltk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# TextFooler/CLARE's PartOfSpeech constraint needs this specific tagger data;
# fetch it once up front instead of crashing mid-attack on a cache miss.
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download(
        "averaged_perceptron_tagger_eng",
        download_dir=os.environ["NLTK_DATA"],
        quiet=True,
    )

import textattack
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import PWWSRen2019
from textattack.datasets import Dataset
from textattack.goal_functions.classification import TargetedClassification
from textattack.models.wrappers import HuggingFaceModelWrapper

from custom_recipes import (
    build_cea_textfooler_reg,
    build_cea_textfooler,
    build_cea_wordnet_mlm_reg,
    build_cea_wordnet_mlm,
    build_clare,
    build_pso_wordnet,
    build_textfooler,
)
from detailed_report import write_detailed_outputs

MODEL = "Niansuh/Prompt-Guard-86M"
DETECTED_FILE = "detected_injections.json"

# textfooler/clare/pso_wordnet/cea_* all use custom builders (see custom_recipes.py):
# textfooler/clare avoid textattack's tensorflow_hub-based USE constraint
# (plain `import tensorflow` crashes outright on this machine); pso_wordnet
# uses WordNet instead of HowNet for substitutions (HowNet gave a near-empty
# candidate pool -- 0/50 evasions, see results/pso_hownet/). The CEA variants
# share the same Cross-Entropy SearchMethod but use different candidate sources.
RECIPE_BUILDERS = {
    "pwws": PWWSRen2019.build,
    "textfooler": build_textfooler,
    "clare": build_clare,
    "pso_wordnet": build_pso_wordnet,
    "cea_wordnet_mlm": build_cea_wordnet_mlm,
    "cea_textfooler": build_cea_textfooler,
    "cea_wordnet_mlm_reg": build_cea_wordnet_mlm_reg,
    "cea_textfooler_reg": build_cea_textfooler_reg,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", choices=RECIPE_BUILDERS.keys())
    parser.add_argument("--n", type=int, default=50, help="number of seed examples to attack")
    parser.add_argument("--query-budget", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-name",
        help="results/<output-name>/ directory name (defaults to the recipe name)",
    )
    args = parser.parse_args()
    output_name = args.output_name or args.recipe

    # Real CUDA GPUs (e.g. a server) are used when available. MPS (Apple GPU)
    # is deliberately skipped even when present: it recompiles its Metal graph
    # for every new sequence length, making per-example attack queries
    # (variable-length text) extremely slow -- CPU was faster on a Mac.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    wrapper = HuggingFaceModelWrapper(model, tok)

    with open(DETECTED_FILE) as f:
        texts = json.load(f)

    random.seed(args.seed)
    sample = random.sample(texts, min(args.n, len(texts)))

    # Ground-truth label = current model prediction (INJECTION or JAILBREAK).
    # Only used to confirm the seed text is indeed caught pre-attack; the
    # attack itself targets label 0 (BENIGN) regardless of which non-benign
    # class this is.
    preds = wrapper(sample)
    labels = preds.argmax(dim=-1).tolist()
    id2label = model.config.id2label
    label_names = [id2label[i] for i in range(len(id2label))]
    dataset = Dataset(list(zip(sample, labels)), label_names=label_names)

    built = RECIPE_BUILDERS[args.recipe](wrapper)
    # Each recipe's build() wires up UntargetedClassification by default, whose
    # "success" is just *any* label change -- including INJECTION<->JAILBREAK,
    # which is NOT an evasion of Prompt Guard. What we actually want is a
    # targeted attack toward label 0 (BENIGN), i.e. genuinely slipping past
    # the guard. Attack.__init__ binds search_method.get_goal_results to the
    # goal function at construction time, so swapping `attack.goal_function`
    # after the fact doesn't propagate -- we must rebuild the Attack object.
    attack = textattack.Attack(
        TargetedClassification(wrapper, target_class=0),
        built.constraints + built.pre_transformation_constraints,
        built.transformation,
        built.search_method,
    )

    out_dir = os.path.join("results", output_name)
    os.makedirs(out_dir, exist_ok=True)

    attack_args = AttackArgs(
        num_examples=len(sample),
        query_budget=args.query_budget,
        random_seed=args.seed,
        shuffle=False,
        disable_stdout=False,
        log_to_csv=os.path.join(out_dir, "results.csv"),
        # "file" wraps every changed word in [[ ]] in both the CSV and the
        # detailed report below, so exactly what got perturbed is visible
        # without needing a diff tool.
        csv_coloring_style="file",
    )

    attacker = Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()

    write_detailed_outputs(output_name, results, id2label, out_dir)

    n_total = len(results)
    n_skipped = sum(1 for r in results if isinstance(r, textattack.attack_results.SkippedAttackResult))
    n_attempted = n_total - n_skipped
    n_success = sum(1 for r in results if isinstance(r, textattack.attack_results.SuccessfulAttackResult))
    n_failed = sum(1 for r in results if isinstance(r, textattack.attack_results.FailedAttackResult))

    summary = {
        "recipe": args.recipe,
        "output_name": output_name,
        "query_budget": args.query_budget,
        "seed": args.seed,
        "n_total_sampled": n_total,
        "n_skipped_already_evading_or_error": n_skipped,
        "n_attempted": n_attempted,
        "n_success_evaded": n_success,
        "n_still_detected_after_attack": n_failed,
        "baseline_detection_rate_pct": 100.0 * n_attempted / n_total if n_total else None,
        "attack_success_rate_pct": 100.0 * n_success / n_attempted if n_attempted else None,
        "post_attack_detection_rate_pct": 100.0 * n_failed / n_attempted if n_attempted else None,
    }
    print("\nSUMMARY_JSON:" + json.dumps(summary))

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs to {out_dir}/:")
    print("  summary.json  - aggregate numbers (attack success rate, etc.)")
    print("  details.json  - structured per-example data (labels/confidence/changed words/full text)")
    print("  report.md     - human-readable per-example report, changes marked with [[ ]]")
    print("  results.csv   - raw TextAttack log (same [[ ]] markup)")


if __name__ == "__main__":
    main()
