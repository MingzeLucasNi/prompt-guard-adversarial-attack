"""
Run a TextAttack recipe against Prompt Guard 86M to measure how often a
known prompt-injection/jailbreak text can be perturbed into evading
detection (i.e. flipped to predicted label BENIGN).

Usage:
    python run_attack.py pwws
    python run_attack.py textfooler
    python run_attack.py clare
    python run_attack.py pso

Requires detected_injections.json (produced by baseline.py) to already exist.
"""
import os

# Must be set before `transformers`/`textattack` are imported (see baseline.py).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import argparse
import json
import random

import nltk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# TextFooler/CLARE's PartOfSpeech constraint needs this specific tagger data;
# fetch it once up front instead of crashing mid-attack on a cache miss.
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

import textattack
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import PSOZang2020, PWWSRen2019
from textattack.datasets import Dataset
from textattack.goal_functions.classification import TargetedClassification
from textattack.models.wrappers import HuggingFaceModelWrapper

from custom_recipes import build_clare, build_textfooler
from detailed_report import write_detailed_outputs

MODEL = "Niansuh/Prompt-Guard-86M"
DETECTED_FILE = "detected_injections.json"

# textfooler/clare use custom TF-free builders (see custom_recipes.py) instead
# of textattack's built-in recipes, which depend on tensorflow_hub -- and
# plain `import tensorflow` crashes outright on this machine.
RECIPE_BUILDERS = {
    "pwws": PWWSRen2019.build,
    "textfooler": build_textfooler,
    "clare": build_clare,
    "pso": PSOZang2020.build,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", choices=RECIPE_BUILDERS.keys())
    parser.add_argument("--n", type=int, default=50, help="number of seed examples to attack")
    parser.add_argument("--query-budget", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    attack_args = AttackArgs(
        num_examples=len(sample),
        query_budget=args.query_budget,
        random_seed=args.seed,
        shuffle=False,
        disable_stdout=False,
        log_to_csv=f"results_{args.recipe}.csv",
        # "file" wraps every changed word in [[ ]] in both the CSV and the
        # detailed report below, so exactly what got perturbed is visible
        # without needing a diff tool.
        csv_coloring_style="file",
    )

    attacker = Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()

    write_detailed_outputs(args.recipe, results, id2label)

    n_total = len(results)
    n_skipped = sum(1 for r in results if isinstance(r, textattack.attack_results.SkippedAttackResult))
    n_attempted = n_total - n_skipped
    n_success = sum(1 for r in results if isinstance(r, textattack.attack_results.SuccessfulAttackResult))
    n_failed = sum(1 for r in results if isinstance(r, textattack.attack_results.FailedAttackResult))

    summary = {
        "recipe": args.recipe,
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

    with open(f"summary_{args.recipe}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs for '{args.recipe}':")
    print(f"  summary_{args.recipe}.json   - 数字汇总（攻击成功率等）")
    print(f"  details_{args.recipe}.json   - 逐条结构化数据（标签/置信度/改动的词/完整文本）")
    print(f"  report_{args.recipe}.md      - 逐条人类可读报告，改动处用 [[ ]] 标出")
    print(f"  results_{args.recipe}.csv    - TextAttack 原始日志（同样带 [[ ]] 标记）")


if __name__ == "__main__":
    main()
