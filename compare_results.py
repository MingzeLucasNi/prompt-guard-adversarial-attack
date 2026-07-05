"""Print a before/after comparison table across all recipes that have been run."""
import json
import os

RECIPES = [
    "pwws",
    "textfooler",
    "clare",
    "pso_hownet",
    "pso_wordnet",
    "cea_wordnet_mlm",
    "cea_textfooler",
    "cea_wordnet_mlm_reg",
    "cea_textfooler_reg",
    "cea_wordnet_mlm_q1000",
    "cea_textfooler_q1000",
]

def main():
    name_width = max(12, max(len(r) for r in RECIPES) + 2)
    print(f"{'recipe':<{name_width}}{'attacked':>10}{'evaded':>10}{'still caught':>14}{'attack success rate':>22}")
    for r in RECIPES:
        path = os.path.join("results", r, "summary.json")
        if not os.path.exists(path):
            print(f"{r:<{name_width}}{'(not run yet)':>10}")
            continue
        with open(path) as f:
            s = json.load(f)
        print(
            f"{r:<{name_width}}{s['n_attempted']:>10}{s['n_success_evaded']:>10}"
            f"{s['n_still_detected_after_attack']:>14}{s['attack_success_rate_pct']:>21.1f}%"
        )


if __name__ == "__main__":
    main()
