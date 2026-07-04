"""Print a before/after comparison table across all recipes that have been run."""
import json
import os

RECIPES = ["pwws", "textfooler", "clare", "pso"]

def main():
    print(f"{'recipe':<12}{'attacked':>10}{'evaded':>10}{'still caught':>14}{'attack success rate':>22}")
    for r in RECIPES:
        path = f"summary_{r}.json"
        if not os.path.exists(path):
            print(f"{r:<12}{'(not run yet)':>10}")
            continue
        with open(path) as f:
            s = json.load(f)
        print(
            f"{r:<12}{s['n_attempted']:>10}{s['n_success_evaded']:>10}"
            f"{s['n_still_detected_after_attack']:>14}{s['attack_success_rate_pct']:>21.1f}%"
        )


if __name__ == "__main__":
    main()
