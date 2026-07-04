"""
Turn a list of textattack AttackResult objects into two persistent,
easy-to-inspect artifacts written to results/<recipe>/:

- details.json: one structured record per example (labels, confidences,
  full before/after text, and the exact words that changed).
- report.md: the same data as a human-readable Markdown file, with an
  overview table up top and one section per example below it, changed
  words marked inline with [[ ]].
"""
import json
import os
import re

BRACKET_RE = re.compile(r"\[\[(.*?)\]\]")


def _label_conf(goal_function_result, id2label):
    label_idx = int(goal_function_result.output)
    label_name = id2label.get(label_idx, str(label_idx))
    raw = goal_function_result.raw_output
    confidence = float(raw[label_idx])
    p_benign = float(raw[0])
    return label_name, confidence, p_benign


def build_records(results, id2label):
    records = []
    for i, result in enumerate(results):
        result_type = type(result).__name__.replace("AttackResult", "") or "Attack"

        orig_marked, pert_marked = result.diff_color(color_method="file")
        removed_words = BRACKET_RE.findall(orig_marked)
        added_words = BRACKET_RE.findall(pert_marked)

        orig_label, orig_conf, orig_p_benign = _label_conf(result.original_result, id2label)
        final_label, final_conf, final_p_benign = _label_conf(result.perturbed_result, id2label)

        records.append(
            {
                "index": i,
                "result_type": result_type,
                "num_queries": result.num_queries,
                "original_label": orig_label,
                "original_confidence": orig_conf,
                "original_p_benign": orig_p_benign,
                "final_label": final_label,
                "final_confidence": final_conf,
                "final_p_benign": final_p_benign,
                "evaded_to_benign": final_label == "BENIGN",
                "removed_words": removed_words,
                "added_words": added_words,
                "original_text_plain": result.original_result.attacked_text.text,
                "perturbed_text_plain": result.perturbed_result.attacked_text.text,
                "original_text_marked": orig_marked,
                "perturbed_text_marked": pert_marked,
            }
        )
    return records


def write_json(records, path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def write_markdown(records, recipe, path):
    lines = [f"# {recipe} — per-example attack results\n"]

    lines.append("## Overview\n")
    lines.append("| # | Result | Before | P(BENIGN) before | After | P(BENIGN) after | Queries |")
    lines.append("|---|--------|--------|-------------------|-------|-------------------|---------|")
    for r in records:
        lines.append(
            f"| {r['index']} | {r['result_type']} | {r['original_label']} "
            f"({r['original_confidence']:.1%}) | {r['original_p_benign']:.1%} | "
            f"{r['final_label']} ({r['final_confidence']:.1%}) | {r['final_p_benign']:.1%} | "
            f"{r['num_queries']} |"
        )

    lines.append("\n## Per-example detail\n")
    for r in records:
        lines.append(f"### Example {r['index']} — {r['result_type']}\n")
        lines.append(
            f"- Before attack: **{r['original_label']}** "
            f"(confidence {r['original_confidence']:.1%}, P(BENIGN)={r['original_p_benign']:.1%})"
        )
        lines.append(
            f"- After attack: **{r['final_label']}** "
            f"(confidence {r['final_confidence']:.1%}, P(BENIGN)={r['final_p_benign']:.1%})"
        )
        lines.append(f"- Evaded to BENIGN: {'yes' if r['evaded_to_benign'] else 'no'}")
        lines.append(f"- Queries used: {r['num_queries']}")
        if r["removed_words"] or r["added_words"]:
            lines.append(f"- Original words changed: {r['removed_words']}")
            lines.append(f"- Replaced/inserted with: {r['added_words']}")
        else:
            lines.append("- No perturbation found (attack made no changes)")
        lines.append("\n**Original text (`[[ ]]` marks changed words):**\n")
        lines.append(f"> {r['original_text_marked']}\n")
        lines.append("**Perturbed text:**\n")
        lines.append(f"> {r['perturbed_text_marked']}\n")
        lines.append("---\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_detailed_outputs(recipe, results, id2label, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    records = build_records(results, id2label)
    write_json(records, os.path.join(out_dir, "details.json"))
    write_markdown(records, recipe, os.path.join(out_dir, "report.md"))
