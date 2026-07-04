"""
Turn a list of textattack AttackResult objects into two persistent,
easy-to-inspect artifacts:

- details_<recipe>.json: one structured record per example (labels,
  confidences, full before/after text, and the exact words that changed).
- report_<recipe>.md: the same data as a human-readable Markdown file, with
  an overview table up top and one section per example below it, changed
  words marked inline with [[ ]].
"""
import json
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
    lines = [f"# {recipe} — 逐条攻击详情\n"]

    lines.append("## 总览\n")
    lines.append("| # | 结果 | 攻击前 | 攻击前P(BENIGN) | 攻击后 | 攻击后P(BENIGN) | 查询次数 |")
    lines.append("|---|------|--------|------------------|--------|------------------|----------|")
    for r in records:
        lines.append(
            f"| {r['index']} | {r['result_type']} | {r['original_label']} "
            f"({r['original_confidence']:.1%}) | {r['original_p_benign']:.1%} | "
            f"{r['final_label']} ({r['final_confidence']:.1%}) | {r['final_p_benign']:.1%} | "
            f"{r['num_queries']} |"
        )

    lines.append("\n## 逐条详情\n")
    for r in records:
        lines.append(f"### Example {r['index']} — {r['result_type']}\n")
        lines.append(
            f"- 攻击前判定: **{r['original_label']}** "
            f"(置信度 {r['original_confidence']:.1%}, P(BENIGN)={r['original_p_benign']:.1%})"
        )
        lines.append(
            f"- 攻击后判定: **{r['final_label']}** "
            f"(置信度 {r['final_confidence']:.1%}, P(BENIGN)={r['final_p_benign']:.1%})"
        )
        lines.append(f"- 是否成功逃逸到 BENIGN: {'是' if r['evaded_to_benign'] else '否'}")
        lines.append(f"- 查询次数: {r['num_queries']}")
        if r["removed_words"] or r["added_words"]:
            lines.append(f"- 被删掉/替换掉的原词: {r['removed_words']}")
            lines.append(f"- 替换成/新插入的词: {r['added_words']}")
        else:
            lines.append("- 没有做出任何改动（未成功找到扰动）")
        lines.append("\n**原文（`[[ ]]` 标出改动处）:**\n")
        lines.append(f"> {r['original_text_marked']}\n")
        lines.append("**攻击后文本:**\n")
        lines.append(f"> {r['perturbed_text_marked']}\n")
        lines.append("---\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_detailed_outputs(recipe, results, id2label):
    records = build_records(results, id2label)
    write_json(records, f"details_{recipe}.json")
    write_markdown(records, recipe, f"report_{recipe}.md")
