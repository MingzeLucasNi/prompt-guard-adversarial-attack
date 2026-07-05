# Adversarial Robustness Evaluation of Meta Prompt Guard 86M
# Meta Prompt Guard 86M 对抗鲁棒性评测

This project evaluates how robust Meta's open-weight Prompt Guard 86M classifier is against standard word-level adversarial NLP attacks.

本项目评测 Meta 开源的 Prompt Guard 86M 分类器在词级对抗攻击下的鲁棒性。

---

## Overview / 项目简介

**EN** — Prompt Guard 86M is a 3-class classifier for `BENIGN`, `INJECTION`, and `JAILBREAK` prompts. We test whether prompts that are originally detected as malicious can be rewritten into semantically similar text that the model misclassifies as benign. All attacks are run through TextAttack with a targeted objective toward `BENIGN`.

**中文** — Prompt Guard 86M 是一个三分类模型，标签为 `BENIGN`、`INJECTION`、`JAILBREAK`。本项目测试：模型原本能正确识别的恶意 prompt，是否可以通过保持语义相近的词级改写，被模型误判为无害。所有攻击都通过 TextAttack 运行，目标统一设为定向逃逸到 `BENIGN`。

---

## Ethics / 伦理声明

This is academic adversarial-robustness research. It does not target real users, private data, or deployed systems. The goal is to measure and improve safety-classifier robustness, not to enable real-world abuse.

这是学术性质的对抗鲁棒性研究，不针对真实用户、私人数据或线上系统。目标是衡量并帮助提升安全分类器鲁棒性，而不是用于真实世界绕过或滥用。

---

## Experimental Setup / 实验设置

- **Model**: `Niansuh/Prompt-Guard-86M`, an ungated public mirror of Meta Prompt Guard 86M.
- **Dataset**: `deepset/prompt-injections`.
- **Seed pool**: 263 ground-truth injection examples.
- **Baseline**: Prompt Guard correctly detects 247 / 263 injection examples before attack, i.e. **93.9%**.
- **Attack seeds**: all attacks sample 50 prompts from the 247 correctly detected examples with seed `42`.
- **Success criterion**: an attack succeeds only if the final prediction becomes `BENIGN`.
- **Query budgets**: PWWS, TextFooler, CLARE, and PSO use 1000 queries per example. CEA is reported both at 5000 queries and, for fair comparison, at 1000 queries.

中文简述：

- **模型**：`Niansuh/Prompt-Guard-86M`，Meta Prompt Guard 86M 的公开镜像。
- **数据集**：`deepset/prompt-injections`。
- **攻击起点**：263 条真实 injection 样本。
- **基线**：攻击前模型正确检出 247 / 263 条，检出率 **93.9%**。
- **攻击样本**：每种攻击从这 247 条已正确检出的样本中，用 seed `42` 抽取 50 条。
- **成功标准**：只有最终被预测为 `BENIGN` 才算真正逃逸成功。
- **查询预算**：PWWS、TextFooler、CLARE、PSO 每条样本 1000 queries；CEA 同时报告 5000-query 版本和更公平的 1000-query 版本。

---

## Attack Variants / 攻击版本

| Recipe | Description |
|---|---|
| `pwws` | PWWS with WordNet substitutions. |
| `textfooler` | TextFooler with embedding substitutions, POS filtering, and sentence-similarity filtering. |
| `clare` | CLARE with contextual masked-LM edits. |
| `pso_hownet` | Historical PSO run using HowNet-style candidates; kept as a superseded baseline. |
| `pso_wordnet` | Current PSO version using WordNet candidates and TextFooler-style constraints. |
| `cea_wordnet_mlm` | CEA search with WordNet ∪ masked-LM candidates. |
| `cea_textfooler` | CEA search with TextFooler-style embedding candidates. |
| `cea_wordnet_mlm_reg` | Experimental CEA variant with a modification-count regularizer. |
| `cea_textfooler_reg` | Experimental CEA-TextFooler variant with a modification-count regularizer. |

The current PSO and CEA variants use TextFooler-style constraints: stopword/repeat protection, word-embedding similarity, POS compatibility, and sentence-level semantic similarity.

当前 PSO 和 CEA 版本都使用 TextFooler 风格约束：stopword/repeat 保护、词向量相似度、词性兼容、句级语义相似度。

---

## Results / 结果

### Main Results

| Attack | Query budget | Samples | Evaded to BENIGN | Still detected | Success rate |
|---|---:|---:|---:|---:|---:|
| Baseline, no attack | — | 263 | — | 247 | — |
| `pwws` | 1000 | 50 | 17 | 33 | **34.0%** |
| `textfooler` | 1000 | 50 | 31 | 19 | **62.0%** |
| `clare` | 1000 | 50 | 8 | 42 | **16.0%** |
| `pso_hownet` | 1000 | 50 | 0 | 50 | **0.0%** |
| `pso_wordnet` | 1000 | 50 | 6 | 44 | **12.0%** |
| `cea_wordnet_mlm` | 5000 | 50 | 32 | 18 | **64.0%** |
| `cea_textfooler` | 5000 | 50 | 39 | 11 | **78.0%** |

### CEA Budget And Regularization Checks

| Attack | Query budget | Samples | Evaded to BENIGN | Still detected | Success rate |
|---|---:|---:|---:|---:|---:|
| `cea_wordnet_mlm_q1000` | 1000 | 50 | 32 | 18 | **64.0%** |
| `cea_textfooler_q1000` | 1000 | 50 | 36 | 14 | **72.0%** |
| `cea_wordnet_mlm_reg` | 1000 | 50 | 32 | 18 | **64.0%** |
| `cea_textfooler_reg` | 1000 | 50 | 34 | 16 | **68.0%** |

Full per-example outputs are stored under `results/<recipe>/` as:

- `summary.json`
- `details.json`
- `report.md`
- `results.csv`

完整逐条结果都保存在 `results/<recipe>/` 下，包括 `summary.json`、`details.json`、`report.md`、`results.csv`。

---

## Key Findings / 关键发现

**EN**

- Prompt Guard 86M performs well before attack, detecting **93.9%** of injection examples.
- TextFooler achieves **62.0%** attack success, showing that small lexical changes can significantly weaken detection.
- PSO is strongly affected by candidate quality: HowNet gives **0.0%**, while WordNet improves to **12.0%**.
- CEA is the strongest family in these experiments. `cea_textfooler` reaches **78.0%** with 5000 queries and **72.0%** with 1000 queries.
- The modification-regularized CEA variants reduce aggressive rewriting pressure, at some cost to attack success.

**中文**

- Prompt Guard 86M 在攻击前表现较好，能检出 **93.9%** 的 injection 样本。
- TextFooler 成功率达到 **62.0%**，说明小幅词级改写已经能明显削弱检测效果。
- PSO 对候选词质量很敏感：HowNet 版本为 **0.0%**，WordNet 版本提升到 **12.0%**。
- CEA 系列是本实验中最强的一组。`cea_textfooler` 在 5000 queries 下达到 **78.0%**，在 1000 queries 下仍有 **72.0%**。
- 加入修改数量正则后，CEA 的大面积改写倾向会降低，但攻击成功率也会有所下降。

---

## Reproduction / 复现方法

```bash
git clone https://github.com/MingzeLucasNi/prompt-guard-adversarial-attack.git
cd prompt-guard-adversarial-attack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python baseline.py
python run_attack.py pwws
python run_attack.py textfooler
python run_attack.py clare
python run_attack.py pso_wordnet

# CEA main runs
python run_attack.py cea_wordnet_mlm --query-budget 5000
python run_attack.py cea_textfooler --query-budget 5000

# Fair 1000-query CEA checks
python run_attack.py cea_wordnet_mlm --query-budget 1000 --output-name cea_wordnet_mlm_q1000
python run_attack.py cea_textfooler --query-budget 1000 --output-name cea_textfooler_q1000

# Modification-regularized CEA checks
python run_attack.py cea_wordnet_mlm_reg --query-budget 1000
python run_attack.py cea_textfooler_reg --query-budget 1000

python compare_results.py
```

CUDA is used automatically when available. CLARE and CEA are much faster on GPU.

有 CUDA 会自动使用。CLARE 和 CEA 在 GPU 上会快很多。

---

## Project Structure / 项目结构

```text
baseline.py              # baseline detection rate and seed-pool generation
run_attack.py             # attack runner
custom_recipes.py         # TextAttack recipe builders
cea_search.py             # CEA search implementation
sbert_encoder.py          # sentence-transformers semantic-similarity constraint
detailed_report.py        # details.json and report.md generation
compare_results.py        # result summary table
detected_injections.json  # cached correctly detected attack seeds
results/<recipe>/         # per-attack outputs
```

---

## Limitations / 局限性

- Each attack uses 50 sampled prompts; larger samples would give tighter estimates.
- The CEA implementation here is a TextAttack-native reimplementation and should not be read as an exact reproduction of the original paper's code.
- Some candidate sources differ from the original papers for practical reasons. For example, the current PSO result uses WordNet because the HowNet candidate pool was too sparse here.
- Results are for Prompt Guard 86M only; newer Prompt Guard versions are not evaluated.

中文：

- 每种攻击只抽样 50 条 prompt，更大的样本量会给出更稳定的估计。
- 这里的 CEA 是 TextAttack 原生重实现，不等同于论文原始代码的精确复现。
- 部分候选词来源因为实验可行性做了调整。例如当前 PSO 使用 WordNet，因为 HowNet 在这里候选词过少。
- 结果只针对 Prompt Guard 86M，不代表更新版本的 Prompt Guard。

---

## References / 参考文献

- Ren, S., Deng, Y., He, K., & Che, W. (2019). *Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency.* ACL. (PWWS)
- Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P. (2020). *Is BERT Really Robust? A Strong Baseline for Language Attack on Text Classification and Entailment.* AAAI. (TextFooler)
- Li, D., Zhang, Y., Peng, H., Chen, L., Brockett, C., Sun, M. T., & Dolan, B. (2021). *Contextualized Perturbation for Textual Adversarial Attack.* NAACL. (CLARE)
- Zang, Y., Qi, F., Yang, C., Liu, Z., Zhang, M., Liu, Q., & Sun, M. (2020). *Word-level Textual Adversarial Attacking as Combinatorial Optimization.* ACL. (PSO)
- Ni, M., Gong, Y., & Liu, W. *Cross-Entropy Attacks to Language Models via Rare Event Simulation.* (CEA)
- Morris, J., Lifland, E., Yoo, J. Y., Grigsby, J., Jin, D., & Qi, Y. (2020). *TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP.* EMNLP.
