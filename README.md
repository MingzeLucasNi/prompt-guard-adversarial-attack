# Adversarial Robustness Evaluation of Meta's Prompt Guard 86M
# 对 Meta Prompt Guard 86M 的对抗鲁棒性评测

Evaluating how well Meta's open-source prompt-injection/jailbreak classifier holds up under standard, published adversarial NLP attacks and CEA candidate-generation variants.

用多种对抗NLP攻击方法及 CEA 候选词生成变体，评测 Meta 开源的 prompt injection / jailbreak 分类器的鲁棒性。

---

## Overview / 项目简介

**EN** — This project measures the adversarial robustness of [Prompt Guard 86M](https://huggingface.co/meta-llama/Prompt-Guard-86M), Meta's open-weight classifier for detecting prompt injection and jailbreak attempts. Using the [TextAttack](https://github.com/QData/TextAttack) framework, word-substitution attack algorithms from the adversarial NLP literature — **PWWS**, **TextFooler**, **CLARE**, **PSO**, and **CEA** variants — are applied to prompts the model already correctly flags as malicious, to see how many can be rewritten into text the model misclassifies as benign.

**中文** — 本项目评测 Meta 开源的 prompt injection / jailbreak 分类器 [Prompt Guard 86M](https://huggingface.co/meta-llama/Prompt-Guard-86M) 的对抗鲁棒性。使用 [TextAttack](https://github.com/QData/TextAttack) 框架，对已经被模型正确判定为恶意的 prompt，应用 **PWWS**、**TextFooler**、**CLARE**、**PSO** 以及 **CEA**（Cross-Entropy Attack）变体，衡量有多少比例可以被改写成模型误判为无害的文本。

---

## Ethics & Responsible Use / 伦理声明

**EN**
- This is academic adversarial-robustness research, not a tool for real-world abuse. The goal is to measure and help improve the robustness of safety classifiers — the same motivation behind the published attack papers this project implements (Ren et al. 2019; Jin et al. 2020; Li et al. 2021; Zang et al. 2020).
- Every component is open and public: an unmodified mirror of Meta's open-weight model, a public benchmark dataset ([`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections)), and peer-reviewed, previously published attack algorithms. No novel exploit, private data, or production system is involved.
- No real users, private data, or deployed systems were targeted.
- This work is not intended to help anyone manipulate, deceive, or bypass safety systems in live applications. Were similar findings ever obtained against a real production deployment (not the case here), the responsible path is disclosure to the vendor, not exploitation.
- All findings are shared openly (open weights, open data, open code) to support the community working on making these classifiers more robust, consistent with standard practice in adversarial ML research.

**中文**
- 这是学术性质的对抗鲁棒性研究，不是用于真实世界恶意利用的工具。目的是衡量并帮助提升安全分类器的鲁棒性——这也是本项目所实现的攻击方法论文本身的初衷（Ren et al. 2019；Jin et al. 2020；Li et al. 2021；Zang et al. 2020）。
- 所用的每一个组件都是公开的：Meta 开源权重模型的原样镜像、公开基准数据集（[`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections)）、以及经过同行评审、已发表的攻击算法。不涉及任何新型漏洞利用，也不涉及任何私有数据或生产系统。
- 全程没有针对任何真实用户、私人数据或线上部署系统。
- 本项目无意帮助任何人在真实应用中操纵、欺骗或绕过安全系统。如果类似方法未来被用于针对真实生产环境（本项目并非如此），负责任的做法是向厂商披露，而非加以利用。
- 所有结果都以完全公开的方式分享（开源权重、开源数据、开源代码），目的是支持社区共同提升这类分类器的鲁棒性，这也是对抗机器学习研究领域的通常做法。

---

## Methodology / 方法

**EN**
- **Target model**: Prompt Guard 86M (3-way classifier: `BENIGN` / `INJECTION` / `JAILBREAK`), loaded from an unmodified public mirror of the official weights.
- **Seed data**: [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) (609 examples total); the 263 examples labeled as injection attempts are used as attack seeds.
- **Baseline**: of those 263, the model correctly flags 247 (93.9%) as non-benign *before any attack* — this is the starting detection rate the attacks try to break.
- **Attack objective**: a **targeted** evasion attack toward label `BENIGN`. Success means the perturbed prompt is misclassified as benign — not merely relabeled between `INJECTION` and `JAILBREAK` (which would not be a real evasion).
- **Attacks evaluated**, all via TextAttack:
  - **PWWS** (Ren et al., 2019) — WordNet synonym substitution guided by word saliency.
  - **TextFooler** (Jin et al., 2020) — counter-fitted word-embedding substitution with POS and semantic-similarity constraints.
  - **CLARE** (Li et al., 2021) — contextualized replace/insert/merge perturbations via a masked language model.
  - **PSO-WordNet** (Zang et al., 2020) — particle swarm optimization over synonym substitutions. Originally proposed with HowNet sememe-based synonyms; the current runnable recipe is explicitly named `pso_wordnet` because it uses **WordNet** instead.
  - **CEA-WordNet-MLM** (Ni, Gong, & Liu — [Cross-Entropy Attacks to Language Models via Rare Event Simulation](https://github.com/MingzeLucasNi/CE-Attack)) — our CEA `SearchMethod` with candidate sets from **WordNet synonyms ∪ RoBERTa masked-language-model predictions**. It is explicitly named `cea_wordnet_mlm` because it is not the original HowNet-style candidate source.
  - **CEA-TextFooler** — the same CEA optimizer, but using TextFooler's word-embedding candidate generator plus POS and semantic-similarity filters instead of CEA-WordNet-MLM's WordNet ∪ masked-LM candidate pool.
- **Substitution-source naming**: old HowNet-based PSO results are kept as `pso_hownet`; current runnable variants are named by their candidate source (`pso_wordnet`, `cea_wordnet_mlm`, `cea_textfooler`) so results are not mistaken for the original paper configurations.
- **Shared constraints for current PSO/CEA variants**: `pso_wordnet`, `cea_wordnet_mlm`, and `cea_textfooler` all use TextFooler's filtering constraints: stopword/repeat protection, word-embedding similarity, POS compatibility, and SBERT sentence-level semantic similarity.
- **Sample size and budgets**: 50 randomly sampled prompts per attack (from the 247 correctly-detected examples). PWWS, TextFooler, CLARE, and PSO use a 1000-query budget per example; CEA variants use 5000 queries so their 50-iteration schedule can run.
- For TextFooler/CLARE/PSO-WordNet/CEA variants, the semantic-similarity constraint is implemented with a PyTorch `sentence-transformers` model rather than the original papers' TensorFlow-based Universal Sentence Encoder, for environment portability. CEA variants also use the same `sentence-transformers` model for their search objective's similarity term.

**中文**
- **目标模型**：Prompt Guard 86M（三分类：`BENIGN` / `INJECTION` / `JAILBREAK`），使用官方权重的原样公开镜像。
- **种子数据**：[`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections)（共609条），取其中标注为注入攻击的263条作为攻击起点。
- **基线**：这263条中，模型在**攻击前**能正确识别247条（93.9%）为非无害——这就是攻击要突破的起始检出率。
- **攻击目标**：**定向**逃逸攻击，目标标签为 `BENIGN`。只有扰动后的文本被误判为无害才算成功——单纯在 `INJECTION` 和 `JAILBREAK` 之间切换不算真正的逃逸。
- **评测的攻击方法**（均通过 TextAttack 实现）：
  - **PWWS**（Ren et al., 2019）——基于词语显著性的 WordNet 同义词替换。
  - **TextFooler**（Jin et al., 2020）——反义词过滤词向量替换，附加词性和语义相似度约束。
  - **CLARE**（Li et al., 2021）——基于掩码语言模型的上下文感知替换/插入/合并扰动。
  - **PSO-WordNet**（Zang et al., 2020）——同义词替换 + 粒子群优化搜索。原论文用 HowNet 义原同义词；当前可运行 recipe 明确命名为 `pso_wordnet`，因为它使用 **WordNet**。
  - **CEA-WordNet-MLM**（Ni, Gong, Liu ——[Cross-Entropy Attacks to Language Models via Rare Event Simulation](https://github.com/MingzeLucasNi/CE-Attack)）——CEA 的 TextAttack `SearchMethod` 版本，候选词来自 **WordNet 同义词 ∪ RoBERTa 掩码语言模型预测**。明确命名为 `cea_wordnet_mlm`，因为它不是原论文的 HowNet 风格候选源。
  - **CEA-TextFooler**——使用同一个 CEA 优化器，但候选词集合改用 TextFooler 的词向量近邻生成方式，并加入词性和语义相似度过滤。
- **替换词来源命名**：旧的 HowNet PSO 结果保留为 `pso_hownet`；当前可运行版本按候选词来源命名为 `pso_wordnet`、`cea_wordnet_mlm`、`cea_textfooler`，避免被误读成原论文配置。
- **当前 PSO/CEA 版本共享约束**：`pso_wordnet`、`cea_wordnet_mlm`、`cea_textfooler` 都使用 TextFooler 的过滤约束：stopword/repeat 保护、词向量相似度、词性兼容，以及 SBERT 句级语义相似度。
- **样本规模与查询预算**：每种攻击从247条已正确检出的样本中随机抽取50条。PWWS、TextFooler、CLARE、PSO 每条样本查询预算为1000次；CEA 系列为5000次，以便跑满50轮迭代计划。
- TextFooler/CLARE/PSO-WordNet/CEA 系列的语义相似度约束改用基于 PyTorch 的 `sentence-transformers` 模型实现，而非原论文中基于 TensorFlow 的 Universal Sentence Encoder（纯粹是为了环境可移植性，攻击目标本身没有变化）。CEA 系列的目标函数里的相似度项也用的同一个 `sentence-transformers` 模型。

---

## Results / 结果

| Attack | Samples attacked | Evaded to BENIGN | Still detected | Attack success rate |
|---|---|---|---|---|
| *Baseline (no attack)* | 263 | — | 247 (93.9%) | — |
| **PWWS** | 50 | 17 | 33 | **34.0%** |
| **TextFooler** | 50 | 31 | 19 | **62.0%** |
| **CLARE** | 50 | 8 | 42 | **16.0%** |
| **PSO-HowNet** (`pso_hownet`, superseded) | 50 | 0 | 50 | 0.0% |
| **PSO-WordNet** (`pso_wordnet`) | 50 | 6 | 44 | **12.0%** |
| **CEA-WordNet-MLM** (`cea_wordnet_mlm`) | 50 | 32 | 18 | **64.0%** |
| **CEA-TextFooler** (`cea_textfooler`) | 50 | 39 | 11 | **78.0%** |

`pso_hownet` is kept only as a superseded historical result to show why the HowNet substitution source was abandoned. The current PSO result is `pso_wordnet`.

Full per-example results (before/after label, confidence, and exactly which words were changed) are in `results/<recipe>/` for each row above. Each folder contains `summary.json`, `details.json`, `report.md`, and `results.csv`.

`pso_hownet` 只作为旧的历史结果保留，用来说明为什么放弃 HowNet 候选词源。当前 PSO 结果是 `pso_wordnet`。

完整的逐条结果（攻击前后标签、置信度、以及具体改动了哪些词）见各个 `results/<recipe>/` 文件夹。每个文件夹都包含 `summary.json`、`details.json`、`report.md` 和 `results.csv`。

---

## Key Findings / 关键发现

**EN**
- Prompt Guard 86M has a high detection rate (93.9%) on the known-injection benchmark before any attack — it works well against the patterns it was trained on.
- A simple, black-box synonym-substitution attack (PWWS) already flips **34%** of correctly-caught prompts to benign.
- TextFooler reaches **62%**, while CLARE reaches **16%** under the same targeted-to-BENIGN objective.
- PSO is highly sensitive to its substitution source: the old HowNet version achieved **0%**, and the constrained WordNet version reaches **12%**.
- The strongest final result is **CEA-TextFooler** at **78%**, followed by **CEA-WordNet-MLM** at **64%**. This suggests that CEA's cross-entropy optimizer benefits strongly from a richer, better-filtered candidate set.
- Taken together, Prompt Guard 86M is accurate on the original benchmark but remains vulnerable to meaning-constrained lexical perturbations. The attack's candidate-generation strategy is as important as the search algorithm itself.

**中文**
- Prompt Guard 86M 在攻击前对已知注入样本的检出率很高（93.9%）——对训练时见过的模式识别得不错。
- 一个简单黑盒同义词替换攻击（PWWS）已经能让 **34%** 的样本从"被正确拦截"变成"被判定无害"。
- TextFooler 成功率为 **62%**；CLARE 在同一定向逃逸目标下为 **16%**。
- PSO 对候选词来源非常敏感：旧的 HowNet 版本为 **0%**，加入当前约束后的 WordNet 版本为 **12%**。
- 最强的最终结果是 **CEA-TextFooler**（**78%**），其次是 **CEA-WordNet-MLM**（**64%**）。这说明 CEA 的交叉熵优化器很依赖候选词集合的质量和过滤方式。
- 总的来看，Prompt Guard 86M 在原始 benchmark 上检出率很高，但仍然容易受到有语义约束的词级扰动影响；攻击的候选词生成策略和搜索算法同样重要。

---

## Reproduction / 复现方法

```bash
git clone https://github.com/MingzeLucasNi/prompt-guard-adversarial-attack.git
cd prompt-guard-adversarial-attack
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['averaged_perceptron_tagger_eng','omw-1.4','stopwords','wordnet','universal_tagset','punkt']]"

python baseline.py               # computes the 93.9% baseline, writes detected_injections.json
python run_attack.py pwws        # ~15-30 min on CPU; seconds/example on a CUDA GPU
python run_attack.py textfooler
python run_attack.py pso_wordnet # slow on CPU (full query budget per example)
python run_attack.py clare       # search + masked-LM based, slowest; a GPU is strongly recommended
python run_attack.py cea_wordnet_mlm --query-budget 5000  # CEA optimizer with WordNet + MLM candidate sets
python run_attack.py cea_textfooler --query-budget 5000   # CEA optimizer with TextFooler-style candidate sets
python compare_results.py        # prints a summary table across all attacks that have been run
```

CUDA is used automatically when available (falls back to CPU otherwise). Each `run_attack.py <recipe>` writes its output to `results/<recipe>/`.

有 CUDA 会自动使用，否则回退到 CPU。每次 `run_attack.py <recipe>` 的结果都会写入 `results/<recipe>/` 文件夹。

---

## Project Structure / 项目结构

```
baseline.py           # computes pre-attack detection rate, selects attack seeds
run_attack.py          # runs one attack recipe end-to-end
custom_recipes.py       # TF-free TextFooler/CLARE, PSO-WordNet, and CEA variant builders
cea_search.py           # CEA's Cross-Entropy Optimization, as a native TextAttack SearchMethod
sbert_encoder.py        # the sentence-transformers-based semantic similarity constraint
detailed_report.py      # turns raw attack results into details.json + report.md
compare_results.py      # prints a summary table across all attacks that have been run
detected_injections.json  # cached list of prompts Prompt Guard correctly flags (attack seed pool)
results/<recipe>/        # summary.json, details.json, report.md, results.csv per attack
```

---

## Limitations / 局限性

**EN**
- Sample size is 50 prompts per attack; a larger sample would give tighter confidence intervals on the success-rate estimates.
- CLARE and CEA variants are slow on CPU; the reported runs were completed with GPU support.
- PSO's substitution source changed from HowNet to WordNet after its first run (0% success); the pre-change numbers are now explicitly stored under `results/pso_hownet/`, while current reruns use `pso_wordnet`.
- The semantic-similarity constraint for TextFooler/CLARE/PSO-WordNet/CEA variants is a PyTorch substitute for the original TensorFlow-based USE constraint, so absolute numbers may differ slightly from the original papers' reported results.
- CEA here is a from-scratch reimplementation of the published algorithm (Algorithm 1) as a TextAttack `SearchMethod`, not a direct port of the author's [reference script](https://github.com/MingzeLucasNi/CE-Attack) (which is an unfinished companion snippet for the paper, not the exact code behind its reported numbers). The runnable variants are named by candidate source (`cea_wordnet_mlm`, `cea_textfooler`) and are not directly comparable to the paper's original candidate-source configuration.
- Per Algorithm 1, CEA variants always substitute *every* editable position on every sampled candidate (there's no "leave this word unchanged" option once it's eligible), so their modification rate can run higher than the other four attacks; the `Sim(x̃,x)` term in the objective is what keeps this in check rather than a hard cap.
- Results reflect this specific model snapshot (Prompt Guard 86M); Meta's newer Prompt Guard 2 models were not evaluated here.

**中文**
- 每种攻击样本量为50条，更大的样本量能让成功率估计的置信区间更紧。
- CLARE 和 CEA 系列在 CPU 上很慢；本文报告的运行结果使用 GPU 完成。
- PSO 的替换词来源在第一次跑完（0%成功）之后从 HowNet 改成了 WordNet；旧数字现在明确放在 `results/pso_hownet/`，当前重跑使用 `pso_wordnet`。
- TextFooler/CLARE/PSO-WordNet/CEA 系列的语义相似度约束用 PyTorch 方案替代了原论文基于 TensorFlow 的 USE 约束，因此绝对数值可能与原论文报告的结果略有出入。
- 这里的 CEA 是照着论文发表的算法（Algorithm 1）重新实现成 TextAttack 的 `SearchMethod`，不是直接照搬作者的[参考代码](https://github.com/MingzeLucasNi/CE-Attack)（那份代码是论文的一个未完成的配套脚本，不是产出论文里那些数字的确切代码）。当前可运行版本按候选源命名为 `cea_wordnet_mlm` 和 `cea_textfooler`，不能直接当作论文原始候选源配置。
- 按 Algorithm 1 的设计，CEA 系列每次采样候选文本时，所有"可编辑"的位置都会被强制替换（一旦某个位置可编辑，就没有"保持不变"这个选项），所以它的修改率可能比其他四种攻击更高；靠目标函数里的 `Sim(x̃,x)` 项来约束，而不是硬性上限。
- 结果只反映 Prompt Guard 86M 这一个模型快照，未评测 Meta 更新的 Prompt Guard 2 系列。

---

## References / 参考文献

- Ren, S., Deng, Y., He, K., & Che, W. (2019). *Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency.* ACL. (PWWS)
- Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P. (2020). *Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment.* AAAI. (TextFooler)
- Li, D., Zhang, Y., Peng, H., Chen, L., Brockett, C., Sun, M. T., & Dolan, B. (2021). *Contextualized Perturbation for Textual Adversarial Attack.* NAACL. (CLARE)
- Zang, Y., Qi, F., Yang, C., Liu, Z., Zhang, M., Liu, Q., & Sun, M. (2020). *Word-level Textual Adversarial Attacking as Combinatorial Optimization.* ACL. (PSO)
- Ni, M., Gong, Y., & Liu, W. *Cross-Entropy Attacks to Language Models via Rare Event Simulation.* [github.com/MingzeLucasNi/CE-Attack](https://github.com/MingzeLucasNi/CE-Attack). (CEA)
- Morris, J., Lifland, E., Yoo, J. Y., Grigsby, J., Jin, D., & Qi, Y. (2020). *TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP.* EMNLP.
