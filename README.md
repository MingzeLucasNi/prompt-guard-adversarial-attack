# Adversarial Robustness Evaluation of Meta's Prompt Guard 86M
# 对 Meta Prompt Guard 86M 的对抗鲁棒性评测

Evaluating how well Meta's open-source prompt-injection/jailbreak classifier holds up under standard, published adversarial NLP attacks.

用五种对抗NLP攻击方法，评测 Meta 开源的 prompt injection / jailbreak 分类器的鲁棒性。

---

## Overview / 项目简介

**EN** — This project measures the adversarial robustness of [Prompt Guard 86M](https://huggingface.co/meta-llama/Prompt-Guard-86M), Meta's open-weight classifier for detecting prompt injection and jailbreak attempts. Using the [TextAttack](https://github.com/QData/TextAttack) framework, five word-substitution attack algorithms from the adversarial NLP literature — **PWWS**, **TextFooler**, **CLARE**, **PSO**, and **CEA** (our own Cross-Entropy Attack) — are applied to prompts the model already correctly flags as malicious, to see how many can be rewritten (while preserving their original meaning) into text the model misclassifies as benign.

**中文** — 本项目评测 Meta 开源的 prompt injection / jailbreak 分类器 [Prompt Guard 86M](https://huggingface.co/meta-llama/Prompt-Guard-86M) 的对抗鲁棒性。使用 [TextAttack](https://github.com/QData/TextAttack) 框架，对已经被模型正确判定为恶意的 prompt，应用五种对抗NLP文献中的词级替换攻击算法——**PWWS**、**TextFooler**、**CLARE**、**PSO**，以及我们自己提出的 **CEA**（Cross-Entropy Attack）——衡量在保持原意的前提下，有多少比例可以被改写成模型误判为无害的文本。

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
  - **PSO** (Zang et al., 2020) — particle swarm optimization over synonym substitutions. Originally proposed with HowNet sememe-based synonyms; here it uses **WordNet** instead (see note below).
  - **CEA** (Ni, Gong, & Liu — [Cross-Entropy Attacks to Language Models via Rare Event Simulation](https://github.com/MingzeLucasNi/CE-Attack)) — our own attack, ported from its reference implementation into a native TextAttack `SearchMethod` (`cea_search.py`) so it runs through the exact same pipeline as the other four. It treats adversarial-example generation as a rare-event simulation: a categorical substitution distribution per editable word position is refined over iterations by repeatedly sampling candidate texts, scoring them by `m(F(x̃)) · Sim(x̃, x)` (targeted-class score times sentence-embedding similarity to the original), and re-fitting each position's distribution by maximum likelihood on the top-`ρ` "elite" samples.
- **Substitution source, unified across PSO and CEA**: both originally relied on HowNet, whose sememe-annotated synonym sets turned out to be sparse for this kind of English text (see PSO's 0% result below). Both now draw from **WordNet synonyms ∪ RoBERTa masked-language-model predictions** — a deliberately larger candidate pool per position than the original papers, so search quality isn't bottlenecked by a thin substitution vocabulary.
- **Sample size**: 50 randomly sampled prompts per attack (from the 247 correctly-detected examples), query budget 1000 per example (CEA needs a higher budget to reach its full 50-iteration schedule — see Reproduction).
- For TextFooler/CLARE, the semantic-similarity constraint is implemented with a PyTorch `sentence-transformers` model rather than the original papers' TensorFlow-based Universal Sentence Encoder, for environment portability (no functional difference to the attack's goal). CEA's objective uses the same `sentence-transformers` model for its similarity term.

**中文**
- **目标模型**：Prompt Guard 86M（三分类：`BENIGN` / `INJECTION` / `JAILBREAK`），使用官方权重的原样公开镜像。
- **种子数据**：[`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections)（共609条），取其中标注为注入攻击的263条作为攻击起点。
- **基线**：这263条中，模型在**攻击前**能正确识别247条（93.9%）为非无害——这就是攻击要突破的起始检出率。
- **攻击目标**：**定向**逃逸攻击，目标标签为 `BENIGN`。只有扰动后的文本被误判为无害才算成功——单纯在 `INJECTION` 和 `JAILBREAK` 之间切换不算真正的逃逸。
- **评测的攻击方法**（均通过 TextAttack 实现）：
  - **PWWS**（Ren et al., 2019）——基于词语显著性的 WordNet 同义词替换。
  - **TextFooler**（Jin et al., 2020）——反义词过滤词向量替换，附加词性和语义相似度约束。
  - **CLARE**（Li et al., 2021）——基于掩码语言模型的上下文感知替换/插入/合并扰动。
  - **PSO**（Zang et al., 2020）——同义词替换 + 粒子群优化搜索。原论文用 HowNet 义原同义词，这里改用 **WordNet**（原因见下）。
  - **CEA**（Ni, Gong, Liu ——[Cross-Entropy Attacks to Language Models via Rare Event Simulation](https://github.com/MingzeLucasNi/CE-Attack)）——我们自己的攻击方法，从原始参考实现移植成了原生的 TextAttack `SearchMethod`（`cea_search.py`），跑在和其他四种攻击完全一样的流程里，方便直接对比。核心思路是把生成对抗样本看作稀有事件模拟：给每个可编辑的词位置维护一个替换词的类别分布，反复采样候选文本、用 `m(F(x̃))·Sim(x̃,x)`（目标类别得分 × 与原文的句向量相似度）打分，取分数最高的前 `ρ` 比例作为"精英样本"，再用极大似然估计更新每个位置的分布，如此迭代。
- **PSO 和 CEA 统一了替换词来源**：两者原本都依赖 HowNet，但其义原标注的同义词集合对这类英文文本来说太稀疏了（下面 PSO 的 0% 结果就是证据）。现在两者都改用 **WordNet 同义词 ∪ RoBERTa 掩码语言模型预测**——比原论文更大的候选词池，避免搜索质量被过窄的替换词表卡住。
- **样本规模**：每种攻击从247条已正确检出的样本中随机抽取50条，每条查询预算1000次（CEA 想跑满50轮迭代需要更高的预算，见"复现方法"）。
- TextFooler/CLARE 的语义相似度约束改用基于 PyTorch 的 `sentence-transformers` 模型实现，而非原论文中基于 TensorFlow 的 Universal Sentence Encoder（纯粹是为了环境可移植性，攻击目标本身没有变化）。CEA 的目标函数里的相似度项也用的同一个 `sentence-transformers` 模型。

---

## Results / 结果

| Attack | Samples attacked | Evaded to BENIGN | Still detected | Attack success rate |
|---|---|---|---|---|
| *Baseline (no attack)* | 263 | — | 247 (93.9%) | — |
| **PWWS** | 50 | 17 | 33 | **34.0%** |
| **TextFooler** | 50 | 31 | 19 | **62.0%** |
| **PSO** (HowNet, superseded) | 50 | 0 | 50 | 0.0% |
| **PSO** (WordNet, current code) | — | — | — | pending re-run |
| **CEA** | — | — | — | pending (running on a GPU server; too slow on CPU) |
| **CLARE** | — | — | — | pending (running on a GPU server; too slow on CPU) |

`results/pso/` currently still holds the **old HowNet-based** run (0% — see Key Findings); the code has since switched PSO to WordNet (see Methodology), so this number needs a re-run before it's representative of the current codebase.

Full per-example results (before/after label, confidence, and exactly which words were changed) are in [`results/pwws/`](results/pwws/), [`results/textfooler/`](results/textfooler/), and [`results/pso/`](results/pso/) — see `report.md` in each folder for a readable summary, or `details.json` for structured data.

`results/pso/` 里目前还是**旧的、基于 HowNet 的跑法**（0%，见"关键发现"）；代码已经改成用 WordNet 了（见"方法"），所以这个数字需要重跑之后才能代表现在的代码。

完整的逐条结果（攻击前后标签、置信度、以及具体改动了哪些词）见 [`results/pwws/`](results/pwws/)、[`results/textfooler/`](results/textfooler/) 和 [`results/pso/`](results/pso/) 文件夹，每个文件夹下 `report.md` 是可读报告，`details.json` 是结构化数据。

---

## Key Findings / 关键发现

**EN**
- Prompt Guard 86M has a high detection rate (93.9%) on the known-injection benchmark before any attack — it works well against the patterns it was trained on.
- A simple, black-box synonym-substitution attack (PWWS) with no gradient access already flips **34%** of correctly-caught prompts to benign while preserving their meaning.
- TextFooler — which searches a larger candidate space under part-of-speech and semantic-similarity constraints — nearly doubles that to **62%**, suggesting the model's decision boundary is not robust to small, meaning-preserving lexical substitutions.
- Attack effectiveness is not uniform across algorithms: PSO, despite using the full 1000-query budget on every example, achieved **0%** success. Its candidate substitutions come from HowNet sense annotations, a much smaller and coarser synonym source than WordNet (PWWS) or counter-fitted embeddings (TextFooler) for this kind of English text — a reminder that an attack's reported strength is inseparable from its underlying substitution vocabulary, not just its search strategy.
- Taken together, this shows neural safety classifiers, including Prompt Guard, are generally vulnerable to word-substitution attacks unless explicitly hardened against them (e.g. via adversarial training) — but *how* vulnerable depends heavily on which attack algorithm (and word-substitution source) is used.

**中文**
- Prompt Guard 86M 在攻击前对已知注入样本的检出率很高（93.9%）——对训练时见过的模式识别得不错。
- 一个不需要梯度信息的简单黑盒同义词替换攻击（PWWS），在保持原意的前提下，已经能让 **34%** 的样本从"被正确拦截"变成"被判定无害"。
- 搜索空间更大、附加了词性和语义相似度约束的 TextFooler，把这个比例几乎翻倍到 **62%**，说明模型的决策边界对保持语义的小幅词汇替换并不鲁棒。
- 攻击效果在不同算法之间差异很大：PSO 虽然每条样本都用满了1000次查询预算，成功率却是 **0%**。它的候选替换词来自 HowNet 义原标注，对这类英文文本而言，是比 WordNet（PWWS）或反义词过滤词向量（TextFooler）小得多、也粗糙得多的同义词来源——这提醒我们，一个攻击方法报告出来的强弱，很大程度上取决于它底层的替换词表，而不只是搜索策略本身。
- 总的来看，这说明包括 Prompt Guard 在内的神经网络安全分类器，如果没有专门做过对抗训练加固，通常都容易被词级替换攻击攻破——但"有多容易被攻破"很大程度上取决于具体用的是哪种攻击算法（以及背后的替换词来源）。

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
python run_attack.py pso         # slow on CPU (full query budget per example)
python run_attack.py clare       # search + masked-LM based, slowest; a GPU is strongly recommended
python run_attack.py cea --query-budget 5000   # needs a bigger budget for its full 50-iteration schedule (100 candidates/iter)
python compare_results.py        # prints a summary table across all attacks that have been run
```

CUDA is used automatically when available (falls back to CPU otherwise). Each `run_attack.py <recipe>` writes its output to `results/<recipe>/`.

有 CUDA 会自动使用，否则回退到 CPU。每次 `run_attack.py <recipe>` 的结果都会写入 `results/<recipe>/` 文件夹。

---

## Project Structure / 项目结构

```
baseline.py           # computes pre-attack detection rate, selects attack seeds
run_attack.py          # runs one of the 5 attacks end-to-end
custom_recipes.py       # TF-free TextFooler/CLARE, WordNet-based PSO, and CEA builders
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
- CLARE (masked-language-model based perturbation) is too slow to complete on CPU; it is being run on a GPU server and this README will be updated once that result is in.
- PSO's substitution source changed from HowNet to WordNet after its first run (0% success); `results/pso/` still holds the pre-change HowNet numbers until it's re-run.
- The semantic-similarity constraint for TextFooler/CLARE is a PyTorch substitute for the original TensorFlow-based USE constraint, so absolute numbers may differ slightly from the original papers' reported results.
- CEA here is a from-scratch reimplementation of the published algorithm (Algorithm 1) as a TextAttack `SearchMethod`, not a direct port of the author's [reference script](https://github.com/MingzeLucasNi/CE-Attack) (which is an unfinished companion snippet for the paper, not the exact code behind its reported numbers) — so this project's CEA results are not directly comparable to the numbers in the paper itself.
- Per Algorithm 1, CEA always substitutes *every* editable position on every sampled candidate (there's no "leave this word unchanged" option once it's eligible), so its modification rate can run higher than the other four attacks; the `Sim(x̃,x)` term in its objective is what keeps this in check rather than a hard cap.
- Results reflect this specific model snapshot (Prompt Guard 86M); Meta's newer Prompt Guard 2 models were not evaluated here.

**中文**
- 每种攻击样本量为50条，更大的样本量能让成功率估计的置信区间更紧。
- CLARE（基于掩码语言模型的扰动）在 CPU 上太慢跑不完，正在服务器GPU上跑，跑完后会更新到本文档。
- PSO 的替换词来源在第一次跑完（0%成功）之后从 HowNet 改成了 WordNet；`results/pso/` 里现在还是改之前基于 HowNet 的数字，等重跑后会更新。
- TextFooler/CLARE 的语义相似度约束用 PyTorch 方案替代了原论文基于 TensorFlow 的 USE 约束，因此绝对数值可能与原论文报告的结果略有出入。
- 这里的 CEA 是照着论文发表的算法（Algorithm 1）重新实现成 TextAttack 的 `SearchMethod`，不是直接照搬作者的[参考代码](https://github.com/MingzeLucasNi/CE-Attack)（那份代码是论文的一个未完成的配套脚本，不是产出论文里那些数字的确切代码）——所以这个项目里跑出来的 CEA 结果，和论文里报告的数字不能直接对比。
- 按 Algorithm 1 的设计，CEA 每次采样候选文本时，所有"可编辑"的位置都会被强制替换（一旦某个位置可编辑，就没有"保持不变"这个选项），所以它的修改率可能比其他四种攻击更高；靠目标函数里的 `Sim(x̃,x)` 项来约束，而不是硬性上限。
- 结果只反映 Prompt Guard 86M 这一个模型快照，未评测 Meta 更新的 Prompt Guard 2 系列。

---

## References / 参考文献

- Ren, S., Deng, Y., He, K., & Che, W. (2019). *Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency.* ACL. (PWWS)
- Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P. (2020). *Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment.* AAAI. (TextFooler)
- Li, D., Zhang, Y., Peng, H., Chen, L., Brockett, C., Sun, M. T., & Dolan, B. (2021). *Contextualized Perturbation for Textual Adversarial Attack.* NAACL. (CLARE)
- Zang, Y., Qi, F., Yang, C., Liu, Z., Zhang, M., Liu, Q., & Sun, M. (2020). *Word-level Textual Adversarial Attacking as Combinatorial Optimization.* ACL. (PSO)
- Ni, M., Gong, Y., & Liu, W. *Cross-Entropy Attacks to Language Models via Rare Event Simulation.* [github.com/MingzeLucasNi/CE-Attack](https://github.com/MingzeLucasNi/CE-Attack). (CEA)
- Morris, J., Lifland, E., Yoo, J. Y., Grigsby, J., Jin, D., & Qi, Y. (2020). *TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP.* EMNLP.
