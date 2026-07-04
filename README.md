# 攻击 Meta Prompt Guard (86M)

用 TextAttack 的标准 recipe 对 `meta-llama/Prompt-Guard-86M`（下载自免gate的镜像
`Niansuh/Prompt-Guard-86M`，权重与官方一致）做 red-team 式对抗攻击：

对 `deepset/prompt-injections` 数据集里被模型判定为 `INJECTION`/`JAILBREAK`
（即已经被正确拦截）的文本做词级扰动，看能不能让模型把它们误判成 `BENIGN`
（绕过检测）。这是标准的"有害→无害"逃逸攻击方向。

## 环境准备（已装好，仅记录）

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

已验证:
- 模型 3 分类: `{0: BENIGN, 1: INJECTION, 2: JAILBREAK}`
- 数据集: train 546条 + test 116条，其中 label=1(injection) 共 263 条

### 踩过的坑（这台机器上遇到的，已经在代码里修好了，不用你管，记录一下原因）

1. **不要用 MPS(Apple GPU)**：变长文本每次 forward 都要重新编译 Metal
   计算图，慢到离谱。两个脚本的设备选择逻辑是"有 CUDA 用 CUDA，否则用
   CPU"，故意跳过 MPS——在 Mac 上会自动落到 CPU（反而比 MPS 快），拿去
   服务器跑如果有 CUDA 会自动用上，不用改代码。
2. **这台 macOS(26.5.1 beta) 上 TensorFlow 直接崩溃**（`import tensorflow`
   触发 `mutex lock failed`），而 TextFooler/CLARE 官方 recipe 的语义相似度
   约束（Universal Sentence Encoder）依赖 `tensorflow_hub`。所以：
   - 卸载了 `tensorflow`/`tensorflow-hub`。
   - 写了 `sbert_encoder.py`，用纯 PyTorch 的 `sentence-transformers`
     (all-MiniLM-L6-v2) 实现同样接口的语义相似度约束。
   - `custom_recipes.py` 里的 `build_textfooler` / `build_clare` 是官方
     recipe 的等价版本，只是把 USE 换成了这个 SBERT 约束（阈值做了对应调整，
     不是严格复现论文数字，但攻防语义一致）。
   - 两个脚本开头都设了 `USE_TF=0`，避免 `transformers` 库自己去探测 TF
     可用性时再次触发这个崩溃。
3. **攻击方向的一个真实 bug（已修好）**：TextAttack 各 recipe 默认用
   `UntargetedClassification`，"成功"只要标签变了就算——包括从
   `INJECTION` 变成 `JAILBREAK`，但这**不是**绕过 Prompt Guard！第一次跑
   PWWS 时发现 3 个"成功"样本全是 1↔2 之间互相跳，没有一个真正跳到
   `BENIGN`。改成了 `TargetedClassification(target_class=0)`（只有真正
   被判成 BENIGN 才算攻击成功），并注意 TextAttack 的 `Attack.__init__`
   会把 goal function 绑定进 search method，构造后直接改
   `attack.goal_function` 不生效，所以 `run_attack.py` 里是重新构造了
   一个 `Attack` 对象。

## 第一步：跑 baseline（攻击前的检出率）

```bash
source .venv/bin/activate
python baseline.py
```

会打印类似:
```
BASELINE detection rate (before any attack): 247/263 = 93.92%
```

并把这 247 条"确实被模型正确拦截"的文本存到 `detected_injections.json`
（后面攻击只对这些"确实被抓到的"样本做，攻击本来就没抓到的样本没有意义）。

## 第二步：分别跑 4 种攻击（各挑 50 条样本，可用 `--n` 调整）

每个都会实时打印 TextAttack 自带的进度条 (`x/50 [==>...]`)，你能直接看到进度。

```bash
python run_attack.py pwws          # WordNet 同义词替换，实测约 20-40秒/条
python run_attack.py textfooler    # 词向量替换 + SBERT 语义约束，实测约 45-50秒/条
python run_attack.py clare         # RoBERTa mask-infill 上下文替换，单条耗时随文本长度波动较大
python run_attack.py pso           # 粒子群优化 + HowNet 同义词，最慢（population=60, iters=20，默认query-budget=1000）
```

CPU 上 50 条 PWWS 大概 15-30 分钟，TextFooler 更慢一些；CLARE / PSO 建议先用
`--n 10` 或 `--n 20` 试跑感受一下单条耗时，觉得可接受再放到 50。
也可以用 `--query-budget` 调小上限（比如 300）以牺牲一点攻击成功率换取
更快的运行速度。

每跑完一个 recipe 会生成 4 个文件，方便检查（数字汇总 + 逐条细节 + 人类可读报告 + 原始日志）:

- `summary_<recipe>.json` — 该 recipe 的汇总数字（攻击成功率、样本数等）
- `details_<recipe>.json` — **逐条结构化数据**：每条样本的攻击前/后标签、
  置信度、P(BENIGN)、查询次数、**具体被换掉的词列表**（`removed_words` /
  `added_words`）、以及改动前后的完整文本（纯文本版 `*_text_plain` 和
  带 `[[ ]]` 标记版 `*_text_marked`）
- `report_<recipe>.md` — 和上面同样的内容，排成人类可读的 Markdown：
  开头是全部样本的总览表格（一眼看出哪几条成功、置信度多少），下面是
  逐条详情，每条都写明"被删掉/替换掉的原词"和"替换成/新插入的词"，
  并把原文和攻击后文本都用 `[[ ]]` 标出具体改动的位置，不用自己肉眼
  找哪里不一样
- `results_<recipe>.csv` — TextAttack 自带的原始逐条日志（同样带 `[[ ]]`
  标记，可以用 Excel/Numbers 打开筛选/排序）

想快速检查某个 recipe 跑得怎么样，直接打开 `report_<recipe>.md` 看总览表格
和逐条详情就够了；想写代码进一步分析（比如统计最常被换的词），读
`details_<recipe>.json`。

## 第三步：汇总对比

```bash
python compare_results.py
```

输出每个 recipe 的：攻击样本数 / 成功逃逸数 / 仍被拦截数 / 攻击成功率(%)。

## 在服务器上跑

```bash
git clone https://github.com/MingzeLucasNi/prompt-guard-adversarial-attack.git
cd prompt-guard-adversarial-attack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('omw-1.4'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('universal_tagset'); nltk.download('punkt')"
python baseline.py
python run_attack.py pwws
python run_attack.py textfooler
python run_attack.py clare
python run_attack.py pso
```

有 CUDA 的话代码会自动检测并使用（见上面"踩过的坑"第1条），不需要额外配置；
`--n 50` 是默认值，CUDA 上应该比这台 Mac 快很多，不需要再分批小样本试跑。

生成的 `results_*.csv` / `summary_*.json` / `details_*.json` / `report_*.md`
默认被 `.gitignore` 排除掉了（属于"跑一次生成一次"的产物，不同机器/参数
跑出来的数字会不一样，没必要塞进版本历史）。如果想把服务器上跑出来的结果
带回本地看，直接 `scp` 或者手动 `git add -f` 这几个文件再 commit 都可以。

## 术语说明（避免和"baseline"数字搞混）

- `baseline.py` 里的 93.92% 是模型在**整个数据集**上、攻击前的检出率。
- 每个 `run_attack.py <recipe>` 里的 "n_attempted" 都是从已经被正确检出的
  247条里抽样 50条，所以这一层的"攻击前检出率"恒为 100%（因为样本本来就是
  从"已被检出"的池子里选的）——真正有意义的数字是 `attack_success_rate_pct`，
  即攻击后有多少比例从"被拦截"变成"被判定为BENIGN"。
- "攻击后检出率" = 100% - 攻击成功率，即攻防两方视角下 Prompt Guard 在该
  攻击方法下的鲁棒检出率。
