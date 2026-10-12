# 问题二：JPI-DLFR 模态缺失情感识别模型复现说明

> 提交包调整：为满足直接附件不超过50MB，`best.pt`置于外部网盘权重包，公开BERT的`model.safetensors`按固定revision下载。固定缺失掩码及其41个元数据JSONL、配方、面板、统计与生成代码均已保留。权重和BERT的放置位置见上级目录`外部权重下载说明.md`。

本说明只对应问题二。提交模型为 JPI-DLFR 的 **M3、训练 seed=854、最佳 epoch=7**，附件3结果使用这一份模型单独预测，未进行多模型集成。问题一的100条自生成特征和问题三的模型、附件4结果不在本包范围内。

## 1. 提交内容与复现范围

| 路径（相对于本 README） | 内容 |
|---|---|
| `run.py`、`src/` | 数据处理、模型、训练、评价、附件3预测的核心代码 |
| `configs/` | 默认参数、运行路径映射及代码和资源完整性登记 |
| `outputs/fm_v1_m3_s854/best.pt` | 最终模型，下载外部权重包后放到此位置；包含微调后的BERT、其他模型权重、训练配置、分类规则、标准化统计和来源绑定 |
| `outputs/fm_v1_m3_s854/config.json` | 该模型实际训练配置，以此为准，不应将默认配置的 seed=1111 当作本模型种子 |
| `outputs/fm_v1_m3_s854/preprocessing.json` | 训练集拟合的音频、视觉标准化参数 |
| `resources/pretrained/bert-base-uncased/` | 原始BERT配置、词表、固定来源与校验信息；公开`model.safetensors`按说明下载后放入该目录 |
| `resources/third_party/EMT-DLFR/` | 上游实现及许可证 |
| `resources/missing_data/mix_v1/` | 固定缺失掩码、样本面板、配方、元数据及校验记录 |
| `requirements.txt`、`requirements-lock.txt` | Python 依赖及固定版本 |
| `附件3_测试集预测结果.csv` | 正式提交结果，30条样本的情感极性和连续强度 |

本包不附带已安装的 Python/虚拟环境，也未重复打包官方附件二、附件三输入数据。运行者须按第3节放置官方原始文件。历史训练日志、其余模型权重、评价结果和图表未打包；可按后文重新训练或评价生成。加载已有 M3 做附件3预测不需要重新训练，也不需要附件二数据。

**精简包当前限制：**完整工程的 `run.py verify --checkpoints` 还要求 `docs/` 及被精简的资源说明文件，不能直接作为本包的验收命令。训练/评价依赖冻结清单登记的 `resources/missing_data/mix_v1/样例检查.md`，该必要原件已保留，不能再次删除。官方输入数据仍须自行放置。

## 2. 运行环境与安装

原验证环境为 Windows x64、Python **3.12.14**、PyTorch **2.8.0+cu128**、transformers **4.44.2**、NumPy **1.26.4**、SciPy **1.14.1**、scikit-learn **1.5.2**。原训练使用 NVIDIA GPU；环境说明中的 CUDA 版本指所用 PyTorch CUDA 构建。GPU运行还需兼容的 NVIDIA 驱动。没有可用GPU时，可以使用下文的 CPU 参数，但训练会较慢。

下面使用 Windows PowerShell。先进入本 README 所在的 `问题2` 目录，安装64位 Python 3.12 后执行。虚拟环境创建在上一级，不属于提交文件。若 `python` 指向其他版本，应替换为 Python 3.12 解释器的实际路径。

```powershell
python -m venv ..\problem2_env
$py = '..\problem2_env\Scripts\python.exe'
& $py -m pip install -r requirements.txt
& $py -m pip install -r requirements-lock.txt
& $py -m pip check
& $py -c "import sys, torch, transformers, numpy; print(sys.version); print(torch.__version__, transformers.__version__, numpy.__version__); print('CUDA available:', torch.cuda.is_available())"
```

后文的 `$py` 沿用这一变量；新开终端后需要重新赋值。使用解释器路径可避免 PowerShell 的环境激活脚本限制。安装依赖需要联网；数据、BERT和依赖准备好以后，训练与预测从本地读取，不需要联网下载模型。

依赖文件中的 PyTorch 链接针对 Windows/Python 3.12，Linux、macOS或其他Python版本不能直接使用该轮子。跨平台需自行安装匹配版本并重新验证。本包不保证不同设备和软件栈逐位产生相同浮点数；重新训练也不保证权重文件哈希与提交模型相同。

## 3. 官方数据准备与目录

只使用官方**对齐版本**，不要用未对齐文件替换。保持原始PKL字节、文件名、划分和样本顺序，不重新序列化或手动修改。

| 官方文件 | 本工程目标位置 | 何时需要 |
|---|---|---|
| 附件2-数据集特征文件中的 `aligned_50.pkl` | `resources/data/attachment2/aligned_50.pkl` | 重新训练、附件二 valid/test 评价 |
| 附件3-模态缺失特征样本的对齐版本 `附件3_01.pkl` 至 `附件3_30.pkl` | `resources/data/attachment3/aligned/` | 附件3预测 |

例如，先把 `$officialRoot` 改成官方数据解压目录，再执行：

```powershell
$officialRoot = 'D:\官方数据\E题数据'
New-Item -ItemType Directory -Force -Path '.\resources\data\attachment2', '.\resources\data\attachment3\aligned' | Out-Null
Copy-Item -LiteralPath (Join-Path $officialRoot '附件2-数据集特征文件\aligned_50.pkl') -Destination '.\resources\data\attachment2\aligned_50.pkl'
Get-ChildItem -LiteralPath (Join-Path $officialRoot '附件3-模态缺失特征样本\对齐版本') -Filter '*.pkl' | Copy-Item -Destination '.\resources\data\attachment3\aligned'
Get-FileHash -LiteralPath '.\resources\data\attachment2\aligned_50.pkl' -Algorithm SHA256
(Get-ChildItem -LiteralPath '.\resources\data\attachment3\aligned' -Filter '*.pkl').Count
```

附件二文件预期 SHA256：

```text
66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd
```

附件三应为30个PKL、30条样本。每个文件只有无标签 `test` 部分及 `text_bert/audio/vision` 字段。各原始文件的预期指纹可在 `configs/runtime_integrity.json` 的 `resource_files` 中查询。

配置中的 `data/aligned_50.pkl`、`pretrained/bert-base-uncased`、`dataset_construction/artifacts/mix_v1` 是兼容路径；`src/runtime.py` 会分别映射到 `resources/data/attachment2/aligned_50.pkl`、`resources/pretrained/bert-base-uncased`、`resources/missing_data/mix_v1`。不需要在工程根目录另建同名旧目录。

## 4. 数据处理规则

### 4.1 划分、输入和标签

- 附件二保留原 train/valid/test 划分，样本数为 **3395/728/727**。train用于训练与统计拟合，valid用于选模，test用于固定权重评价，不合并划分。
- 文本为 `text_bert[3,50]`，三通道为 token IDs、attention mask、token type IDs；音频为 `[50,74]`，视觉为 `[50,35]`。
- 正文长度为 `attention_mask.sum()-2`，扣除CLS和SEP，不能按音频或视觉非零行数计算。保留正文内部缺口及原生全零样本。
- 音频/视觉去掉原始第0个占位后左移，尾部补零至50步；不删除中间零行，不压紧序列。文本中的CLS、SEP和padding保持原位置。
- 音频/视觉中的NaN和正负Inf按读取器规则置零并报告；拒绝非有限文本、标签和float32转换溢出。结构异常直接报错。
- 真值极性由连续标签确定：小于0为负向，等于0为中性，大于0为正向。M3输出采用三分类logits的argmax，回归强度独立输出，不依据分类改符号、不强制中性强度归零、不裁剪。

### 4.2 标准化

音频/视觉仅在原始train的有效正文、且非整行全零的位置拟合逐通道均值和总体标准差（float64，ddof=0）；有效行内部的零坐标参与统计，标准差小于 `1e-6` 时尺度取1。原生/人工全零行和特殊位、padding保持零。

完整与缺失视图共享统计。valid、test、附件三复用 checkpoint 保存的参数，不重新拟合。单独的 `preprocessing.json` 用于阅读；推理实际使用权重中绑定的标准化参数。

### 4.3 固定缺失规则

训练使用已保存的 `mix_v1` 掩码，各epoch不重新抽样。文本被遮蔽的token换为同一词表的UNK（ID=100），attention mask不变；音频/视觉被遮蔽行置零。掩码中True表示人工移除，正文区间采用从0开始的左闭右开坐标。

主混合配方为 NONE/TAV/OTHER = 20%/60%/20%；OTHER在T、A、V、TA、TV、AV之间尽量均分。基准缺失率为0.10/0.20/0.40/0.60；单段占1/3、多段占2/3，多模态的共同缺口与共同加附加缺口按75%/25%计划分配。配额用最大余数法和固定随机流分配，最终比例可能因短样本及观测保护而变化，实际值以metadata为准。

基准遮蔽数在 `L>=2` 时为 `min(L-1,max(1,floor(r*L+0.5)))`；`L<=1`时为0。对原本有观测的受损模态，至少保留一个原始可观测位置。主集每个k最多尝试32次，失败后减小共同k，仍保留全部样本。缺失生成种子为 **20260924**，使用NumPy PCG64及按样本身份派生的独立种子，与模型训练seed=854不同。

固定参数见 `resources/missing_data/mix_v1/recipe.json`；具体区间、回退和实际缺失率见 `metadata/`；生成实现见 `src/data/construction/generator.py`。复现实验直接使用包内正式掩码，不覆盖manifest、acceptance或重新抽样。已撤销的 `exploratory_a3like` 视图不参与现用实验，撤销范围由 `exploratory_retirement.json` 记录。

配对评价在每个面板全部条件的共同适用样本交集上计算：

| 面板 | 条件 | valid/test共同样本数 |
|---|---|---|
| modality | 40%居中，T/A/V/TA/TV/AV/TAV | 721/726 |
| rate | TAV居中，0/20/40/60/80% | 713/719 |
| position | TAV 40%，头/中/尾 | 708/717 |
| shape | TAV各缺8个位置，1×8、2×4、4×2、8×1 | 409/397 |
| synchrony | TAV各40%，共同起点/独立起点 | 724/727 |

全量clean为728/727条；rate面板的0%不能拿全量clean指标替代。时间长度按正文位置数解释，没有秒级时间戳。训练恢复监督仅作用于“有效正文且原本可观测且被人工遮蔽”的位置；预测不读取完整参考视图或真值。

## 5. 模型和实际参数

M3由可微调BERT文本编码器、音频/视觉LSTM、EMT多模态交互、融合表示上的三分类头和回归头组成。训练使用完整与固定缺失双视图，共享模型参数；另有高层一致性和低层重建辅助目标。推理只输入当前观测，辅助头不生成替换输入。

| 参数 | 本次M3设置 |
|---|---|
| seed / 最大epoch / 选中epoch | 854 / 20 / 7 |
| 训练batch / 梯度累积 / 评价batch | 16 / 4 / 16 |
| 优化器 | Adam，betas=(0.9,0.999)，eps=1e-8，amsgrad=false，无学习率调度器 |
| BERT学习率 / 其余学习率 | 2e-5 / 1e-4 |
| 各参数组weight decay | 0.001 |
| A/V LSTM | 单层，hidden分别为16/32 |
| EMT | d_model=128，2层、4头、前馈扩展倍数4 |
| 融合表示 / 分类头 | 128维 / Linear(128,3) |
| 每视图预测损失 | CrossEntropy + 0.5×L1；完整与缺失视图各权重1 |
| 高层一致性 / 低层重建权重 | 1 / 1；低层使用SmoothL1 |
| 类别权重 / label smoothing | 无 / 0 |
| 早停 | patience=8，比较容差1e-12 |
| 选模 | clean/mixed valid指标等权平均，依次最大Accuracy、最大Macro-F1、最小MAE、最早epoch |
| 随机与加载 | num_workers=0，cpu_threads=4，BERT梯度检查点开启 |

完整配置以 `outputs/fm_v1_m3_s854/config.json` 为准，未在表中列出的dropout等参数也保存在该文件中。不要为复现修改`src/`和`configs/`中已登记指纹的文件；需要新实验时使用新的配置副本和新run名称。

已有模型 SHA256：

```text
524f9e91383601dfaee4d8ce7c8db7e90710306cd81784d2634cc953d1c74284
```

## 6. 用提交模型复现附件3预测

本节只需要附件三对齐输入、已安装依赖、包内M3权重与本地BERT。无需附件二或训练用冻结数据验收说明文件。

```powershell
# 只读：检查权重身份和输入结构，不执行模型前向、不生成结果。
& $py -B -X utf8 run.py predict --version four-models --run-name fm_v1_m3_s854 --expected-seed 854 --check-only

# 正式推断：默认优先GPU，无GPU则CPU；生成新的时间戳结果目录。
& $py -B -X utf8 run.py predict --version four-models --run-name fm_v1_m3_s854 --expected-seed 854 --device auto --batch-size 16
```

输入不复制进包内时，可以在上述命令追加 `--input-dir 'D:\实际路径\附件3-模态缺失特征样本\对齐版本'`。相对输入路径按本工程目录解析。强制CPU时使用 `--device cpu`。

只读检查成功输出 `checked_not_predicted`；这不表示已经完成推断。正式预测成功输出 `status: completed`，新文件位于 `outputs/attachment3/fm_v1_m3_s854_时间戳/`：

- `predictions.csv`：30条完整预测，UTF-8 BOM，保留原始浮点精度。
- `summary.json`：模型/输入指纹、覆盖检查、环境及预测分布。
- `results.md`：结果展示表；仅展示时将强度保留六位小数。

正式提交CSV位于本目录，文件名为 `附件3_测试集预测结果.csv`。重新推断不会自动覆盖它；重复执行默认另建新目录。也可指定 `--output-dir outputs/attachment3/reproduce_s854`，但该目录必须尚不存在。

CSV字段说明：

| 字段 | 含义 |
|---|---|
| sample_id | 由原始文件名派生的确定性编号，如附件3_01 |
| source_file | 原始PKL文件名，用于一一对应 |
| sample_index | 文件内从0开始的索引；本批文件均为0 |
| polarity | Negative、Neutral或Positive |
| intensity | 未裁剪的连续情感强度 |

附件三包含3条完整样本，也全部纳入30条覆盖，不剔除。附件三没有真值，不能计算Accuracy、F1、MAE或Pearson；30/30覆盖率不代表准确率。提交CSV的SHA256为 `19ecfaf19f1eb4be4573e4fb0f27b2835cf5fee7e996075665a7099cc2fc2da6`。原预测使用CUDA、batch=16、PyTorch 2.8.0+cu128；其他设备可能出现微小数值差异，应按sample_id比较类别与强度，而非要求重新导出的CSV字节完全一致。

## 7. 重新训练、评价与对照实验

**先决条件：**完成第2、3节，并保留第8节指出的冻结文件原件。下面的训练/评价命令不会绕过缺失数据清单校验。

### 7.1 重新训练M3

```powershell
& $py -B -X utf8 run.py train --version four-models --method M3 --config outputs/fm_v1_m3_s854/config.json --seed 854 --device cuda --run-name reproduce_m3_s854
```

这会从基础BERT和新初始化的EMT开始训练，使用固定缺失数据和原选模规则，不是从提交的`best.pt`继续训练。结果保存在 `outputs/reproduce_m3_s854/`，包含新`best.pt`、实际配置和训练记录。run名称必须未被占用；再次运行应更换名称。CPU训练可将 `--device cuda` 改为 `--device cpu`。

不能把最大20轮改成固定训练7轮来冒充原过程；第7轮是原训练按valid选出的最佳权重。重新训练后的最佳轮次及指标以新运行实际结果为准。

### 7.2 对已有M3做有标签评价

```powershell
& $py -B -X utf8 run.py evaluate --version four-models --run-name fm_v1_m3_s854 --split valid --paired --device cuda
& $py -B -X utf8 run.py evaluate --version four-models --run-name fm_v1_m3_s854 --split test --paired --device cuda
```

评价不更新参数，输出位于 `outputs/fm_v1_m3_s854/evaluations/` 的新时间戳目录。省略`--paired`仅计算clean/mixed主条件；保留它会增加受控缺失面板。要评价新训练模型，将run名称改为 `reproduce_m3_s854`。

评价指标为三分类Accuracy、Macro-F1、连续强度MAE与Pearson。valid评价还会与checkpoint保存的选模指标进行严格一致性检查，跨设备数值误差也可能触发报错，应先使用原环境复核，不能修改权重或阈值使其通过。

### 7.3 其他问题二方案

| 方法 | 输入与目标 | 当前入口的选模协议 |
|---|---|---|
| M0 | 完整输入、回归基线 | 原标准化回归，valid MAE |
| M1 | 完整输入、分类与回归联合预测 | clean valid Accuracy为首要指标 |
| M2 | 缺失输入、回归及完整DLFR | 原标准化缺失流程，mixed valid MAE |
| M3 | 缺失输入、联合预测及完整DLFR | clean/mixed valid联合选模 |

```powershell
& $py -B -X utf8 run.py train --version four-models --method M0 --seed 1111 --device cuda --run-name reproduce_m0_s1111
& $py -B -X utf8 run.py train --version four-models --method M1 --seed 854 --device cuda --run-name reproduce_m1_s854
& $py -B -X utf8 run.py train --version four-models --method M2 --seed 1111 --device cuda --run-name reproduce_m2_s1111
```

M0/M2不要传入M3配置；它们采用各自的MAE流程，不能与历史Accuracy版M0/M2混称。M0/M2评价入口只支持`--split test`；M2的该入口还会评价valid/test配对面板。M1/M3支持`valid`和`test`。

问题二已有汇总中的M1/M3多种子统计使用214、626、854、1111、6458、9609六个seed。复现这类汇总需分别训练各seed、在同一条件评价，再对每个run的指标作等权平均；不能以本包唯一的M3 seed=854结果代替六种子均值。其余历史模型权重及跨实验汇总产物未随包提交。本包的附件3结果始终只对应seed=854。

## 8. 文件校验与精简包限制

当前训练数据manifest登记了 `resources/missing_data/mix_v1/样例检查.md`。该文件已从原工程恢复并核对原字节，随本包保留；它不是可以任意删除的阅读材料。原件为 **6391字节**，SHA256为：

```text
aa38f0e5c5d350f66b751a489cade5de489d0a24b83b8f44b83df93f49c3a821
```

若后续删除或修改这个文件，训练和附件二评价会报 `Artifact integrity failure: 样例检查.md`。必须从未精简的原工程同路径恢复原件；只创建同名空文件或重写文字不满足校验。恢复动作不改变训练数据、掩码或模型。本说明不能替代该原件。

完整工程的 `run.py verify --checkpoints` 还检查`docs/`及全部原资源集合，精简包移除了其中的报告/说明，因此该总检查仍会报缺项。不要通过删除登记记录、刷新acceptance或禁用来源检查使其“通过”。本包的模型输入检查使用第6节的 `predict --check-only`；训练数据和来源检查仍由训练/评价入口执行。预测检查通过不等于已运行有标签评价或完成重新训练。

官方附件二按第3节就位后，可执行下面的只读专项检查，核验冻结文件、源数据哈希、当前代码及模型数据绑定，不训练、不写出文件：

```powershell
& $py -B -X utf8 -c "from src.runtime import ROOT; from src.data.missing import MixDataStore; from src.four_model_checkpoint import load_checkpoint; s=MixDataStore(ROOT/'resources/missing_data/mix_v1', ROOT/'resources/data/attachment2/aligned_50.pkl'); c=load_checkpoint(ROOT/'outputs/fm_v1_m3_s854/best.pt', ROOT); assert s.binding()==c['data_binding'], 'Checkpoint data binding mismatch'; print('Frozen data and checkpoint binding: passed')"
```

常见错误处理：

| 报错 | 处理 |
|---|---|
| `Attachment 3 input directory missing` | 放置附件三对齐版，或显式传入`--input-dir` |
| `Reader source fingerprint mismatch` | 检查附件二是否为原始对齐版；不要修改PKL或清单哈希 |
| `Artifact integrity failure` | 按错误指出的路径恢复冻结原件，不重新生成来替换已绑定文件 |
| `Required project files are missing`（verify） | 完整工程总检查与精简包布局不同，见本节说明 |
| `CUDA unavailable` | 检查GPU驱动/依赖，或对对应命令指定`--device cpu` |
| 输出目录已存在 | 使用新的run名称/输出目录，不覆盖提交模型与结果 |
| `Registered implementation content changed` | 检查代码、配置是否被修改，恢复对应原件 |

本说明编写时未重新训练模型。已用提交包代码、提交模型及外部原始附件三完成30文件/30样本的只读输入与checkpoint检查；恢复必要原件后，另以外部原始附件二通过冻结文件、源数据哈希、模型数据绑定及保存配置与checkpoint一致性检查。外部数据仅用于检查，未复制进本包。上述检查不等于重新推断或完整复现实验，正式CSV来自已完成的原预测。

## 9. 上游资源与许可证

EMT-DLFR上游为 `https://github.com/sunlicai/EMT-DLFR`，固定提交 `e4216215565292360e9c3fc43c8cd6e127df19c2`，许可证保留于 `resources/third_party/EMT-DLFR/LICENSE`。本工程包含任务适配和联合预测改进，不能等同于完全复现原论文数据实验。

BERT来源为 `google-bert/bert-base-uncased`，固定revision `86b5e0934494bd15c9632b12f734a8a67f723594`，来源、下载链接和SHA256见 `resources/pretrained/bert-base-uncased/source_manifest.json`，许可证也在同目录。`model.safetensors`是原始初始化权重；提交的`best.pt`包含微调后的BERT。当前模型构建先读取本地原始BERT再恢复完整checkpoint，因此不能直接删除基础权重或用它替代训练后的M3权重。
