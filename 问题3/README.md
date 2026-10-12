# 问题3：M1预测与双目标解释复现说明

## 交付文件

- `附件4_测试集预测与解释结果.csv`：正式提交结果，20行、104列。`evidence/keyframes/`含表中实际引用的34张关键帧。
- `src/`：附件4读取、三模态精确Shapley遮蔽、局部窗口解释、忠实度检验、QG-HTA时间回溯、表格导出和审计代码。
- `q3_handoff/q3_adapter/`与`q3_handoff/original_source/`：冻结完整输入M1预测接口和原模型实现。原工程内部运行标识`mt_m0_avstd_s1111_v1`保持不变，论文称M1。
- `q3_handoff/checkpoint/`：实际配置、预处理统计、验证记录；`best.pt`从网盘权重包取得。
- `q3_handoff/original_source/pretrained/bert-base-uncased/`：公开BERT配置、词表、来源和哈希；`model.safetensors`按固定revision另行下载。
- `results/`：本次正式运行的中间预测、逐样本解释、时间轴、忠实度和审计记录。历史运行记录中的原机路径已替换为相对路径；为控制体积，展示图只保留SVG，原项目中的PNG未重复装入。

## 输入与环境

官方附件4原始数据不重复打包。设置环境变量`E_TOPIC_DATA_ROOT`为官方`E题数据`目录的绝对路径，其中应有`附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本/01.pkl`等文件。若不设置，则代码从本目录的`official_data/`读取。

在本目录安装环境：

```powershell
conda env create --prefix .\.env-q3 --file .\environment-q3.yml
$py = '.\.env-q3\python.exe'
```

按上级`外部权重下载说明.md`放置M1的`best.pt`和公开BERT的`model.safetensors`并核对SHA256。官方附件2的验证集只在复核验证指标时需要；附件4推断及解释不使用标签。

## 专项模型与数据处理规则

问题三不另行训练模型，而是冻结问题二的完整三模态联合预测模型M1，在其输出上增加无训练参数的解释层。原工程配置中的`method=M0`及运行标识是历史内部命名，提交材料统一称该完整输入模型为M1；它不是处理缺失模态的M3。M1使用BERT文本编码、音频与视觉LSTM编码及两层四头EMT融合，输出负向／中性／正向三类logits和连续情感强度。实际结构、类别顺序和参数以`q3_handoff/checkpoint/config.json`及源码为准，环境版本以`environment-q3.yml`为准。

M1沿用官方附件2的train／valid／test划分（分别为3395／728／727条）：仅训练集拟合音视频标准化统计，验证集选取最佳checkpoint，测试集仅作独立评价；附件4的20条样本仅用于无标签推断。文本保留CLS、SEP及padding；音视频移除原特征数组的CLS槽位、左移并在尾部补零。原量纲全零视觉行在标准化前识别并保持为零。实际预处理统计见`q3_handoff/checkpoint/preprocessing.json`，不得重新用附件4拟合。模型输入及有效长度按`q3_handoff/q3_adapter/`的读取协议构造。

实际训练配置记录于`config.json`：随机种子1111、batch size 16、梯度累积4、交叉熵加0.5倍L1损失，BERT学习率`2e-5`，其余模块学习率`1e-4`；验证集选出的最佳epoch为3，早停耐心值8。问题三复现仅需加载该已训练checkpoint，不需重新训练。权重下载地址、放置位置及哈希见上级`外部权重下载说明.md`。

解释对分类固定完整输入的预测类别，以其相对最强竞争类别的logit边际为目标；对强度使用未经裁剪的回归输出。三模态层枚举全部8种保留组合计算精确Shapley值；局部层以宽度1、3、5的连续窗口遮蔽，最多选出3个互不重叠的高影响窗口。文本正文以`[UNK]`遮蔽，音视频有效原始特征行置零，序列长度不变。输出表示相对于该遮蔽基线的敏感性，不是因果解释；跨模态作用比较使用Shapley，不能直接比较各模态局部归一化峰值。

问题一QG-HTA仅用于把可信文本证据回溯到原视频播放时间；附件4未提供官方逐特征行时间戳，不按视频时长均分特征行。无可靠时间的证据留空并保留映射状态。样本13的视觉特征全零，按不可用模态处理，不把原视频画面冒称模型视觉证据。解释验证和逐样本审计文件保留在`results/`；正式交付结果是本目录的`附件4_测试集预测与解释结果.csv`。

## 复现附件4预测与基础解释

以下命令以本目录为当前目录；GPU可用时选`cuda`，否则改为`cpu`：

```powershell
& $py -B .\src\audit_attachment4.py
& $py -B .\src\run_attachment4.py --device cuda
& $py -B .\src\run_fidelity_validation.py --device cuda --random-repeats 10
& $py -B .\src\export_tables.py
& $py -B .\src\audit_results.py
```

训练集、验证集和测试集沿用官方附件2原划分。预测使用M1的三分类logits取`argmax`与回归头连续输出。解释目标是原预测类别相对最强竞争类别的logit margin和原强度输出；三模态层计算8个保留组合的精确Shapley遮蔽敏感性，局部层使用长度1、3、5的连续窗口遮蔽。解释并非因果贡献。

## 复现秒级证据与最终宽表

本步骤还需要原始视频，以及问题1环境中的FFmpeg、MFA声学模型、Whisper和公开BERT。将`Q1_RESOURCE_ROOT`设为这些资源所在工作目录的绝对路径；该目录结构应与`问题1/environment/README.md`一致，包含`.envs/q1/`、`.envs/mfa/`、`q1_work/mfa_data/`和`模型/`。问题1核心代码已在相邻的`问题1/code/`中。

```powershell
$env:Q1_RESOURCE_ROOT = '此处填写问题1工具与模型所在目录'
$q1py = Join-Path $env:Q1_RESOURCE_ROOT '.envs\q1\python.exe'
& $q1py -B .\src\run_qg_hta_attachment4.py
& $q1py -B .\src\merge_qg_hta_evidence.py
& $q1py -B .\src\audit_qg_hta.py
& $py -B .\src\build_explanation_cards_and_summary.py
& $py -B .\src\export_submission_csv.py
```

正式结果表中未获得可靠时间的证据保留空值；关键帧来自原视频。提交表内的关键帧路径为相对路径。历史`results/`中也包含相应中间文件，可在不重新提取视频的情况下核对行数、位置和数值。
