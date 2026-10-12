# 问题1运行环境

问题1使用三部分环境：

- `.envs/q1`：Python 3.11、PyTorch、Transformers、OpenSMILE、FFmpeg和数据处理依赖；
- `.envs/mfa`：Montreal Forced Aligner，单独隔离以避免与主环境冲突；
- `工具/OpenFace_2.2.0_win_x64`：OpenFace官方Windows二进制。
- `模型/bert-base-uncased`：问题1使用的本地BERT权重与分词器；
- `模型/whisper-small`：用于人声、语言与官方文本一致性核验的本地Whisper权重；ASR结果不替换官方文本；
- `q1_work/mfa_data`：MFA的`english_us_arpa`声学模型和发音词典。该路径保持纯英文，以避开MFA/Kaldi在Windows下对中文路径的编码问题。

本机已验证配置：Python 3.11.16、PyTorch 2.14.0+cu130、CUDA 13.0、Transformers 4.57.6、OpenSMILE 2.6.0、FFmpeg 7.1.1、MFA 3.3.10、OpenFace 2.2.0。

## 一键安装

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\环境\setup_q1.ps1
```

默认安装RTX 4060可用的CUDA 13.0版PyTorch。若需在无NVIDIA显卡的电脑上重建，增加`-CpuOnly`。

只安装主环境时：

```powershell
powershell -ExecutionPolicy Bypass -File .\环境\setup_q1.ps1 -SkipMfa -SkipOpenFace
```

## 使用主环境

无需激活也可以直接运行：

```powershell
.\.envs\q1\python.exe .\环境\check_q1_environment.py
```

重新下载并验证BERT模型：

```powershell
.\.envs\q1\python.exe .\环境\download_bert_model.py
```

重新下载Whisper-small模型：

```powershell
.\.envs\q1\python.exe .\环境\download_whisper_model.py
```

如需激活：

```powershell
conda activate .\.envs\q1
```

## 强制对齐模型

MFA程序与语言模型分开安装。本项目已经下载`english_us_arpa`声学模型和配套词典。运行时固定项目内模型目录：

```powershell
$env:MFA_ROOT_DIR = "$PWD\q1_work\mfa_data"
conda run --prefix .\.envs\mfa mfa version
```

## OpenFace

批量视频特征提取使用：

```powershell
.\工具\OpenFace_2.2.0_win_x64\FeatureExtraction.exe -f <视频路径> -out_dir <输出目录> -aus -pose -gaze
```

正式全量处理前，必须先在少量样本上核对CSV字段、时间戳、检测置信度和人脸跟踪成功率。

## 已完成的冒烟测试

- BERT在RTX 4060上输出`(1, 6, 768)`；
- Whisper-small已在附件1正常、静音、环境声和文本错配样本上完成辅助识别与质量门控；
- OpenSMILE对附件1样本音频成功输出固定74维逐帧特征；
- OpenFace对附件1真实视频成功生成CSV；
- MFA命令、英文声学模型和词典均可读取。

OpenFace已固定选择35个字段。上述测试证明工具链和三个分层对齐分支可以运行；全量正式特征仍须在人工复核门控阈值后重新生成和审计。

## 锁定实际版本

安装成功后保存真实环境快照：

```powershell
conda list --prefix .\.envs\q1 --explicit > .\环境\q1-conda-explicit.txt
.\.envs\q1\python.exe -m pip freeze > .\环境\q1-pip-freeze.txt
conda list --prefix .\.envs\mfa --explicit > .\环境\mfa-conda-explicit.txt
```
