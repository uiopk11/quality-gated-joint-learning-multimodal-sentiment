# 问题1提交材料

- `问题1_100条多模态时序特征.pkl.gz`：100条样本统一压缩特征文件。
- `code/`：QG-HTA质量审计、分层对齐、特征生成、汇总与审计代码。
- `environment/`：Python/MFA环境、版本锁定、自检与公开模型下载脚本。
- `audit/`：样本清单、模态清单、特征字典、完整性审计和典型样本图。

读取已提交的100条特征：

```powershell
python .\读取提交特征.py
```

`code/read_features_example.py`用于读取完整处理流程产生的单样本`features.npz`，不是上述统一归档的读取入口。

若需从官方附件1重新提取特征，先安装`environment/`记录的Python、MFA、FFmpeg、OpenFace与公开模型，再把`code/config.yaml`里的每个`<PROJECT_ROOT>`替换为本机工作目录的绝对路径。`data_root`和`label_file`须指向官方原始附件1；`output_root`应设为新的空目录。进入`code/`后依次运行`python step1_quality_audit.py`、`python calibrate_alignment_thresholds.py`、`python step2_generate_features.py`、`python audit_outputs.py`和`python build_combined_features.py`。官方原始数据不在提交包中。

提交特征副本仅将内部开发机绝对路径替换为`<PROJECT_ROOT>`，不改变数值特征。校验结果见`audit/submission_feature_audit.json`。该占位符仅存在于追溯元数据和复现配置中，重新运行前需要替换。
