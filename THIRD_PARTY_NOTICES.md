# 第三方代码与资源

- **EMT-DLFR**：上游仓库 https://github.com/sunlicai/EMT-DLFR ，原工程登记提交为`e4216215565292360e9c3fc43c8cd6e127df19c2`。代码与许可证保留于`问题2/resources/third_party/EMT-DLFR/`及问题3冻结接口的原模型实现中。任务适配代码不等于原论文实验的完全复现。
- **BERT**：`google-bert/bert-base-uncased`，固定revision为`86b5e0934494bd15c9632b12f734a8a67f723594`。配置、词表、来源清单与许可证按原件保留，初始化权重不入Git。
- **Whisper、OpenSMILE、OpenFace、MFA、FFmpeg**：作为外部工具依赖使用。安装、版本和下载方式见`问题1/environment/`，二进制、模型和已安装环境未装入仓库。
- **官方竞赛附件**：作为原始输入使用，未重新分发原视频、原始PKL或标签工作簿。仓库保留自主生成特征、预测与解释结果，复现者需自行取得官方输入。

第三方内容遵守其随附许可证。项目原创代码与论文暂未指定统一开源许可证；私有仓库归档不授予额外公开再分发权。
