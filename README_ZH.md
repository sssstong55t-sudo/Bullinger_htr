# Bullinger 项目交付版使用说明

当前版本：2026 年 10 月 14 日。

## 从哪里开始

本仓库的当前交付入口是 `outputs/bullinger_htr_delivery_EN/`。旧版本已移至 `legacy/`，只用于历史追溯。

- 官方正文地名识别：GLiNER 与官方历史地名变体词表，P 82.99%、R 61.16%、F1 70.42%。这是官方标注一致性，标注完整性尚未确认。
- 历史 500 行 HTR 测试：CER 13.3917%、WER 38.6088%。这是另一项实验，不是地名错误率，也不是图像到地名的端到端成绩。
- WER<37%、P≥85% 的严格目标尚未达到。

## 安装和运行

在仓库根目录使用 Python 3.10 或兼容环境：

```sh
python3 -m venv .venv-nlp
.venv-nlp/bin/python -m pip install -r requirements.txt
.venv-nlp/bin/python scripts/download_model.py
.venv-nlp/bin/python outputs/bullinger_htr_delivery_EN/scripts/run_official_ner.py --output-dir outputs/official_ner_run01
```

首次需要联网下载固定版本模型，之后 CPU 离线运行。输出目录必须尚不存在。这个命令评价固定官方正文，不识别新手稿图像。不同环境可能产生数值差异。

## Colab 训练

使用交付包中的 `Bullinger_Clean_Colab.ipynb`，先阅读 [训练说明](outputs/bullinger_htr_delivery_EN/TRAINING.md)。它依赖已有 Google Drive 数据与 checkpoint，不是上传笔记本即可从零复现。默认不选择任务，避免自动消耗算力。最好的 Colab checkpoint 未上传 GitHub。

## 提交材料

交付包包含代码、原始结果、官方参考数据和来源审核。原始大图像数据、模型权重、环境、个人审核表不在仓库内。完整数据和模型应按说明另行取得。所有方案选择依据验证集；不得根据测试结果反复调参。

[交付包中文说明](outputs/bullinger_htr_delivery_EN/README_ZH.md) · [结果细节](outputs/bullinger_htr_delivery_EN/RESULTS.md) · [英文首页](README.md)
