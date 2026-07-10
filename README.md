# NuWa-Assimilation

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

NuWa-Assimilation是一个基于人工智能的气象资料同化与预报框架，由国防科大气象海洋学院数值预报创新团队开发。

## 目录

- [项目概述](#项目概述)
- [模型架构](#模型架构)
- [环境配置](#环境配置)
- [数据集说明](#数据集说明)
- [使用方法](#使用方法)
- [引用信息](#引用信息)

## 项目概述

支持多种深度学习气象预报模型（ClimaX、AFNONet、ViT、FuXi），具备资料同化能力和完整的评估框架。

## 模型架构

| 模型 | 架构类型 |
|------|----------|
| ClimaX | 视觉Transformer |
| AFNONet | 傅里叶神经算子 |
| ViT | 视觉Transformer |
| FuXi | Swin Transformer |

## 环境配置

```bash
conda create -n nuwa python=3.9
conda activate nuwa
pip install -r requirements.txt
pip install -e .
```

## 数据集说明

使用ERA5再分析数据集，69个气象变量，64×128空间分辨率，6小时间隔，HDF5格式存储。

## 使用方法

```bash
# 训练模型
python src/train.py model=climax trainer=gpu

# 评估模型
python src/eval.py model=climax ckpt_path=/path/to/checkpoint.ckpt
```

## 引用信息

```bibtex
@article{climax2023,
  title={ClimaX: A Foundation Model for Weather and Climate},
  author={Prasad, Gautam and others},
  journal={arXiv preprint arXiv:2301.10343},
  year={2023}
}
```

## 许可证

MIT License