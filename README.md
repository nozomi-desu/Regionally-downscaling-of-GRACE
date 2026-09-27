# Regionally adaptive soft-constrained downscaling of GRACE

本仓库整理文章 **Regionally adaptive soft-constrained downscaling of GRACE terrestrial water storage anomalies** 使用的 RWSC 代码与处理后数据。提供数据驱动 α/β 的五组实验、冻结权重、数值验证源数据和 USGS 同一 3° 网格内的细网格差分验证。原始下载文件、论文文档、图像成品、旧实验、假设性结果和运行日志未纳入仓库。

## 模型与实验范围

| 文章模型 | 原项目目录 | α | β |
| --- | --- | --- | --- |
| M0 | M0 | 0 | 0 |
| M1 | M1 | 全局固定均值 | 全局固定均值 |
| M2 | M2 | 空间自适应 | 全局固定均值 |
| M3 | M3 | 全局固定均值 | 空间自适应 |
| M4 | M5 | 空间自适应 | 空间自适应 |

文章 M4 对应原项目 **M5**，保留原目录名以维持冻结记录的可追溯性。α、β 的固定面积加权均值分别为 0.4623363684、0.1685511373；权重仅用训练期计算，训练前固定。所有模型使用 13 输入通道的 residual U-Net、seed 42、120 epochs、batch size 4，输出单位为 mm EWH。214 个有效观测月份按训练/验证/测试分为 156/22/36；测试期为 2020–2022，缺失月份不插补。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `scripts/preprocess/` | 网格、距平、辅助属性、训练数据与数据驱动 α/β 的构建代码 |
| `scripts/train/`、`utils/mass_closure.py` | 数据读取、U-Net、损失、训练与质量闭合算子 |
| `scripts/eval/` | 推理、内部/外部验证和 USGS 网格内差分验证 |
| `data_driven_alpha_beta/M*/config.yaml` | 五组原始冻结配置 |
| `data_driven_alpha_beta/weights/` | 冻结 α/β、来源覆盖、权重诊断和生成元数据 |
| `validation_rebuild/tables/` | 粗尺度、谐波、SMAP、地下水、频谱、区域和训练轨迹的处理后 CSV |
| `validation_rebuild/cache/` | 处理后空间数组、共同掩膜、bootstrap 分布和权重重建数据 |
| `validation_rebuild/audit/` | 原实验的月份划分、配置/预测/数据/代码 SHA-256 与统计定义 |
| `validation_extension_20260908/groundwater/` | QC 后月度 USGS 观测及 QC 摘要 |
| `data_processed/full_model_smap_test_smap_monthly_overlap.nc` | 处理后 2020–2022 SMAP 月度观测，约 35.6 MiB |
| `results/usgs_intra3deg_validation/` | 处理后井/网格序列、模型井位置序列、细网格配对和粗网格汇总 |
| `tests/`、`tools/` | 原有科学单元测试、数值核对和发布文件校验 |

`provenance/published_files.csv` 列出所有发布文件的大小、SHA-256 和类别。`validation_rebuild/audit/` 中的历史通过记录描述原实验；当前发布包的完整性由 `tools/verify_publication.py` 单独核对。历史清单中的绝对路径是来源标识，不代表 GitHub 仓库内已有这些文件。

## 未上传的大型数据

以下大型处理后数据没有上传。详细相对路径与字节大小见 [`provenance/omitted_large_data.csv`](provenance/omitted_large_data.csv)。这些数据没有公开下载链接，不应把本仓库描述为无需补充数据即可完整重训。

| 数据 | 大小 | 对复现的影响 |
| --- | --- | --- |
| `grace_downscaling_training_dataset_mm_area.zarr` | 178.6 MiB，共 700 个文件 | 完整训练与全局评价需要；已提供逐文件哈希及月份划分 |
| 处理后 GLDAS-Noah TWSA | 212.6 MiB | 从头重建 β 需要 |
| 处理后 CLM5 TWSA | 152.3 MiB | 从头重建 β 需要 |
| 五组全局 `inference/test.nc` | 每组 190.9 MiB | 从全局预测重算评价需要；井位置预测已提供小型导出 |
| 五组 `best.pt` | 合计约 112.0 MiB | 直接推理需要；冻结检查点 SHA-256 已提供 |
| MLR/RF 的全局预测 | 未随本次发布复制 | 原重算脚本的共同掩膜还包含这两个统计基线 |

上述文件保存在原实验工作区；获取后按表中相对路径放入仓库，并与原清单核对。原始 JPL、WGHM、ERA5、GLDAS、CLM5、SMAP、USGS、冰川、人类活动与流域下载文件也未上传。独立地下水评价还读取 WGHM 原始地下水分量；USGS 脚本若重跑原始 QC，需要原始 USGS 下载页。处理后的 QC 月度表及差分验证结果已纳入。

## 环境与运行

使用 Python 3.11；原评价环境版本见 `validation_rebuild/audit/environment.json`。`requirements.txt` 是运行依赖范围，未声称是原环境的完整锁文件。训练需要匹配硬件的 PyTorch/CUDA 环境。

```bash
python -m pip install -r requirements.txt
python tools/verify_publication.py
python tools/verify_intra3deg_results.py results/usgs_intra3deg_validation
python validation_rebuild/scripts/test_evaluation_core.py
python -m unittest tests.test_data_driven_alpha_beta tests.test_mass_projection tests.test_model_matrix tests.test_usgs_intra3deg
```

补齐上述大型训练数据后，可生成适合当前机器的配置再训练。配置生成只转换路径，保留原模型、损失、随机种子与训练参数：

```bash
python tools/prepare_paper_configs.py
python scripts/train/train.py --config configs/paper_local/M4.yaml --device cuda
```

原始冻结配置保持不变。`validation_rebuild/scripts/` 的六处执行脚本只将机器绝对根目录转换为文件相对根目录，计算逻辑保持不变；逐文件原始哈希见 `provenance/publication_changes.csv`。重新执行全局评价前须补齐全局预测、训练集和相关分量；权重重建须补齐 GLDAS/CLM5，并指定新的输出目录，避免覆盖冻结权重。

```bash
python scripts/preprocess/build_data_driven_alpha_beta.py --output-dir validation_rebuild/cache/rebuilt_weights_local
```

## 文章数值与源数据的差异

本次整理未修改实验数值。提供的文章中 M4 粗尺度 MAE 在摘要为 **1.6742 mm**，正文及 Table 2 为 **1.384 mm**；现有冻结重算表 `validation_rebuild/tables/coarse_full_metrics.csv` 中 M4 corrected 的值为 **1.67974 mm**。Table 2 的 M4 半年振幅 MAE 为 **0.135 mm**，重算表中为 **0.152859 mm**。这些差异尚未由相应版本的预测与评价记录解释，引用时应核对所采用版本，不把文章中的“最低 MAE”当作本数据包已验证的结论。

评价同时保留正负结果。JPL 一致性不代表独立 0.5° 真值，SMAP/井水位相关也不是总储量幅度精度。α/β 的统计诊断与外部关联须按其来源和有效观测覆盖解释。
