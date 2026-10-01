# MelodyMap：城市音乐街区与夜间消费活力探索

本项目以墨尔本的音乐场所和行人传感器为案例，探索音乐场所空间分布与行人活动之间的关系，并通过合成城市实验考察传感器覆盖率、分析半径、坐标误差、城市中心性混杂和已知效应强度对分析结果的影响。

## 阅读入口

- [研究报告（PDF）](Report_MelodyCity.pdf)
- [主要分析笔记本](melodymap-project/MelodyMap.ipynb)
- [案例数据来源说明](melodymap-project/data/case_study/PROVENANCE.md)
- [已保存图表](melodymap-project/outputs/figures/)

## 项目结构

```text
Report_MelodyCity.pdf             研究报告
melodymap-project/
├── MelodyMap.ipynb               真实案例与合成实验的主要分析入口
├── requirements.txt             Python 依赖
├── melodymap/                   数据处理、空间分析、模拟和可视化代码
├── data/case_study/              小规模真实案例及来源说明
├── data/demo/                    合成演示数据
└── outputs/                     已保存图表及演示报告等产物
```

## 本地运行

准备 Python 3.10 或更高版本，在本仓库目录下执行：

```sh
cd melodymap-project
python -m pip install -r requirements.txt
python -m jupyter lab MelodyMap.ipynb
```

在 Jupyter 中按顺序运行笔记本单元格。笔记本使用仓库中附带的案例数据和本地生成的合成数据；安装好依赖后，这条分析路径不需要下载完整的在线数据集。新生成的笔记本结果保存到 `melodymap-project/outputs/notebook/`。

另有合成演示数据流程，可在 `melodymap-project` 目录下运行：

```sh
python -c "from melodymap.pipeline import run_pipeline; run_pipeline(mode='demo')"
```

该流程会重新生成演示数据，并更新本地处理结果与图表。

## 数据与结果说明

真实案例包含三个音乐场所、三个行人传感器，以及长期平均行人流量参考值；具体出处见[来源说明](melodymap-project/data/case_study/PROVENANCE.md)。这些参考值不代表 2024 年夜间平均值。

`data/demo/` 中的数据和合成城市实验均用于方法演示与敏感性分析，不能当作墨尔本的真实观测结果。空间相关性和行人流量也不能直接证明音乐场所对消费或经济活动的因果影响。

仓库保留报告、笔记本、源代码、案例数据、合成演示数据、已有处理结果和图表。下载的原始 CSV、新生成的笔记本输出，以及系统和 Python 缓存由忽略规则排除。
