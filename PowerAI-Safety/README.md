# PowerAI-Safety

**基于多模态AI的电力设备智能安全检测与风险评估系统**

Multimodal AI-based Intelligent Safety Inspection and Risk Assessment System for Power Equipment

---

本系统面向电力设备安全巡检，融合**可见光图像、红外热成像与电气时序数据**三个模态，
结合电力行业标准规则库与检索增强生成（RAG），自动完成设备缺陷检测、异常分析、
风险评估与结构化巡检报告生成。

系统遵循技术方案的核心设计原则：

> **底层检测由专有模型完成，阈值与风险等级由配置文件决定，大模型只负责解释、归因与建议。**

---

## 目录

- [一、当前实现状态](#一当前实现状态)
- [二、系统架构](#二系统架构)
- [三、目录结构](#三目录结构)
- [四、快速开始](#四快速开始)
- [五、核心检测流程](#五核心检测流程)
- [六、三个检测分支](#六三个检测分支)
- [七、规则引擎](#七规则引擎)
- [八、多模态融合与风险分级](#八多模态融合与风险分级)
- [九、RAG 知识库](#九rag-知识库)
- [十、大模型层](#十大模型层)
- [十一、HTTP 接口](#十一http-接口)
- [十二、配置说明](#十二配置说明)
- [十三、核心设计约束](#十三核心设计约束)
- [十四、已验证能力](#十四已验证能力)
- [十五、已知限制与后续工作](#十五已知限制与后续工作)
- [十六、许可证注意事项](#十六许可证注意事项)

---

## 一、当前实现状态

后端检测链路已完整打通并可端到端运行；前端页面、知识库文档与训练脚本尚未落地。

| 模块 | 状态 | 说明 |
| --- | :---: | --- |
| `backend/vision/` 可见光检测 | ✅ 已完成 | YOLO 后端 + 启发式兜底双实现 |
| `backend/vision/infrared_detector.py` 红外检测 | ✅ 已完成 | 三种温度解析路径，无需训练权重 |
| `backend/timeseries/` 电气时序 | ✅ 已完成 | Transformer 自编码器 + 统计判据双路径 |
| `backend/risk/` 规则引擎 | ✅ 已完成 | 18 条规则，配置驱动 |
| `backend/fusion/` 多模态融合 | ✅ 已完成 | 结果级融合，含权重重归一化 |
| `backend/rag/` 知识库检索 | ✅ 已完成 | 默认离线 TF-IDF，索引为空时安全降级 |
| `backend/llm/` 大模型层 | ✅ 已完成 | mock / dashscope / openai 三种提供方 |
| `backend/services/` 编排与存储 | ✅ 已完成 | SQLite 持久化 + Markdown 报告 |
| `backend/main.py` FastAPI 服务 | ✅ 已完成 | 10 个接口 |
| `datasets/synthesize.py` 合成样本 | ✅ 已完成 | 5 个场景，零下载即可跑通全链路 |
| `scripts/run_demo.py` 命令行演示 | ✅ 已完成 | 不启服务验证整条链路 |
| `models/` 模型权重 | ⬜ 空 | 两个视觉分支均运行在降级模式 |
| `frontend/` Web 前端 | ⬜ 未创建 | 访问 `/` 返回 404，可改用 `/docs` 调试接口 |
| `knowledge/` 标准文档 | ⬜ 未创建 | 知识库为空，报告「依据标准」章节留空 |
| `training/` 训练脚本 | ⬜ 未创建 | 报告中的训练提示指向 `training\train_yolo.py` 等尚未落地的路径 |
| `tests/` 测试 | ⬜ 未创建 | — |

> **关于降级模式**：缺少模型权重不会导致系统报错，只会让对应分支切换为降级实现，
> 并在结果、报告与接口响应中**显式标注** `backend` 字段与警告信息。
> 详见 [十三、核心设计约束](#十三核心设计约束)。

---

## 二、系统架构

技术方案定义的五层架构与代码模块的对应关系：

```text
┌──────────────────────────────────────────────────────────┐
│ 应用层     backend/main.py (FastAPI)                      │
│            /api/inspect  /api/inspections  /api/stats     │
├──────────────────────────────────────────────────────────┤
│ 知识分析层  backend/rag/    → 标准条款检索                 │
│            backend/llm/    → 解释、归因、处置建议          │
│            backend/risk/   → 规则判定 + 风险分级           │
├──────────────────────────────────────────────────────────┤
│ 融合层     backend/fusion/risk_fusion.py                  │
│            加权求和 + 权重重归一化 + 一致性加成            │
├───────────────┬──────────────────┬───────────────────────┤
│ 可见光分支     │ 红外分支          │ 电气时序分支           │
│ vision/       │ vision/          │ timeseries/           │
│ yolo_detector │ infrared_detector │ anomaly_detector      │
│ YOLO / 启发式  │ 辐射/灰度/伪彩     │ Transformer / 统计     │
├───────────────┴──────────────────┴───────────────────────┤
│ 数据层     data/samples/  data/uploads/  data/reports/    │
│            data/powerai.db (SQLite)   knowledge/          │
└──────────────────────────────────────────────────────────┘
```

编排入口是 `backend/services/inspection_service.py::InspectionService.inspect()`，
它按「三分支 → 规则 → 融合 → RAG → 大模型 → 报告 → 落盘」的顺序串联全流程。

---

## 三、目录结构

```text
PowerAI-Safety/
├── backend/
│   ├── main.py                    FastAPI 入口与全部路由
│   ├── config.py                  配置加载（点号访问 + 路径解析）
│   ├── core/schemas.py            跨模块数据契约（dataclass 领域模型）
│   ├── vision/
│   │   ├── yolo_detector.py       可见光检测：YOLO + 启发式兜底
│   │   ├── infrared_detector.py   红外测温、热点提取与热缺陷判级
│   │   ├── image_processor.py     图像 IO/绘制（中文路径与中文标签兼容）
│   │   ├── colormaps.py           红外伪彩色调色板生成与反演
│   │   └── labels.py              中英文类别名双向映射
│   ├── timeseries/
│   │   ├── signal_loader.py       CSV/Excel/JSON 读取与通道名归一化
│   │   ├── preprocessing.py       重采样、去噪、归一化、滑窗
│   │   ├── stft.py                FFT / STFT / 谐波分析
│   │   ├── transformer_model.py   Transformer 自编码器与持久化
│   │   └── anomaly_detector.py    业务指标 + 模型/统计双路评分
│   ├── risk/
│   │   ├── rule_engine.py         配置驱动规则求值
│   │   └── risk_level.py          分数 → 风险等级映射
│   ├── fusion/risk_fusion.py      多模态结果级融合
│   ├── rag/
│   │   ├── document_loader.py     PDF/DOCX/TXT/MD 解析与切分
│   │   ├── embedding.py           向量化（TF-IDF / sentence-transformers）
│   │   └── retriever.py           索引构建、持久化与余弦检索
│   ├── llm/
│   │   ├── base.py                提供方接口与图像编码
│   │   ├── mock_provider.py       离线模板生成（默认）
│   │   ├── api_provider.py        OpenAI 兼容客户端（含 Dashscope/Qwen-VL）
│   │   ├── factory.py             提供方选择与自动回退
│   │   └── report_generator.py    结构化事实 + Markdown 报告渲染
│   └── services/
│       ├── inspection_service.py  主流程编排
│       └── database.py            SQLite 持久化
├── config/
│   ├── config.yaml                全部阈值、权重、路径的唯一事实来源
│   └── rules.yaml                 18 条风险规则
├── datasets/synthesize.py         合成样本生成器
├── scripts/run_demo.py            命令行端到端演示
├── models/                        模型权重存放目录（当前为空）
├── data/
│   ├── samples/                   合成样本 + manifest.json（含真值）
│   ├── uploads/                   API 上传文件落盘位置
│   ├── reports/                   报告归档（report.md / report.json / annotated/）
│   ├── index/                     知识库索引
│   └── powerai.db                 SQLite 数据库
└── requirements.txt
```

---

## 四、快速开始

### 4.1 环境

本项目在项目内 `.venv` 中运行，该虚拟环境以 `--system-site-packages` 方式创建，
从 Anaconda base 继承 `torch` / `numpy` / `pandas` / `scikit-learn` / `matplotlib`，
无需重复下载（torch 约 2GB）。

已验证环境：**Python 3.10.16 (Anaconda)**，Windows 11。

```powershell
cd PowerAI-Safety

# 首次配置（若 .venv 已存在可跳过）
python -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

> **为什么需要 `--system-site-packages`**：base 环境已自带 PyTorch 等大型依赖，
> 继承可避免重复安装。注意 base 中的 `opencv-python 4.7` 按 numpy 1.x 编译，
> 在 numpy 2.x 下无法导入，因此 `requirements.txt` 在 venv 内单独安装了
> `opencv-python>=4.10` 覆盖该版本——这一步**不可省略**。

### 4.2 生成合成样本（可选）

`data/samples/` 中已包含生成好的样本，如需重新生成：

```powershell
.venv\Scripts\python datasets\synthesize.py                  # 默认输出到 data/samples
.venv\Scripts\python datasets\synthesize.py --seed 7         # 更换随机种子
.venv\Scripts\python datasets\synthesize.py --no-pseudo-color
```

生成 5 个场景，每个场景包含可见光 JPEG、16 位辐射 PNG、可选伪彩色图与其标定文件、
时序 CSV，并把真值写入 `data/samples/manifest.json`。

| 场景 | 说明 | 三相温度 (℃) | 预期热级别 | 时序形态 |
| --- | --- | --- | --- | --- |
| `normal` | 正常运行 | 38.0 / 39.2 / 40.1 | normal | 正常日负荷曲线 |
| `oil_leak` | 渗漏油 + 油污 | 44.0 / 45.5 / 46.8 | normal | 正常 |
| `overheat` | 接线端子过热 | 52.0 / 55.0 / **96.5** | critical | 负荷上升 |
| `critical` | 多模态一致异常 | 58.0 / 61.0 / **118.0** | critical | 电流冲击 |
| `foreign_object` | 悬挂异物 | 36.5 / 37.4 / 38.2 | normal | 正常 |

> ⚠️ 合成数据仅用于**打通流程与回归测试**，不能用于评估模型精度。
> 真实数据应来自 CPLID / IDDD / thermal-images-equip / ETT 等公开数据集。

### 4.3 命令行跑通全链路（推荐首先执行）

不需要启动服务，直接验证整条链路：

```powershell
.venv\Scripts\python scripts\run_demo.py                       # 全部 5 个场景
.venv\Scripts\python scripts\run_demo.py --scenario overheat   # 单个场景
.venv\Scripts\python scripts\run_demo.py --markdown            # 附带完整报告
```

输出示例（`overheat` 场景）：

```text
场景 overheat —— 接线端子过热（红外严重，可见光无异常）
  真值       : 可见光缺陷=无  相温=52.0, 55.0, 96.5℃  预期热级别=critical  时序形态=load_rise
  可见光分支 : backend=heuristic_fallback  VisualScore=0.0  检出=['transformer(0.38)', 'transformer(0.31)']
  红外分支   : backend=radiometric  Tmax=97.26℃  环境=25.0℃  参考T2=52.72℃(three_phase)
               ΔT=44.54K  δ=61.64%  级别=critical  ThermalScore=100.0
  时序分支   : backend=statistical  异常窗口=1个  占比=35.6%  ElectricalScore=35.65
  规则命中   : ['相对温差严重(90.00)', '绝对温度超限(75.00)', ...]  →  RuleScore=90.0
  融合结果   : RiskScore=79.47  等级=异常
               红外 100.0×0.50=50.0 + 时序 35.6×0.29=10.2 + 规则 90.0×0.21=19.3 = 79.5（异常）
```

注意其中两点：可见光分支因无训练权重被识别为**不具备判断能力**而非「判定为正常」，
其权重被剔除并重新分配（红外 0.50 / 时序 0.29 / 规则 0.21，而非配置的 0.35 / 0.20 / 0.15）；
真值 `critical` 在此例中融合为「异常」，因为可见光模态缺失——这正是设计预期。

### 4.4 启动 HTTP 服务

```powershell
.venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
```

- 接口文档： <http://127.0.0.1:8000/docs>
- 系统状态： <http://127.0.0.1:8000/api/health>
- 前端首页： <http://127.0.0.1:8000/>（`frontend/index.html` 尚未创建，当前返回 404）

---

## 五、核心检测流程

`InspectionService.inspect()` 的七个阶段（对应技术方案第十四节）：

```text
上传可见光/红外图像 + 时序 CSV + 三相温度（可选）
        │
        ├─ 1. 可见光分支 ─→ 目标检测 → VisualScore
        ├─ 2. 红外分支   ─→ 温度场 → 热点提取 → 判级 → ThermalScore
        └─ 3. 时序分支   ─→ 滑窗 → 业务指标 + 异常检测 → ElectricalScore
        │
        ├─ 4. 规则引擎   ─→ 规则命中 → RuleScore
        ├─ 5. 多模态融合 ─→ RiskScore → 风险等级
        ├─ 6. RAG 检索   ─→ 标准条款依据
        ├─ 7. 大模型     ─→ 综合分析 + 处置建议 → Markdown 报告
        │
        └─ 落盘：data/reports/<ID>/{report.md, report.json}
                 + SQLite data/powerai.db
```

**容错原则**：任一分支失败只记入 `report.warnings`，**不中断整体流程**，也不返回 500。
传感器 CSV 格式异常时，系统仍会用图像模态给出结论，并在报告中明确标注「时序数据不可用」——
而不是把缺失模态当作「正常」。仅当所有模态都处理失败时才抛出异常。

---

## 六、三个检测分支

### 6.1 可见光分支 `backend/vision/yolo_detector.py`

双后端设计，由 `create_visible_detector()` 依配置自动选择：

| 后端 | `backend` 字段 | 触发条件 | 缺陷识别能力 |
| --- | --- | --- | --- |
| YOLO | `yolo` | 权重文件存在且 ultralytics 可用 | ✅ 完整 |
| 启发式兜底 | `heuristic_fallback` | 权重缺失 / ultralytics 未安装 / 加载失败 | ❌ 仅定位显著区域 |

**检测类别**（`config.yaml` 中的 `device_classes` / `defect_classes`）：
设备类 `transformer`、`bushing`、`terminal`、`conservator`、`radiator`；
缺陷类 `oil_leak`、`rust`、`deformation`、`foreign_object`、`broken_part`、`oil_stain`。

**视觉异常分**只统计缺陷类目标：

```text
base  = 100 × max(严重度权重 × 置信度)
bonus = min(15, 5 × (缺陷数 − 1))
score = min(100, base + bonus)
```

各类缺陷的严重度权重在 `config.yaml` 的 `defect_severity` 中配置
（如 `broken_part: 0.90`、`rust: 0.45`）。

**启发式兜底**（CLAHE → 双边滤波 → Canny → 形态学闭运算 → 轮廓）只输出设备类结果，
`visual_score` 恒为 `0.0`，并在 `notes` 中说明自身局限。

### 6.2 红外分支 `backend/vision/infrared_detector.py`

**温度解析三路径**（结果中的 `backend` 字段）：

| 路径 | 输入 | 说明 |
| --- | --- | --- |
| `radiometric` | 16 位灰度 | DN 满量程 (0/65535) 直接映射到 `[temp_min_c, temp_max_c]` |
| `gray_linear` | 8 位灰度 | 按配置区间线性映射 |
| `pseudo_color_palette` | 伪彩色 | 调色板反演 + 色彩距离作为可信度信号 |

同时也支持读取同名 `.calib.json` 标定文件（如 `xxx.jpg.calib.json`）。

**参考温度 T2** 的取值优先级（`_reference_temperature`）：

```text
three_phase  → 三相温度中的最低相（最可靠）
warm_median  → 较热一半像素的中位数
ambient      → 环境温度
```

> 判级用 **ΔT = T1 − T2**（发热点与**正常相对应点**之差），而不是「最高温 − 环境温度」。
> 后者会把盛夏中正常运行的设备判成异常。全图中位数同样不可用——冷背景会把中位数拉低，
> 从而系统性放大 ΔT。相对温差按 **DL/T 664** 计算：`δ = (T1 − T2) / (T1 − T0) × 100%`。

**判级**同时采用绝对温度法与相对温差法，取二者中**更严重**的一级。
热点提取阈值取 `max(95 百分位, 基线 + 0.5 × ΔT告警阈值)`，后一项用于防止在低对比度图像中
切出虚假热点。热风险分从 `severity_score`（normal 5 / warning 45 / alarm 75 / critical 95）
取值，并按**达到告警级以上的热点数**小幅加分（`min(10, 3 × (异常热点数 − 1))`）——
只数异常热点是因为三相设备本来就有三个接线端子，全部计入会虚增分数。

**三相温度**可由调用方通过 `three_phase_temps` 参数直接传入（**现场实测值远比图像估算可靠，
强烈建议传入**）；未传入时按「面积最大的三个同尺度区域，按水平中心排序」启发式估算，
并在三个区域面积比小于 0.15 时放弃估算。

### 6.3 电气时序分支 `backend/timeseries/`

```text
CSV/Excel/JSON
   ↓  signal_loader：通道名归一化（支持中文/英文/拼音列名）
   ↓  preprocessing：重采样 → 裁剪极端值 → 去噪 → 归一化 → 滑动窗口
   ↓
   ├─ 业务可解释指标（两条路径都会计算）
   └─ 异常评分
        ├─ transformer：自编码器重构误差（需训练权重）
        └─ statistical：统计判据（无权重时的默认路径）
```

**通道**：`voltage`、`current`、`power`、`temperature`、`load`、`frequency`，
以及三相分相通道 `voltage_a/b/c`、`current_a/b/c`。

**业务指标**（`BusinessMetrics`）：`load_rise_ratio`（近 1/4 时段相对前 1/2 基线的负荷变化）、
`voltage_deviation_ratio`、`current_imbalance_ratio`（GB/T 15543）、`max_load_ratio`、
`load_trend_slope`、`temp_rise_ratio` 等。这些指标**同时供规则引擎使用**，即使走模型路径也会计算。

**统计判据的关键设计**：不对 z-score 峰值用固定阈值。n 个标准正态样本的最大 |z| 期望约为
`sqrt(2·ln(2n))`——n=96 时就有 3.2，n=1000 时约 3.9。固定用「峰值 > 3」会把大量正常白噪声序列
判成异常。因此代码用 **z 峰值 ÷ 纯噪声预期值**（`expected_max_z`）作为判据，
默认 `zscore_factor_warning = 1.15`、`zscore_factor_critical = 2.0`。

**评分取各指标中的最大值而非加权求和**——负荷阶跃会同时抬高 z-score 与电流不平衡度，
相加会造成同一根因重复计分。

**Transformer 自编码器**（`transformer_model.py`）：仅在正常工况窗口上训练，
编码器把 `(T, C)` 压成单一隐向量，解码器重构整个窗口，以重构 MSE 为异常分，
阈值取验证集指定的分位数。之所以用自编码器而非分类器，是因为现场故障样本稀缺且形态未知，
无监督方法不需要故障标签。验证集从序列**末尾**切分且不打乱，避免时序泄漏。
权重以 `model.pt`（仅 `state_dict`）+ `meta.json`（结构超参、阈值、归一化器、训练指标）两个文件保存——
归一化统计必须与模型一同持久化，否则训练/推理尺度不一致会产生虚假告警。

---

## 七、规则引擎

`backend/risk/rule_engine.py` 对 `config/rules.yaml` 中的规则求值，**新增设备类型只需追加规则，
无需改动 Python 代码**。当前共 **18 条规则**：

| 类别 | 数量 | 示例 |
| --- | :---: | --- |
| 红外 / 温度 | 7 | 绝对温度达严重异常（≥110℃）、相对温差严重（ΔT≥40K）、三相温度不平衡 |
| 可见光 / 外观 | 4 | 渗漏油、部件破损、结构变形、悬挂异物 |
| 电气时序 | 5 | 负荷持续升高、负载率偏高、电压偏差超限、三相电流不平衡 |
| 多模态一致性 | 2 | 多模态一致指向过热、视觉与红外位置吻合 |

**条件类型**：`threshold`（阈值比较）、`contains`（列表成员，中英文标签均可匹配）、
`any_of`、`all_of`（可递归嵌套）。

**RuleScore 取命中规则中的最大严重度，而非求和**——多条规则通常指向同一根因
（例如绝对温度与相对温差会同时命中），求和会系统性抬高分数。
每条命中记录引用字段的实际取值，便于人工复核。

规则依据的行业标准：DL/T 664（带电设备红外诊断应用规范）、DL/T 572（电力变压器运行规程）、
DL/T 741（架空输电线路运行规程）、GB/T 12325（供电电压偏差）、GB/T 15543（三相电压不平衡）。

---

## 八、多模态融合与风险分级

`backend/fusion/risk_fusion.py` 实现技术方案第六节的结果级融合公式：

```text
RiskScore = w1 × VisualScore + w2 × ThermalScore
          + w3 × ElectricalScore + w4 × RuleScore
```

默认权重 `visual 0.30 / thermal 0.35 / electrical 0.20 / rule 0.15`（配置中权重不必和为 1，代码会归一化）。

三项关键处理：

1. **权重重归一化** — 现场往往只有部分模态。缺失模态的权重被剔除并在剩余模态间按比例重分配，
   否则缺失模态会把总分系统性拉低，造成**漏报**。
2. **规则权重按需参与** — 规则分只在**有规则命中**时才计入权重。
   「未命中」意味着「未发现配置中定义的异常模式」，而不是「风险为 0」，
   不应参与加权稀释其它模态的分数。
3. **一致性加成** — 三个及以上模态同时 ≥ 60 分时加 6 分（幅度可配置），
   多模态互相印证的结论可信度高于单一模态。

**风险分级**（`config.yaml` 的 `fusion.levels`，闭区间下界、开区间上界，最高档含 100）：

| 分数 | 等级 | 代码 | 颜色 |
| --- | --- | --- | --- |
| 0–30 | 正常 | `normal` | 绿 |
| 30–60 | 关注 | `attention` | 黄 |
| 60–80 | 异常 | `abnormal` | 橙 |
| 80–100 | 严重异常 | `critical` | 红 |

`docs` 与报告中会给出**完整算式**，例如
`红外 100.0×0.50=50.0 + 时序 35.6×0.29=10.2 + 规则 90.0×0.21=19.3 = 79.5（异常）`。

---

## 九、RAG 知识库

```text
knowledge/*.pdf|.docx|.txt|.md
   ↓  document_loader：解析 → 分块（默认 480 字符，重叠 80）
   ↓  embedding：向量化
   ↓  retriever：建立索引并持久化到 data/index/
   ↓
检测结果 → compose_retrieval_query → 余弦检索 top-k → 报告「依据标准」章节
```

**默认使用离线 TF-IDF**（`analyzer="char_wb"`，2–3 字符 n-gram），无需下载模型即可运行。
`requirements.txt` 中 `sentence-transformers` 与 `faiss-cpu` 为可选依赖：
配置 `rag.embedding.backend: "sentence_transformers"` 可切换语义向量，
依赖缺失时**整体回退**到 TF-IDF（不做部分降级，避免向量维度混用）。

检索用 numpy 全量矩阵乘法。数百到数千个文本块的场景下耗时 < 1ms，
因此**刻意不引入 FAISS**；如需替换，改动点集中在 `KnowledgeBase._load_index` 与 `search`。

索引以 `chunks.jsonl` + `vectors.npy` + `embedder.pkl` + `meta.json` 四个文件持久化，
`meta.json` 记录知识库目录的 SHA-256 签名（`路径:大小:修改时间`），内容变化时自动重建。

**当前知识库为空**——`knowledge/` 目录尚未创建，索引签名为 `empty`，
报告第四章会显示「知识库中未检索到相关条款」并给出提示。
放入文档后调用 `POST /api/knowledge/rebuild` 重建即可。

---

## 十、大模型层

大模型**不承担底层检测**，只负责三件事：多模态结果解释、归因分析、处置建议。
风险等级与阈值始终由 `risk_fusion` 与 `rule_engine` 决定。

| 提供方 | `provider` | 说明 |
| --- | --- | --- |
| 模板生成 | `mock` | **默认**。完全离线、确定性输出，不虚构数字，只复述真实字段 |
| Dashscope | `dashscope` | Qwen-VL 系列，OpenAI 兼容接口 |
| OpenAI | `openai` | `gpt-4o-mini` 等 |

**API Key 只从环境变量读取**，不写入配置文件：

```powershell
$env:DASHSCOPE_API_KEY = "sk-xxxx"
$env:OPENAI_API_KEY = "sk-xxxx"
```

Key 缺失、网络异常、`requests` 未安装等情况一律**自动回退到 `MockProvider`** 并记录警告，
不会中断流程。调用失败时返回的文本带醒目的 `【大模型调用失败】` 标记——
这是为了防止「模型调用失败」被误读成「模型未发现风险」。

`MockProvider` 按 `【检测结果】/【综合分析】/【判定依据】/【处置建议】` 四个章节组织输出，
归因逻辑基于真实字段门控（如 `thermal_score ≥ 50`、`load_rise_ratio ≥ 0.15`、
`visual_thermal_iou > 0.05` 同时成立时，才给出「载流回路接触电阻增大」的结论）。
它不是假实现，而是真实模型的对照基线。

`ReportGenerator` 的结构化事实（`build_facts`）同时喂给模板生成与 API 提示词，
保证两条路径输入完全一致。Markdown 报告包含五个章节，并在附录中列出**本次实际使用的判定阈值**
与融合权重，确保报告可追溯。

---

## 十一、HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 前端页面（`frontend/` 未创建时返回 404 提示） |
| `GET` | `/api/health` | 系统状态：各分支后端、模型加载情况、知识库、数据库 |
| `GET` | `/api/providers` | 大模型提供方可用性 |
| `POST` | `/api/inspect` | 执行一次多模态巡检 |
| `GET` | `/api/inspections` | 巡检历史列表（支持 `limit` / `offset` / `risk_level` 过滤） |
| `GET` | `/api/inspections/{id}` | 单次巡检完整结果 |
| `GET` | `/api/inspections/{id}/report` | 导出 Markdown 报告 |
| `DELETE` | `/api/inspections/{id}` | 删除记录 |
| `GET` | `/api/stats` | 风险统计 |
| `POST` | `/api/knowledge/rebuild` | 重建知识库索引 |
| `GET` | `/files/...` | 标注图等静态资源 |

### `POST /api/inspect`

`multipart/form-data`，三个文件字段**至少提供一个**：

| 字段 | 类型 | 必填 | 说明 |
| --- | :---: | :---: | --- |
| `visible_image` | File | 三选一 | 可见光图像（jpg/png/bmp/tif/webp） |
| `thermal_image` | File | 三选一 | 红外热像图 |
| `timeseries_csv` | File | 三选一 | 时序数据（csv/xlsx/xls/json） |
| `device_name` | Form | — | 设备名称 |
| `location` | Form | — | 检测位置 |
| `operator` | Form | — | 检测人 |
| `three_phase_temps` | Form | — | 三相温度实测值，逗号分隔，如 `62.1,64.3,71.8` |

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/inspect `
  -F "visible_image=@data/samples/overheat_visible.jpg" `
  -F "thermal_image=@data/samples/overheat_thermal.png" `
  -F "timeseries_csv=@data/samples/overheat_timeseries.csv" `
  -F "device_name=1号主变" `
  -F "three_phase_temps=52.0,55.0,96.5"
```

---

## 十二、配置说明

**所有阈值、权重、分级边界只能来自 `config/*.yaml`**，大模型在任何情况下都不得改写这些数值，
只能引用和解释。配置文件是系统唯一的事实来源。

### `config/config.yaml` 主要配置段

| 配置段 | 内容 |
| --- | --- |
| `app` | 应用名称、版本、数据/上传/报告目录、数据库路径 |
| `device` | 当前聚焦设备（MVP 阶段为 `transformer`） |
| `models` | 模型权重目录 |
| `vision.visible` | YOLO 权重路径、置信度/IoU 阈值、类别列表、缺陷严重度权重、兜底参数 |
| `vision.infrared` | 温度映射区间、环境温度、绝对温度阈值、ΔT 阈值、三相不平衡阈值、热点提取参数 |
| `timeseries` | 采样间隔、窗口大小、步长、去噪/归一化参数、额定容量与额定电压、模型超参、统计判据阈值 |
| `fusion` | 四项融合权重、四档风险分级边界与配色、一致性加成 |
| `rag` | 知识库目录、分块参数、top-k、embedding 后端 |
| `llm` | 提供方选择、超时、token 上限、温度、各提供方模型与 Key 环境变量名 |
| `report` | 报告标题、是否附插图/原始分数、免责声明 |

配置读取支持点号访问，缺失键会抛出带完整路径的 `AttributeError`，便于定位拼写错误：

```python
from backend.config import get_config
cfg = get_config()
print(cfg.fusion.weights.thermal)        # 0.35
print(cfg.path("vision", "visible", "weights"))   # 解析为绝对路径
```

> **红外温度区间务必按实际相机设置修改** `temp_min_c` / `temp_max_c`，
> 否则绝对温度判断无意义。伪彩色与无标定的灰度图只能得到**相对**温度分布。
> 合成样本的 `TEMP_MIN_C` / `TEMP_MAX_C` 常量也需与配置保持同步。

---

## 十三、核心设计约束

以下约束是代码中反复强调的不变量，修改代码时不应破坏：

1. **阈值归配置，判断归规则，解释归大模型。**
   `risk_level.py`、`llm/base.py`、`llm/factory.py` 三处都独立声明了这一边界。

2. **「无能力判断」≠「判断为正常」。**
   可见光兜底检测器无法识别缺陷，其 `visual_score` 恒为 0。这个 0 表示*没有能力判断*，
   而非*判断为正常*，因此按**模态缺失**处理（`VisibleResult.score_available`），
   权重被剔除并重归一化。若让它参与融合，会以「视觉 0 分」的名义稀释其它模态，
   使系统即使红外和时序都指向严重也到不了「严重异常」。

3. **模态缺失必须显式标注原因。**
   `FusionResult.missing_reasons` 区分「未提供数据」与「提供了但该分支不具备判断能力」，
   防止读者把「权重被剔除」误读成「该模态判定为正常」。

4. **分数的合成取最大而非求和。**
   规则引擎与统计判据都遵循这一原则，避免同一根因重复计分。

5. **降级必须可见。**
   每个分支都在结果的 `backend` 字段中标注自己运行的模式，
   并在 `notes` / `warnings` 中说明局限。绝不用降级结果伪装成完整功能。

6. **相对量优于绝对量。**
   红外用 ΔT（对正常相对应点）而非绝对温升；统计判据用 z 峰值 / 噪声预期而非固定 z 阈值。

7. **归一化统计必须与模型一同持久化。**
   否则训练/推理尺度不一致会产生虚假告警。

8. **任何可选依赖缺失都只降级、不崩溃。**
   `pymupdf`、`python-docx`、`jieba`、`sentence-transformers`、`requests`、API Key 缺失时
   均回退到可用实现并记录日志。

---

## 十四、已验证能力

`scripts/run_demo.py` 对 5 个合成场景端到端跑通，方向一致性符合真值：

- `overheat`：三相温度 52.0/55.0/96.5℃ → 识别参考温度 T2 = 52.72℃（`three_phase`），
  ΔT = 44.54K，δ = 61.64%，判为 `critical`，命中 7 条规则，RuleScore 90，融合为「异常」。
- `critical`：可见光检出 `broken_part` + `rust`，红外 118℃，时序 `current_spike`，
  三模态一致 → 触发一致性加成。
- `normal`：各分支均无异常，融合为「正常」。
- 缺模态场景：可见光权重被正确剔除并重归一化，报告附「权重已按比例重新分配」说明。
- 知识库为空时不报错，报告给出明确提示而非静默留空。

红外分支无需任何训练权重即可工作（`radiometric` 路径）。
若尚无 YOLO 权重，可先用 `--scenario` 单独验证红外与时序分支。

---

## 十五、已知限制与后续工作

### 当前限制

- **无模型权重**：`models/` 为空，可见光分支运行在启发式兜底模式，时序分支运行在统计判据模式。
  两者都**不具备真实的缺陷识别/模式学习能力**，当前输出仅供链路验证。
- **无前端页面**：`frontend/` 尚未创建，需通过 `/docs` 或 curl 调试接口。
- **知识库为空**：`knowledge/` 尚未创建，报告「依据标准」章节始终为空。
- **训练脚本缺失**：报告注释中给出的 `training\train_yolo.py`、`training\train_timeseries.py`
  路径尚未实现；`datasets/synthesize.py` 文档中提到的 `prepare_*.py` 数据准备脚本同样尚未落地。
- **无自动化测试**：`tests/` 尚未创建。
- **合成数据不可用于精度评估**：仅用于流程验证与回归。
- **红外绝对温度依赖标定**：伪彩色与无标定灰度图只能得到相对温度分布；
  真实相机需提供 16 位辐射数据或 `.calib.json` 标定文件。

### 后续工作

按技术方案的实施路线，优先级建议如下：

1. **接入真实数据** — CPLID / IDDD（可见光绝缘子）、thermal-images-equip（红外）、ETT（时序），
   补齐 `datasets/prepare_*.py`。
2. **训练模型** — 补齐 `training/train_yolo.py` 与 `training/train_timeseries.py`，
   使两个分支脱离降级模式。
3. **构建知识库** — 向 `knowledge/` 放入 DL/T 664、DL/T 572、GB/T 12325 等标准文档，
   让「依据标准」章节具备实际内容。
4. **Web 前端** — 实现 `frontend/index.html`：图片上传、检测结果展示、历史记录、
   风险统计、报告导出。
5. **验证真实 VLM** — 配置 `DASHSCOPE_API_KEY` 切换到 Qwen-VL，与 `MockProvider` 输出对比评估。
6. **补充测试** — 为规则引擎、融合重归一化、温度解析、统计判据等关键逻辑建立单元测试。
7. **扩展设备类型** — 从变压器扩展到绝缘子、断路器、避雷器，
   只需扩充 `vision.visible.*_classes` 与 `config/rules.yaml`。

---

## 十六、许可证注意事项

**代码许可证与数据集许可证是两回事**，需分别核对。

- **Ultralytics (YOLO)** 采用 **AGPL-3.0**，对使用与再发布存在开源义务。
  论文/科研验证可使用开源实现；**正式商业部署前须重新核对许可条件**，
  或改用其它检测框架、取得商业许可。
- **数据集**需分别查看数据来源、引用要求与再分发限制。
  例如 `thermal-images-equip` 要求使用数据时引用原作者，
  ETT、CPLID、IDDD 等亦各有其引用与许可条款。
- 本项目依赖的其它库（FastAPI、OpenCV、PyMuPDF、PyYAML 等）各自采用宽松许可证，
  但仍建议在正式发布前统一核查。

```text
论文 / 科研验证
      ↓  可使用开源实现进行实验
正式商业项目
      ↓  重新检查全部依赖许可证
      ↓  确认模型权重与数据集的许可
      ↓  必要时更换模型或取得商业许可
```

---

## 参考

- 项目总览（背景需求 / 技术逻辑 / 最终效果）：[`../README.md`](../README.md)
- 交互式接口文档：启动服务后访问 `/docs`
