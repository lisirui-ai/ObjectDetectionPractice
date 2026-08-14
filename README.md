

# ObjectDetectionPractice



基于 PyTorch + HuggingFace 的目标检测模型实战系列

以 **Transformer 端到端检测 → 可变形注意力加速 → 单阶段实时检测** 为主线，循序渐进覆盖目标检测核心技术栈

每个 Notebook 均配有详细中文注释，适合目标检测入门与进阶学习



---



## 📋 目录

- [项目概览](#-项目概览)
- [目录结构](#-目录结构)
- [Notebook 介绍](#-notebook-介绍)
- [学习路径](#-学习路径)
- [环境依赖](#️-环境依赖)
- [快速开始](#-快速开始)

---



## 🗺 项目概览


| #   | Notebook                            | 核心技术                                                                               | 亮点                                                                     |
| --- | ----------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 01  | DETR 模型架构实现（HuggingFace）            | DETR · ResNet-50 骨干 · Transformer Encoder-Decoder · Object Queries · 二分图匹配 · 匈牙利算法 | 从零手写 DETR 完整架构，HuggingFace 官方 API 推理 + 边界框可视化                          |
| 02  | Deformable DETR 模型架构实现（HuggingFace） | Deformable Attention · 多尺度特征图 · 参考点偏移 · grid_sample · 稀疏采样 · 快速收敛                  | 经典注意力 vs 可变形注意力对比实现 + grid_sample 原理演示 + HuggingFace 推理                |
| 03  | YOLOv5s 模型架构解析（TorchHub）            | YOLOv5s · CSP-Darknet · C3 模块 · SPPF · FPN + PAN · Detect 头 · NMS · torch.hub      | torch.hub 加载预训练模型，逐层解析 Backbone / Neck / Head 完整结构，结合架构图对照，含推理与检测结果可视化 |


---



## 📁 目录结构

```
ObjectDetectionPractice/
├── 📓 01.detr_ModelArchitecture_HuggingFace.ipynb
├── 📓 02.DeformableDetr_ModelArchitecture_HuggingFace.ipynb
├── 📓 03.YOLOv5s_ModelArchitecture_TorchHub.ipynb
│
├── 📂 images/                             # 实验用图及可视化输出
│   ├── street.jpg                         # 推理测试图（Notebook 03）
│   ├── yolov5_arch.png                    # YOLOv5s 架构示意图（Notebook 03）
│   ├── layer_map.png                      # 层号与架构图速查图（Notebook 03）
│   └── results/                           # 检测结果保存目录（Notebook 03）
│       └── street.jpg                     # [自动生成] YOLOv5s 推理结果图
│
├── 📂 model_cache/                        # 模型权重缓存目录（gitignored）
│   ├── detr-resnet-50/                    # [自动生成] DETR 权重缓存（Notebook 01）
│   ├── deformable-detr/                   # [自动生成] Deformable DETR 权重缓存（Notebook 02）
│   └── ultralytics_yolov5_master/        # [自动生成] YOLOv5 仓库缓存（Notebook 03）
│
├── 🤖 yolov5s.pt                         # YOLOv5s 预训练权重（Notebook 03）（gitignored）
├── 📄 requirements.txt                    # 项目依赖清单
├── 📄 .gitignore
├── 📄 LICENSE
└── 📄 README.md
```

---



## 📚 Notebook 介绍



### 01. DETR 模型架构实现

> `01.detr_ModelArchitecture_HuggingFace.ipynb`

从零手写 **DETR（DEtection TRansformer）** 完整架构，将目标检测建模为集合预测问题，以 ResNet-50 提取空间特征后展平为序列，送入标准 Transformer Encoder-Decoder；Decoder 侧的 100 个可学习 Object Queries 并行解码出边界框与类别，训练时以**匈牙利算法**（二分图最优匹配）为每个查询分配唯一真实目标，彻底摆脱 Anchor 设计与 NMS 后处理。同步演示 HuggingFace 官方 API 推理，并进行检测结果边界框可视化。


| 章节                      | 内容                                                                            |
| ----------------------- | ----------------------------------------------------------------------------- |
| 环境准备与骨干网络预览             | ResNet-50 结构查看 · 末尾两层（AvgPool + FC）去除原因说明                                     |
| DETR 模型架构定义与推理测试        | CNN 骨干 · 1×1 投影层 · 位置编码 · Transformer · 分类头 + 回归头 · Object Query 各层次 Shape 说明 |
| 特征图尺寸辅助计算               | 不同输入分辨率下特征图展平序列长度推导（800×1200 → 25×38=950 Token）                               |
| HuggingFace 官方 API 推理实验 | 加载 `facebook/detr-resnet-50` · 图像预处理 · 推理 · 后处理 · 边界框 + 类别标签可视化               |


> **DETR 核心原理** · CNN 骨干（去掉最后两层的 ResNet-50）提取 `[B,2048,H/32,W/32]` 特征图，经 1×1 投影压缩至 256 维后展平为长度 `H/32×W/32` 的 Token 序列，加入 2D 正弦位置编码后送入 Transformer Encoder；Decoder 以 100 个可学习 Object Queries 为初始输入，通过交叉注意力逐步查询 Encoder 输出，最终各 Query 输出分别送入 FFN 预测类别和边界框（cx, cy, w, h），训练时以匈牙利算法强制一对一匹配。
>
> 参考论文：*End-to-End Object Detection with Transformers*（Carion et al., Facebook AI Research, ECCV 2020）

---



### 02. Deformable DETR 模型架构实现

> `02.DeformableDetr_ModelArchitecture_HuggingFace.ipynb`

针对 DETR 收敛慢（需 500 epoch）、小目标检测弱两大痛点，引入**可变形注意力机制**：每个 Query 仅对参考点附近稀疏 K 个位置采样，计算复杂度从 O((HW)²) 降至 O(HW·K)，结合**多尺度特征图**（P3/P4/P5/P6），训练轮次压缩至 **50 epoch**，小目标性能显著提升。并行实现经典注意力与可变形注意力，通过热力图直观对比两者的注意力分布差异，同时演示 `grid_sample` 可微分采样原理。


| 章节                               | 内容                                                                  |
| -------------------------------- | ------------------------------------------------------------------- |
| 经典注意力 vs 可变形注意力对比                | 全局注意力热力图 vs 稀疏采样权重热力图 · 复杂度公式对比 · 收敛速度分析                            |
| grid_sample 函数演示                 | 3×3 特征图 + 2×2 采样网格 · 归一化坐标系 [-1,1] 说明 · 双线性插值原理                     |
| 注意力图可视化                          | 稀疏采样权重展开为密集 8×8 矩阵 · 与经典注意力热力图并排对比                                  |
| 可变形注意力完整实现（V2）                   | 偏移量预测线性层 · 注意力权重预测 · grid_sample 采样 · 加权求和输出                        |
| HuggingFace Deformable DETR 推理实验 | 安装 `timm` 依赖 · 加载 `SenseTime/deformable-detr` · 图像预处理 · 推理 · 边界框可视化 |


> **可变形注意力核心设计** · 每个 Query 预测 K 个采样偏移量（相对参考点），通过 `grid_sample` 在特征图上双线性插值得到采样特征，再以预测的注意力权重加权求和；多尺度版本同时在 P3/P4/P5/P6 四个尺度采样并融合，无论特征图分辨率多大，每次前向传播的注意力计算量固定为 O(HW·M·K)（M 为头数，K 为采样点数，默认 4）。
>
> 参考论文：*Deformable DETR: Deformable Transformers for End-to-End Object Detection*（Zhu et al., SenseTime & CUHK, ICLR 2021）

---



### 03. YOLOv5s 模型架构解析

> `03.YOLOv5s_ModelArchitecture_TorchHub.ipynb`

通过 `torch.hub` 加载 **YOLOv5s** 并逐层解析 Backbone / Neck / Head 完整架构，结合官方架构图对照每一层的模块类型、空间尺寸变化与跳连关系。Backbone（CSP-Darknet，层 0–9）以 C3 残差模块 + SPPF 提取多尺度特征；Neck（FPN + PAN，层 10–23）自顶向下融合高层语义、自底向上融合定位细节；Head（Detect，层 24）在 P3/P4/P5 三个尺度各输出 `3×(5+80)=255` 通道预测，最终经 NMS 得到检测框。


| 章节                   | 内容                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------- |
| 模型环境准备与加载            | `torch.hub.load` 加载 `ultralytics/yolov5` · 权重缓存重定向到 `model_cache/` · AutoShape 封装说明 |
| 整体架构概览               | Input / Backbone / Neck / Head 四区块划分 · 层索引与架构图对应关系速查表 · 跳连来源标注                      |
| Backbone 主干网络（层 0–9） | Conv(k=6,s=2,p=2) → P1/2 · C3(n=1/2/3) 残差模块 · P3/P4/P5 跳连输出 · SPPF(k=5) 空间金字塔池化     |
| Neck 特征融合网络（层 10–23） | FPN 自顶向下（Upsample + Concat + C3）· PAN 自底向上（Conv 下采样 + Concat + C3）· 双路径特征融合         |
| Head 检测头（层 24）       | Detect 层三尺度（P3/8·P4/16·P5/32） · 每头 1×1 Conv 输出 255 通道 · `3×(5+80)` 解释               |
| 推理示例                 | AutoShape 端到端推理 · letterbox 缩放 · NMS 过滤 · 检测结果打印与保存                                 |


> **YOLOv5s 架构要点** · C3 模块为简化版 CSP（跨阶段局部网络），将主干拆为两路再合并，在减少参数量的同时保持梯度流畅；SPPF 以三个串行 MaxPool(k=5) 替代原始 SPP 的三个并行 MaxPool，速度提升约 2×，感受野等价；Detect 头在每个尺度预测 3 个 anchor 框，`(tx,ty,tw,th)` 相对 anchor 的偏移量经 sigmoid 解码后还原为绝对坐标，`objectness` 置信度与类别概率联合过滤。
>
> 参考仓库：[ultralytics/yolov5](https://github.com/ultralytics/yolov5)（Glenn Jocher, Ultralytics, 2020）

---



## 🛤 学习路径

```
Transformer 端到端检测层
  01 DETR（ResNet-50 骨干 + Transformer Encoder-Decoder + Object Queries + 匈牙利匹配）
       ↓
       问题：收敛慢（500 epoch）· 小目标弱 · 全局注意力复杂度高
       ↓
  02 Deformable DETR（可变形注意力 + 多尺度特征图）
     ├── 可变形注意力：稀疏采样 K 点，复杂度 O(HW·K)
     ├── 多尺度特征：P3 / P4 / P5 / P6 四尺度融合
     └── 收敛加速：500 epoch → 50 epoch
       ↓
单阶段实时检测层
  03 YOLOv5s（CSP-Darknet Backbone + FPN+PAN Neck + Detect Head）
     ├── Backbone (层 0–9)：C3 残差 + SPPF 空间金字塔
     ├── Neck (层 10–23)：FPN 自顶向下 + PAN 自底向上
     └── Head (层 24)：P3/P4/P5 三尺度 Detect，NMS 后处理
```

建议按编号顺序学习：先通过 DETR 理解「将检测建模为集合预测」的核心思想（01），再通过 Deformable DETR 深入可变形注意力与多尺度融合机制，掌握加速收敛的关键改进（02），最后学习工业界广泛部署的单阶段检测器 YOLOv5s，从逐层架构解析到实时推理全链路贯通（03）。**01 → 02 建议连续学习：先吃透 DETR 端到端思路，再对比可变形注意力带来的改进，效果最佳。**

---



## ⚙️ 环境依赖

> **说明**：所有 Notebook 共用同一套环境，按需安装对应模块依赖即可。

**核心依赖** · Python 3.14.5


| 包名             | 版本           | 用途                                                       |
| -------------- | ------------ | -------------------------------------------------------- |
| `torch`        | 2.12.0+cu132 | 深度学习框架核心（CUDA 13.2 加速）                                   |
| `torchvision`  | 0.27.0+cu132 | 图像预处理与视觉数据集工具（ResNet-50 等骨干网络）                           |
| `transformers` | 5.14.1       | HuggingFace 模型加载与推理（DETR · Deformable DETR）              |
| `timm`         | 1.0.28       | PyTorch Image Models，Deformable DETR 骨干网络依赖（Notebook 02） |
| `ultralytics`  | 8.4.118      | YOLOv5 / YOLOv8 推理与训练框架（Notebook 03）                     |
| `torchinfo`    | 1.8.0        | 模型结构摘要与参数统计（`summary()` 逐层展示输入输出形状）                      |
| `torchviz`     | 0.0.3        | 计算图可视化（生成 Graphviz 格式的前向图，辅助理解模型结构）                      |
| `matplotlib`   | 3.10.9       | 图像展示、边界框绘制与注意力热力图可视化                                     |
| `seaborn`      | 0.13.2       | 统计图表（注意力热力图对比，Notebook 02）                               |
| `numpy`        | 2.4.6        | 数值计算基础库                                                  |
| `pillow`       | 12.2.0       | 图像读取、格式转换与预处理                                            |
| `requests`     | 2.34.2       | HTTP 请求，用于从 URL 下载测试图像（Notebook 01 / 02）                 |


---



## 🚀 快速开始

**1. 克隆仓库**

```bash
git clone git@github.com:lisirui-ai/ObjectDetectionPractice.git
cd ObjectDetectionPractice
```

**2. 安装依赖**

```bash
# torch / torchvision 的 CUDA wheel 托管于 PyTorch 官方源，需追加 --extra-index-url
# 其余包从 PyPI 正常安装，requirements.txt 已固定所有版本，保证环境可复现
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
```

**3. 准备模型权重**

所有 Notebook 均通过 HuggingFace Hub 或 `torch.hub` 在首次运行时**自动下载**并缓存至 `model_cache/` 对应子目录，无需手动下载。

- **Notebook 01**：自动下载 `facebook/detr-resnet-50` 缓存至 `model_cache/detr-resnet-50/`
- **Notebook 02**：自动下载 `SenseTime/deformable-detr` 缓存至 `model_cache/deformable-detr/`
- **Notebook 03**：通过 `torch.hub` 自动下载 `ultralytics/yolov5` 仓库代码至 `model_cache/ultralytics_yolov5_master/`，并使用本地 `yolov5s.pt` 权重文件

**4. 启动 Jupyter**

```bash
jupyter notebook
```

按编号顺序打开 Notebook 开始学习即可。

---



## 📄 License

[MIT](LICENSE) © 2026