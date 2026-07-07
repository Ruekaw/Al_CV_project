# 铝晶粒显微图像晶粒闭合与尺寸统计工程方案

## 一、目标与当前状态

**目标**：输入不规则铝晶粒显微图，输出闭合晶界图 + 基于 ASTM E112 截线法的平均晶粒尺寸（L_bar / 等效直径 d / 晶粒度 G）。

**实验室当前要求**：先做晶粒闭合看效果，试样状态细节后续再细化。因此本方案以"出干净闭合晶界图"为一期核心交付，尺寸统计为二期。试样状态不作为开工前置条件。

**技术路线**：路线 A（边界空间）。MatSAM 给候选边界，本工程在其上加一层边界清洗（删支晶/噪点段 + 补断口）得到干净晶界图，再喂截线法。不依赖 region 空间的 AR/circularity，不做 Watershed 实例分离（降为可选扩展）。

## 二、数据

**位置**：`pretrain_sets/`（已就位，31 张）

**规格**：统一 `2736×2196` RGB JPG，文件名 `{编号}晶.jpg`，按编号分四组：
- `1-1晶.jpg`（1 张）
- `10xxx` 组（13 张）：10198–10230
- `20xxx` 组（12 张）：20133–20155
- `30xxx` 组（5 张）：30464–30507

统一分辨率意味着形态学核与长度阈值无需逐图重标定，只需与 `um_per_pixel` 绑定。

**标注状态**：未打标。采用两阶段验证（见第六章），一期不依赖 GT。

**比例尺**：`um_per_pixel` 暂未确定，需从图内比例尺检测或实验室提供。一期可先输出像素单位结果，二期补物理单位。

## 三、总体流程

```
原图 (pretrain_sets/*.jpg)
  ↓ 3.1 预处理 + 尺度标定
增强图 + um_per_pixel
  ↓ 3.2 候选边界生成（MatSAM 主 / 传统法 baseline）
候选边界二值细线图
  ↓ 3.3 边界段切分与特征提取
边界段集合 {seg_i}（长度/端点/方向/曲率/孤立距离）
  ↓ 3.4 支晶/噪点边界剔除（删短段、删孤立段、删不闭合段）
清洗后边界图（被删段红色覆盖，供复核）
  ↓ 3.5 晶界断口修复（端点方向一致性对接）
干净闭合晶界图（补连段蓝色高亮）   ← 一期交付核心
  ↓ 3.7 ASTM E112 多方向截线法
L_bar / d / G + 截线可视化报告    ← 二期交付
```

3.6 Watershed 实例分离为**可选扩展**，需要逐晶粒面积/晶粒度分布时再启用，不进入主流程。

## 四、模块设计

### 4.1 预处理与尺度标定

**输入**：原图（BGR）
**操作**：
1. BGR → 灰度；
2. CLAHE（clipLimit=2.0, grid=(8,8)）增强晶界对比度；
3. 中值滤波（ksize=3）去椒盐；
4. 双边滤波（d=5）保边平滑。

**尺度标定**（输出物理单位的前提，优先级从高到低）：
1. 实验室提供 `um_per_pixel`（最准）；
2. 图内比例尺检测：检测比例尺线段 + OCR/模板识别数值；
3. 配置文件人工填写。

一期若三者都暂缺，`um_per_pixel=None`，所有长度类输出以像素为单位，并在结果中标注"px（未标定）"。

**输出**：增强灰度图 + `um_per_pixel`

### 4.2 候选边界生成

**主路径：MatSAM**
1. PromptGenerator 生成点 prompt（Canny 粗分割质心 + 网格点，`method_type=1`）；
2. SamAutomaticMaskGenerator 生成候选 mask；
3. 按 `area_threshold` / 灰度阈值筛 mask；
4. 对每个 mask Laplacian 取边界；
5. 骨架化 → 膨胀(square=5) → 腐蚀(square=3) → 再骨架化 → 膨胀(square=2)，连成细线。

**baseline：传统法**（对照 + GPU 不可用时兜底）
```
灰度 → CLAHE → 双边 → 自适应阈值/OTSU → Canny → 骨架化 → 形态学清洗 → 候选边界图
```

**输出**：二值细线边界图（两条路径同构，便于对比）

**实现要点**：
- notebook 中 `area_threshold=25000`、`min_size=200` 等参数按论文数据集调优，本方案全部从 YAML 读取；
- 开发期可用 `vit_b` 提速迭代，交付/评估用 `vit_h`。

### 4.3 边界段切分与特征提取

**输入**：候选边界二值细线图
**操作**：
1. `skimage.morphology.skeletonize` 保证单像素宽；
2. `cv2.connectedComponents` 连通域分析 → 边界段集合 `{seg_i}`；
3. 每段提取：
   - 长度 `L_i`（像素数）；
   - 端点坐标 `(p_i1, p_i2)`；
   - 端点处切线方向 `d_i1, d_i2`（取端点附近 N 个骨架点拟合）；
   - 平均曲率 / 方向一致性；
   - 到最近长段距离 `D_i`。

**输出**：`List[Segment]`，供 4.4 / 4.5 使用

### 4.4 支晶/噪点边界剔除（核心）

**输入**：`{seg_i}`
**判定规则**（阈值与 `um_per_pixel` 绑定，无标定时用像素默认值，标 *待首跑调参*）：

1. **噪点段**：`L_i < L_min`（默认 15px*）→ 删；
2. **孤立短线**：`D_i > D_iso`（默认 40px*）且 `L_i < L_short`（默认 80px*）→ 删；
3. **支晶臂段**：长度中等、方向一致性高（曲率低，接近直线）、**无法参与任何闭合环** → 删。
   - 闭合检测二选一：
     - **形态学法（推荐，简单）**：边界图闭运算 + 填孔得填实区域，未落在任何填实区域边界上的段判为不闭合；
     - **图论法**：以端点为节点、近邻端点为边建图，找最小环，不在任何环上的段判为不闭合。

4. 保留段做轻量清洗（去毛刺、平滑），不过度腐蚀。

**可视化**：被删段在原图上红色覆盖，便于实验室复核。

**输出**：清洗后边界图 + 删除段记录

> 关键点：边界空间看"参不参与闭合"，而非 region 空间的"形状细不细长"。这与截线法目标天然对齐。

### 4.5 晶界断口修复

**输入**：清洗后边界图 + 端点列表
**操作**：
```
对每个端点 p：
  在半径 R_connect（默认 35px*）内搜索候选端点 q
  检查 p→q 方向与 p、q 处切线方向的夹角是否 < angle_thresh（默认 30°*）
  若一致 → 用直线（或沿局部梯度方向的曲线）连接 p、q
```
修复后再骨架化保证单像素宽。

**可视化**：补连段蓝色高亮。

**输出**：干净闭合晶界图（**一期交付核心**）

### 4.6（可选）粘连晶粒实例分离

需要逐晶粒面积/晶粒度分布时启用：
1. 干净晶界图反相 + 填孔 → 晶粒前景；
2. 距离变换 → 局部极大值为 marker；
3. Marker-controlled Watershed；
4. 面积阈值合并过分割。

主流程不依赖本模块。

### 4.7 ASTM E112 截线法统计（二期）

1. **多方向截线**：0°/45°/90°/135°（至少 0°/90°），避免择优取向偏置；
2. **统计阈值停止**：累计交点数达 `N_min`（默认 400）即停；
3. **比例尺换算**：
   $$\bar{L} = \frac{\sum_j L_j}{\sum_j P_j} \times \text{um\_per\_pixel} \quad (\mu m)$$
   `L_j` 为第 j 条线像素长度，`P_j` 为该线与晶界交点数；
4. **输出三件套**：
   - 平均线截距长度 `L_bar`（主指标）；
   - 等效晶粒直径 `d ≈ 1.128 · L_bar`；
   - ASTM 晶粒度号 `G`（按标准公式）。

**输出**：`L_bar` / d / G + 截线交点可视化（多方向不同色）

## 五、工程实现

### 5.1 依赖修复（开工第一件事）

`MatSAM/utils/metrics.py:13` 顶层 `import gala.evaluate as ev`，经 `utils/prompt_generator.py:3` 的 `from .metrics import PostPrecess` 传递性触发。gala 在 Windows 上装不上（C 扩展 + 旧 networkx），但仅用于评估指标 `Metric.get_vi()`，核心流程不需要。

**修复**：
1. 新建 `MatSAM/utils/postprocess.py`，把 `PostPrecess` 类整体迁入；
2. `prompt_generator.py` 改 `from .postprocess import PostPrecess`；
3. `metrics.py` 中 gala 改延迟导入（`get_vi` 内部 `import gala.evaluate as ev`）+ try/except 守护；
4. 无 gala 时 VI 指标降级跳过，仅输出 IoU/ARI/Dice。

不做这步，核心流程在 Windows 上第一行 import 就报错。

### 5.2 项目目录结构

```
CVproject/
├── pretrain_sets/              # 31 张原图（已就位）
├── MatSAM/                     # 上游依赖（修复 gala 后用）
├── plan.md                     # 本文件
├── config.yaml                 # 全部参数
├── src/
│   ├── preprocess.py           # 4.1
│   ├── candidate_boundary.py   # 4.2（MatSAM + baseline）
│   ├── segment_features.py     # 4.3
│   ├── boundary_clean.py       # 4.4
│   ├── boundary_repair.py      # 4.5
│   ├── watershed_optional.py   # 4.6（可选）
│   ├── intercept.py            # 4.7
│   ├── viz.py                  # 四联图/复核可视化
│   └── pipeline.py             # 串起 4.1→4.5（一期）/ →4.7（二期）
├── runs/                       # 输出：四联图、边界图、统计表
└── scripts/
    └── batch_run.py            # 批处理 pretrain_sets
```

### 5.3 配置（config.yaml）

```yaml
preprocess:
  clahe_clip_limit: 2.0
  clahe_grid: [8, 8]
  median_kernel: 3
  bilateral_d: 5

scale:
  um_per_pixel: null            # null=像素单位；填数值开物理单位

matsam:
  model_type: vit_h             # 开发期可改 vit_b
  checkpoint: MatSAM/checkpoints/sam_vit_h_4b8939.pth
  device: cuda                  # 无 GPU 改 cpu 并切 baseline
  layers: 0
  scales: 3
  n_per_side_base: 32
  method_type: 1
  pred_iou_thresh: 0.90
  stability_score_thresh: 0.92
  area_threshold: 25000         # 待首跑调参
  min_size: 200

boundary_clean:                 # 待首跑调参，与 um_per_pixel 绑定
  L_min: 15
  L_short: 80
  D_iso: 40
  closure_method: morphology    # morphology | graph

repair:                         # 待首跑调参
  R_connect: 35
  angle_thresh: 30

intercept:
  directions: [0, 45, 90, 135]
  N_min: 400
  primary_metric: L_bar
```

### 5.4 输入输出

**输入**：`pretrain_sets/*.jpg` + `config.yaml`
**输出**（`runs/{图名}/`）：
1. `01_preprocess.png` 预处理图；
2. `02_candidate_boundary.png` 候选边界图；
3. `03_cleaned_boundary.png` 剔除后边界图（被删段红色）；
4. `04_final_boundary.png` 干净闭合晶界图（补连段蓝色）；
5. `05_intercept.png` 截线可视化（二期）；
6. `result.json` 统计结果（L_bar/d/G + 各方向交点数）；
7. `report.png` 四联图合集（供实验室一眼复核）。

## 六、验证（两阶段）

### 近期：无 GT，肉眼 + 闭合率（一期主打）

无 GT 时靠以下自检 + 实验室人眼：
1. **晶粒闭合率**：边界图 → 闭运算 + 填孔 → 数填实区域个数；闭合得好区域数稳定且与肉眼晶粒数接近；
2. **四联图复核**：`01→02→03→04` 并排，实验室看"支晶删对了没、断口补对了没、晶粒闭合了没"；
3. **被删段/补连段可视化**：红色删段、蓝色补段叠在原图上，直接判断对错；
4. **MatSAM vs baseline 对比**：同图两条路径的 04 图并排，看增益。

从 `pretrain_sets` 挑 5–10 张代表性图（覆盖四组编号 + 不同视场密度）跑通端到端，参数按 config 默认值，据反馈调。

### 后期：小批量 GT，定量（二期）

流程稳定后标 10–20 张 GT（标"干净晶界图"，与 04 输出同构），算：
1. 晶界保留率（真晶界段未误删比例）；
2. 支晶剔除准确率（被删段中真支晶比例，需单独标支晶 mask）；
3. 断口修复正确率（补连段中真同晶界比例）；
4. `L_bar` 与人工截线法相对误差；
5. IoU/ARI（vs GT）；
6. 单图处理时间。

专门标难例比盲标一堆更省力。

## 七、风险与应对

1. **MatSAM 过/欠分割**：仅作候选，经 4.4/4.5 清洗；保留 baseline 兜底。
2. **支晶判定误删真晶界**：用"是否参与闭合"而非纯形状；删段红色可视化便于复核。
3. **断口修复误连**：方向 + 距离双约束；补连段蓝色可视化。
4. **参数对尺度敏感**：全部阈值与 `um_per_pixel` 绑定进 YAML，提供参数扫描。
5. **gala 在 Windows 装不上**：见 5.1，拆 `PostPrecess` + 延迟导入。
6. **无 GPU**：切传统 baseline，标注"CPU 模式"。
7. **`um_per_pixel` 未定**：一期输出像素单位并标注，二期补标定。
8. **"晶粒直径"物理含义**：输出明确为二维截面统计尺寸（L_bar/d/G），非三维真实直径，报告中注明。

## 八、落地阶段

**一期（晶粒闭合，实验室当前要求）**
- 阶段 1：依赖修复（5.1）+ 项目骨架（5.2）+ config.yaml + 4.1 预处理/尺度占位；
- 阶段 2：4.2 候选边界（MatSAM 跑通 + baseline）；
- 阶段 3：4.3 段切分 + 4.4 剔除（调 `L_min/D_iso/闭合检测`）；
- 阶段 4：4.5 断口修复（调 `R_connect/angle_thresh`）；
- 阶段 5：四联图 + 批处理脚本，交付 `pretrain_sets` 上的闭合晶界图，实验室肉眼复核。

**二期（尺寸统计）**
- 阶段 6：尺度标定落地 + 4.7 截线法，输出 L_bar/d/G；
- 阶段 7：标 10–20 张 GT，定量验证 + 调参，形成批处理工具。

**可选**：4.6 Watershed 实例分离，需要逐晶粒分布时再加。
