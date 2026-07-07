# CVproject：铝晶粒显微图像晶界闭合与尺寸统计

本项目用于处理铝晶粒显微镜图像，目标是从不规则晶粒/支晶/噪点较多的原图中提取较干净的晶界网络，并进一步用截线法估计晶粒尺寸。

当前阶段重点是跑通 MatSAM 候选边界生成，再接后处理完成晶界清洗、断口修复和直径统计。

## 任务流程

```text
原始显微图像
  -> 预处理增强
  -> MatSAM 或 baseline 生成候选晶界
  -> 边界段切分
  -> 删除噪点/孤立段/支晶段
  -> 断口修复
  -> ASTM E112 多方向截线统计
  -> 输出晶界图、报告图、直径结果
```

## 项目结构

```text
CVproject/
├── pretrain_sets/              # 输入样本图像
├── MatSAM/                     # MatSAM/SAM 源码
├── src/                        # 主流程模块
├── scripts/batch_run.py        # 批处理入口
├── config.yaml                 # 全局参数
├── plan.md                     # 项目方案
├── CLOUD_MATSAM.md             # 云端 MatSAM 运行说明
├── requirements-cloud.txt      # 云端基础依赖，不含 torch
└── runs/                       # 本地输出目录，已被 .gitignore 忽略
```

config.yaml` 中已经把 `matsam.points_per_batch` 做成可调参数，默认 `64`，如果云端 OOM 可降到 `32` 或 `16`。

## 云端快速运行

上传代码后，在云端执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-cloud.txt
```

下载 `vit_b` 权重：

```bash
mkdir -p MatSAM/checkpoints
wget -O MatSAM/checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

跑一张探针图：

```bash
python scripts/batch_run.py --names 1-1晶 --runs runs_matsam_cloud_probe
```

如果日志中出现 `method=matsam`，说明 MatSAM 路径已经跑通；如果出现 `method=baseline`，先看前面的 `[matsam] ...` 错误信息，通常是缺依赖、缺权重或 CUDA OOM。

## 批处理

跑全部样本：

```bash
python scripts/batch_run.py --runs runs_matsam
```

只跑指定图：

```bash
python scripts/batch_run.py --names 1-1晶 10198晶 --runs runs_matsam_probe
```

## 输出文件

每张图会输出到 `runs/{图名}/`：

```text
01_preprocess.png        # 预处理增强图
02_candidate_boundary.png # 候选晶界二值图
03_cleaned_boundary.png  # 删除噪点/支晶后的叠加图
04_final_boundary.png    # 修复后的最终晶界叠加图
05_intercept.png         # 截线统计可视化
report.png               # 总览报告图
result.json              # 统计结果
```

`result.json` 中主要关注：

- `method`：实际使用 `matsam` 还是 `baseline`
- `closure_grain_count`：闭合区域粗略计数
- `intercept.d_px`：像素单位的等效晶粒直径
- `intercept.d_um`：物理单位直径，需先设置 `scale.um_per_pixel`

## 权重与大文件

`.gitignore` 默认忽略以下内容：

- `MatSAM/checkpoints/*.pth`
- `runs/`
- `runs_review/`
- `dist/`
- Python 缓存文件

因此 push 到远端时不会携带几 GB 的 SAM 权重和运行输出。云端需要按上面的命令重新下载 checkpoint。

## 下一步建议

1. 云端先跑 `vit_b` + `points_per_batch: 64`。
2. 若 OOM，降到 `32` 或 `16`。
3. 若 `vit_b` 效果不够，再换 24GB+ 显存尝试 `vit_h`。
4. MatSAM 跑通后，再回到 `boundary_clean` 和 `repair` 参数调优。
