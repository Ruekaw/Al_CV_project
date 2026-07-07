"""批处理 pretrain_sets，跑通端到端并汇总结果。

用法（在项目根目录）：
  python scripts/batch_run.py                  # 跑全部
  python scripts/batch_run.py --limit 5        # 只跑前 5 张
  python scripts/batch_run.py --names 1-1晶 10198晶
"""
import os
import sys
import json
import time
import argparse
import yaml

# 让 `from src...` 在直接运行脚本时也能找到项目根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline  # noqa: E402
from src.image_io import imread_unicode  # noqa: E402


def list_images(pretrain_dir: str, names=None, limit=None):
    files = sorted(f for f in os.listdir(pretrain_dir) if f.lower().endswith((".jpg", ".png", ".tif")))
    if names:
        wanted = {n + ext for n in names for ext in (".jpg", ".png")}
        files = [f for f in files if f in wanted or os.path.splitext(f)[0] in set(names)]
    if limit:
        files = files[:limit]
    return [os.path.join(pretrain_dir, f) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default="pretrain_sets")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--names", nargs="*", default=None, help="只跑指定图名（不带扩展名）")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    images = list_images(args.input, args.names, args.limit)
    if not images:
        print(f"未找到图像：{args.input}")
        return

    print(f"共 {len(images)} 张，输出到 {args.runs}/")
    results = []
    t0 = time.time()
    for i, p in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {os.path.basename(p)} ...", flush=True)
        try:
            r = run_pipeline(p, cfg, runs_dir=args.runs)
        except Exception as e:
            import traceback
            print(f"  失败: {e}")
            traceback.print_exc()
            r = {"image": os.path.basename(p), "error": str(e)}
        results.append(r)
        m = r.get("intercept", {})
        print(f"    method={r.get('method')} "
              f"L_bar_px={m.get('L_bar_px')} d_px={m.get('d_px')} "
              f"grains~{r.get('closure_grain_count')} "
              f"t={r.get('elapsed_sec')}s")

    summary_path = os.path.join(args.runs, "summary.json")
    os.makedirs(args.runs, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总写入 {summary_path}  总耗时 {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
