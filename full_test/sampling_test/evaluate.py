#!/usr/bin/env python3
"""
预测结果统计评估脚本

配置：
  PRED_PATH   - 要评估的 predictions JSON 文件路径
  SKIP_INDICES - 按样本编号（1-based）排除不参与统计的样本，例如 [1, 5, 12]
"""

import json
from pathlib import Path

# ─── 配置 ─────────────────────────────────────────────────────────────────────
PRED_PATH    = Path("/aifs4su/zhuhan/chenjiale/AI4Health/full_test/sampling_test/predictions_video_protein_ppi.json")
SKIP_INDICES = [6, 7, 14, 16, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30]   # 例：[1, 3, 7]，填入不参与统计的样本编号（1-based）
# ──────────────────────────────────────────────────────────────────────────────

with open(PRED_PATH, encoding="utf-8") as f:
    records = json.load(f)

print(f"预测文件: {PRED_PATH.name}")
print(f"总样本数: {len(records)}")
print(f"排除编号: {SKIP_INDICES if SKIP_INDICES else '无'}\n")

# 打印所有样本列表（含编号），方便决定 SKIP_INDICES
print(f"{'编号':>4}  {'pt_id':<12} {'equipment':<12} {'真实标签':<6} {'预测结果':<6} {'置信度':<6}  {'是否正确'}")
print("─" * 72)
for i, r in enumerate(records, 1):
    true   = r.get("true_label", "")
    pred   = r.get("result", "")
    conf   = r.get("confidence")
    conf_s = f"{conf:.2f}" if conf is not None else "  -  "
    skip   = "（跳过）" if i in SKIP_INDICES else ""
    correct = ""
    if pred and not skip:
        correct = "✓" if pred == true else "✗"
    print(f"[{i:2d}/{ len(records)}]  {r['pt_id']:<12} {r.get('equipment',''):<12} {true:<6} {pred or '-':<6} {conf_s:<6}  {correct}{skip}")

# ─── 统计 ─────────────────────────────────────────────────────────────────────
active = [r for i, r in enumerate(records, 1) if i not in SKIP_INDICES]
valid  = [r for r in active if r.get("result") in ("阳性", "阴性")]
skipped_count = len(records) - len(active)
no_pred_count = len(active) - len(valid)

if not valid:
    print("\n没有可统计的有效预测结果。")
else:
    tp = sum(1 for r in valid if r["result"] == "阳性" and r["true_label"] == "阳性")
    tn = sum(1 for r in valid if r["result"] == "阴性" and r["true_label"] == "阴性")
    fp = sum(1 for r in valid if r["result"] == "阳性" and r["true_label"] == "阴性")
    fn = sum(1 for r in valid if r["result"] == "阴性" and r["true_label"] == "阳性")

    n       = len(valid)
    acc     = (tp + tn) / n
    prec    = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1      = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else float("nan")

    n_pos = sum(1 for r in valid if r["true_label"] == "阳性")
    n_neg = sum(1 for r in valid if r["true_label"] == "阴性")

    print(f"\n{'═'*40}")
    print(f"  参与统计样本数:  {n}  （排除 {skipped_count} 个，无预测 {no_pred_count} 个）")
    print(f"  真实阳性 / 阴性: {n_pos} / {n_neg}")
    print(f"\n  混淆矩阵:")
    print(f"              预测阳性   预测阴性")
    print(f"  真实阳性      {tp:3d}        {fn:3d}")
    print(f"  真实阴性      {fp:3d}        {tn:3d}")
    print(f"\n  准确率  (Accuracy) : {acc:.4f}  ({tp+tn}/{n})")
    print(f"  精确率  (Precision): {prec:.4f}")
    print(f"  召回率  (Recall)   : {recall:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"{'═'*40}")
