#!/usr/bin/env python3
import json
from pathlib import Path

PRED_PATH = Path("/aifs4su/zhuhan/chenjiale/AI4Health/predictions_video_gpt-gemini-3.5-flash.json")


def norm(raw: str) -> str:
    if "阳" in str(raw): return "阳性"
    if "阴" in str(raw): return "阴性"
    return "未知"


def metrics(y_true, y_pred, label):
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def confusion_matrix(y_true, y_pred, labels):
    print("\n混淆矩阵（行=真实，列=预测）")
    header = " " * 8 + "  ".join(f"预测{l}" for l in labels)
    print(header)
    for tl in labels:
        row = f"真实{tl}  " + "  ".join(
            f"{sum(t == tl and p == pl for t, p in zip(y_true, y_pred)):8d}"
            for pl in labels
        )
        print(row)


def main():
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total   = len(data)
    failed  = [d for d in data if norm(d["result"]) == "未知"]
    valid   = [d for d in data if norm(d["result"]) != "未知"]

    print(f"总样本: {total}  有效预测: {len(valid)}  失败/未知: {len(failed)}")

    if not valid:
        print("无有效预测，退出。")
        return

    y_true = [d["true_label"] for d in valid]
    y_pred = [norm(d["result"]) for d in valid]

    n_correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = n_correct / len(valid)
    print(f"\n准确率 (Acc): {acc:.4f}  ({n_correct}/{len(valid)})")

    labels = ["阳性", "阴性"]
    print(f"\n{'':12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 44)
    for lbl in labels:
        p, r, f = metrics(y_true, y_pred, lbl)
        print(f"{lbl:12} {p:10.4f} {r:10.4f} {f:10.4f}")

    # Macro 平均
    ps, rs, fs = zip(*[metrics(y_true, y_pred, lbl) for lbl in labels])
    print(f"{'Macro avg':12} {sum(ps)/len(ps):10.4f} {sum(rs)/len(rs):10.4f} {sum(fs)/len(fs):10.4f}")

    confusion_matrix(y_true, y_pred, labels)

    # 置信度分布
    confs = [d["confidence"] for d in valid if d["confidence"] is not None]
    if confs:
        avg_conf = sum(confs) / len(confs)
        correct_confs  = [d["confidence"] for d in valid
                          if d["confidence"] is not None and norm(d["result"]) == d["true_label"]]
        wrong_confs    = [d["confidence"] for d in valid
                          if d["confidence"] is not None and norm(d["result"]) != d["true_label"]]
        print(f"\n置信度统计")
        print(f"  平均置信度:   {avg_conf:.4f}")
        print(f"  预测正确均值: {sum(correct_confs)/len(correct_confs):.4f}" if correct_confs else "")
        print(f"  预测错误均值: {sum(wrong_confs)/len(wrong_confs):.4f}"   if wrong_confs   else "")


if __name__ == "__main__":
    main()