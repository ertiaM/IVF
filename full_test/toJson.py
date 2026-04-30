#!/usr/bin/env python3
"""
新版 Excel → JSON 转换脚本
数据源：【胚胎发育动力学-蛋白质组学整合分析项目】结构化数据（2026-04-23）.xlsx

与旧版的主要差异：
  1. 以"训练患者名单"sheet为基准，只处理有视频+妊娠结局的样本
  2. 表10 新增字段：d01, d02, D0Ending, et_pt_id
  3. 表12 新增字段：opu_pt_id, opu_emno, d3msg, d3level, npmsg, nplevel,
                    pyday, cohem, emtech, lddate, jddate（第1/2胚胎各一套）
  4. 输出目录：/aifs4su/zhuhan/chenjiale/AI4Health/full_test/
"""

import pandas as pd
import json
import numpy as np
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── 路径配置 ────────────────────────────────────────────────────────────
EXCEL = Path(
    "/aifs4su/hansirui/中山妇产数据"
    "/【胚胎发育动力学-蛋白质组学整合分析项目】结构化数据（2026-04-23）.xlsx"
)
OUT_DIR    = Path("/aifs4su/zhuhan/chenjiale/AI4Health/full_test")
OUT_NESTED = OUT_DIR / "medical_records.json"
OUT_FLAT   = OUT_DIR / "medical_transfer_samples.json"

assert EXCEL.exists(), f"文件不存在: {EXCEL}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 工具函数 ────────────────────────────────────────────────────────────
def to_py(v):
    """pandas/numpy 类型 → JSON 可序列化的 Python 原生类型。"""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if (np.isnan(v) or np.isinf(v)) else float(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        fmt = "%Y-%m-%d %H:%M:%S" if (v.hour or v.minute or v.second) else "%Y-%m-%d"
        return v.strftime(fmt)
    if isinstance(v, str):
        return v.strip() or None
    return str(v)


def row2dict(row) -> dict:
    """DataFrame 行 → dict，过滤掉以 '_' 开头的辅助列。"""
    return {k: to_py(v) for k, v in row.items() if not str(k).startswith("_")}


def df2list(df: pd.DataFrame) -> list:
    """DataFrame → list[dict]。"""
    if df is None or df.empty:
        return []
    return [row2dict(r) for _, r in df.iterrows()]


def nid(v) -> str | None:
    """规范化 ID 为字符串，避免 int/float/str 混用导致跨表比对失败。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return str(int(float(v)))
    except Exception:
        return str(v).strip()


def qry(df: pd.DataFrame, cid: str, pid: str = None) -> pd.DataFrame:
    """按 couple_id（_cid）及可选的 pt_id（_pid）过滤 DataFrame。"""
    if df is None or df.empty or "_cid" not in df.columns:
        return pd.DataFrame()
    mask = df["_cid"] == cid
    if pid is not None and "_pid" in df.columns:
        mask &= df["_pid"] == pid
    return df[mask]


# ─── 读取 Excel ──────────────────────────────────────────────────────────
print(f"读取文件: {EXCEL}")
xl = pd.ExcelFile(EXCEL)
print(f"共 {len(xl.sheet_names)} 个 Sheet: {xl.sheet_names}\n")

T: dict[int, pd.DataFrame] = {}

for sn in xl.sheet_names:
    m = re.match(r"^(\d+)\.", str(sn))
    if not m:
        continue
    n = int(m.group(1))
    df = pd.read_excel(xl, sheet_name=sn)
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if "id" in df.columns:
        df["_cid"] = df["id"].apply(nid)
    if "pt_id" in df.columns:
        df["_pid"] = df["pt_id"].apply(nid)

    T[n] = df
    print(f"  表 {n:2d}: {df.shape[0]:5d} 行 × {df.shape[1]:3d} 列  ({sn})")

# 读取训练患者名单（有视频 + 有妊娠结局的样本基准）
train_df = pd.read_excel(xl, sheet_name="训练患者名单")
train_df.dropna(how="all", inplace=True)
train_df.columns = ["equipment", "couple_id", "pt_id"]
train_df["_cid"] = train_df["couple_id"].apply(nid)
train_df["_pid"] = train_df["pt_id"].apply(nid)
print(f"\n训练患者名单: {len(train_df)} 条 (couple, pt_id) 对")
print(f"成功加载 {len(T)} 张数据表\n")


# ─── 构建嵌套 JSON（按夫妇 → 按周期） ────────────────────────────────────
print("构建嵌套结构（medical_records.json）...")

# 以训练名单中的 couple_id 为基准
couple_ids = train_df["_cid"].dropna().unique().tolist()
print(f"  训练名单涉及 {len(couple_ids)} 对夫妇")

# 每对夫妇的有效 pt_id 集合（来自训练名单）
valid_pid_by_cid: dict[str, set] = {}
for _, row in train_df.iterrows():
    cid = row["_cid"]
    pid = row["_pid"]
    if cid and pid:
        valid_pid_by_cid.setdefault(cid, set()).add(pid)

nested_samples: list[dict] = []

for cid in couple_ids:
    # ── 患者级别（与周期无关）──────────────────────────────────────────
    basic_df = qry(T.get(1), cid)
    w_fv_df  = qry(T.get(2), cid)
    m_fv_df  = qry(T.get(3), cid)
    exam_df  = qry(T.get(6), cid)

    # 获取该夫妇在训练名单中的仪器信息
    equip_rows = train_df[train_df["_cid"] == cid]["equipment"].unique().tolist()
    equipment = equip_rows[0] if equip_rows else None

    patient = {
        "couple_id":         cid,
        "equipment":         equipment,
        "basic_info":        row2dict(basic_df.iloc[0]) if not basic_df.empty else {},
        "woman_first_visit": row2dict(w_fv_df.iloc[0])  if not w_fv_df.empty  else {},
        "man_first_visit":   row2dict(m_fv_df.iloc[0])  if not m_fv_df.empty  else {},
        "exam_reports":      df2list(exam_df),
        "cycles":            [],
    }

    # ── 周期级别：只保留训练名单中的 pt_id ────────────────────────────
    valid_pids = valid_pid_by_cid.get(cid, set())
    cycles_df  = qry(T.get(4), cid)

    for _, cr in cycles_df.iterrows():
        pid = nid(cr.get("pt_id"))
        if pid is None or pid not in valid_pids:
            continue

        # ── 标签：表 14 → clinic_preg ──────────────────────────────
        out_df     = qry(T.get(14), cid, pid)
        label_val  = None
        out_detail = {}
        if not out_df.empty:
            out_detail = row2dict(out_df.iloc[0])
            label_val  = out_detail.get("clinic_preg")

        # ── 移植记录（表 12）──────────────────────────────────────
        tr_df = qry(T.get(12), cid, pid)

        cycle = {
            "pt_id":                pid,
            "cycle_type":           to_py(cr.get("ctype")),
            # 表 4
            "cycle_record":         row2dict(cr),
            # 表 5：检验报告
            "lab_tests":            df2list(qry(T.get(5),  cid, pid)),
            # 表 7：周期监测记录
            "monitoring_records":   df2list(qry(T.get(7),  cid, pid)),
            # 表 8：用药记录
            "medication_records":   df2list(qry(T.get(8),  cid, pid)),
            # 表 9：IVF 手术记录
            "ivf_surgery_records":  df2list(qry(T.get(9),  cid, pid)),
            # 表 10：胚胎培养情况（含新字段 d01/d02/D0Ending/et_pt_id）
            "embryo_culture":       df2list(qry(T.get(10), cid, pid)),
            # 表 11：胚胎培养统计
            "embryo_stats":         df2list(qry(T.get(11), cid, pid)),
            # 表 12：胚胎移植情况（含新字段 opu_pt_id/opu_emno/d3msg 等）
            "embryo_transfer":      row2dict(tr_df.iloc[0]) if not tr_df.empty else {},
            # 表 13：妊娠监控记录
            "pregnancy_monitoring": df2list(qry(T.get(13), cid, pid)),
            # ── 标签（来自表 14）──────────────────────────────────
            "label": {
                "clinic_preg":  label_val,
                "outcome_full": out_detail,
            },
        }
        patient["cycles"].append(cycle)

    if patient["cycles"]:
        nested_samples.append(patient)

with open(OUT_NESTED, "w", encoding="utf-8") as f:
    json.dump(nested_samples, f, ensure_ascii=False, indent=2, default=str)
print(f"  已保存 → {OUT_NESTED}")


# ─── 构建平铺 JSON（每次移植 = 一条样本，适合ML训练） ──────────────────
print("\n构建平铺结构（medical_transfer_samples.json）...")

flat_samples: list[dict] = []

for pat in nested_samples:
    cid = pat["couple_id"]
    for cyc in pat["cycles"]:
        if not cyc["embryo_transfer"]:
            continue

        flat_samples.append({
            # ── 索引 ──────────────────────────────────────────────────
            "couple_id":            cid,
            "pt_id":                cyc["pt_id"],
            "cycle_type":           cyc["cycle_type"],
            "equipment":            pat["equipment"],
            # ── 患者级别特征（表 1-3, 6）─────────────────────────────
            "basic_info":           pat["basic_info"],
            "woman_first_visit":    pat["woman_first_visit"],
            "man_first_visit":      pat["man_first_visit"],
            "exam_reports":         pat["exam_reports"],
            # ── 周期级别特征（表 4, 5, 7-13）─────────────────────────
            "cycle_record":         cyc["cycle_record"],
            "lab_tests":            cyc["lab_tests"],
            "monitoring_records":   cyc["monitoring_records"],
            "medication_records":   cyc["medication_records"],
            "ivf_surgery_records":  cyc["ivf_surgery_records"],
            "embryo_culture":       cyc["embryo_culture"],
            "embryo_stats":         cyc["embryo_stats"],
            "embryo_transfer":      cyc["embryo_transfer"],
            "pregnancy_monitoring": cyc["pregnancy_monitoring"],
            # ── 标签（表 14 clinic_preg）──────────────────────────────
            "label":                cyc["label"]["clinic_preg"],
            "outcome_detail":       cyc["label"]["outcome_full"],
        })

with open(OUT_FLAT, "w", encoding="utf-8") as f:
    json.dump(flat_samples, f, ensure_ascii=False, indent=2, default=str)
print(f"  已保存 → {OUT_FLAT}")


# ─── 统计摘要 ────────────────────────────────────────────────────────────
total_cyc = sum(len(p["cycles"]) for p in nested_samples)
n_flat    = len(flat_samples)
n_label   = sum(1 for s in flat_samples if s["label"] is not None)
n_pos     = sum(1 for s in flat_samples if s["label"] == "阳性")
n_neg     = sum(1 for s in flat_samples if s["label"] == "阴性")

print(f"""
═══════════════ 统计摘要 ═══════════════
  训练名单样本数:             {len(train_df)}
  涉及夫妇数:                 {len(nested_samples)}
  总周期数:                   {total_cyc}
  含移植周期数（平铺样本）:   {n_flat}
  含标签的样本数:             {n_label}
    阳性（临床妊娠）:         {n_pos}
    阴性（未妊娠）:           {n_neg}
════════════════════════════════════════
""")
