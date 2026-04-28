#!/usr/bin/env python3
"""
使用胚胎培养视频帧 + 最小文本（胚胎评分 + 移植记录）+ 夫妻蛋白/检验指标
预测 IVF 冷冻胚胎移植（FET）临床妊娠结局。

视频匹配逻辑：
  - 文件夹名 {前缀}_{新鲜周期pt_id}，按后半部分 pt_id 索引
  - 视频文件名 {pt_id}_{胚胎no}.avi，按 no 数字排序取前 MAX_VIDEOS 个
  - 每个视频均匀抽 FRAMES_PER_VIDEO 帧

蛋白数据处理：
  - result 为非数值（含 "-"、None 等）的条目直接过滤
  - 按性别分为男方/女方，按 itemname 分组呈现
  - abnormal 字段（↑/↓）标注异常方向
"""

import json
import re
import time
import base64
import requests
import cv2
from pathlib import Path
from typing import Optional

# ─── 路径 ────────────────────────────────────────────────────────────────────
BASE            = Path("/aifs4su/zhuhan/chenjiale/AI4Health")
JSON_PATH       = BASE / "medical_records.json"
PROTEIN_PATH    = BASE / "protein_records.json"
OUT_PATH        = BASE / "pre_test/results/predictions_video_protein.json"
OUT_TEXT_PATH   = BASE / "pre_test/results/inputs_video_protein.json"
VIDEO_ROOT      = Path("/aifs4su/zhanqimin/ai_for_healthcare/data"
                       "/港科大-深圳中山AI项目资料(2026-03-05)/胚胎影像视频")

# ─── API ─────────────────────────────────────────────────────────────────────
API_URL = "https://api3.xhub.chat/v1/chat/completions"
API_KEY = "sk-Co7aVxrhEYKOomDszhPgOvksE1sLEaOLl4MVV5g1AnOaTE1z"
MODEL   = "gemini-2.5-flash"

# ─── 采样参数 ─────────────────────────────────────────────────────────────────
MAX_VIDEOS         = 4
FRAMES_PER_VIDEO   = 4
IMAGE_LONG_SIDE    = 512
IMAGE_JPEG_QUALITY = 55

REQUEST_INTERVAL = 2.0
MAX_SAMPLES      = None  # 调试时改为 3


# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一位生殖医学专家，专门从事 IVF（试管婴儿）胚胎评估。

你将收到：
1. 一批胚胎培养视频帧（来自新鲜周期，每组帧对应一枚胚胎从受精到囊胚期的完整培养过程）
2. 夫妻双方的实验室检验指标（lab_tests，包含男方/女方的血常规、凝血、激素、免疫等）

请综合以上信息，预测本次冷冻胚胎移植（FET）是否会发生临床妊娠。

══════════════════ 视频帧说明 ══════════════════
- 每组帧标注了来源胚胎编号（no），帧按时间顺序均匀采样自培养视频
- 视频记录胚胎从受精（Day0）到囊胚期（Day5-7）的完整培养过程
- 重点观察：细胞分裂对称性、碎片率、囊胚形成质量、透明带完整性

══════════════════ 字段说明 ══════════════════

▌ lab_tests（夫妻实验室检验指标）
  结构为 {"男方": {"检查项目名": [{"采样日期": ..., "指标": ..., "结果": ..., "单位": ..., "异常": ..., "参考范围": ...}, ...]}, "女方": {...}}
  - 同一指标可能有多条记录，对应不同检查日期，请结合时序综合判断
  - 异常字段：↑ 表示偏高，↓ 表示偏低，正常则为空
  - 重点关注：女方激素水平（FSH/LH/E2/AMH/PRL）、凝血功能、免疫指标；男方精液相关指标

══════════════════ 输出格式（严格）══════════════════
必须且只能输出以下 JSON，不得有任何额外文字或 markdown 代码块：
{
  "cot": "详细推理过程，结合视频帧观察与检验指标综合分析",
  "result": "阳性",
  "confidence": 0.75
}
result 只能填写"阳性"或"阴性"，confidence 为 0.0~1.0 之间的浮点数。
"""


# ─── 视频索引 ─────────────────────────────────────────────────────────────────
def build_video_index(video_root: Path) -> dict[str, Path]:
    """按文件夹后半部分（新鲜周期 pt_id）建立索引。"""
    index: dict[str, Path] = {}
    if not video_root.exists():
        print(f"[WARN] 视频目录不存在: {video_root}")
        return index
    for folder in video_root.iterdir():
        if not folder.is_dir():
            continue
        parts = folder.name.split("_", 1)
        if len(parts) == 2:
            index[parts[1]] = folder
    print(f"[INFO] 视频索引: {len(index)} 个文件夹")
    return index


VIDEO_INDEX = build_video_index(VIDEO_ROOT)


def get_video_paths(fresh_pt_id: str) -> list[Path]:
    """
    返回该 pt_id 对应文件夹内的视频列表，按胚胎编号 no 数字升序，取前 MAX_VIDEOS 个。
    文件名格式：{pt_id}_{no}.avi
    """
    folder = VIDEO_INDEX.get(str(fresh_pt_id))
    if folder is None:
        return []
    videos = []
    for f in folder.glob("*.avi"):
        m = re.match(rf"^{re.escape(str(fresh_pt_id))}_(\d+)\.avi$", f.name)
        if m:
            videos.append((int(m.group(1)), f))
    videos.sort(key=lambda x: x[0])
    return [f for _, f in videos[:MAX_VIDEOS]]


# ─── 帧提取与压缩 ─────────────────────────────────────────────────────────────
def extract_frames_b64(video_path: Path,
                        n_frames: int = FRAMES_PER_VIDEO) -> list[str]:
    """均匀抽取 n_frames 帧，压缩后返回 base64 列表。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > IMAGE_LONG_SIDE:
            scale = IMAGE_LONG_SIDE / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", frame,
                              [cv2.IMWRITE_JPEG_QUALITY, IMAGE_JPEG_QUALITY])
        frames.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return frames


def no_from_path(video_path: Path) -> str:
    """从文件名 {pt_id}_{no}.avi 提取胚胎编号。"""
    m = re.search(r"_(\d+)\.avi$", video_path.name)
    return m.group(1) if m else video_path.stem


# ─── 蛋白/检验数据处理 ────────────────────────────────────────────────────────
def is_numeric(value) -> bool:
    """判断 result 是否为有效数值（过滤 "-"、None 等）。"""
    if value is None:
        return False
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def build_protein_index(protein_path: Path) -> dict[str, dict]:
    """
    从 protein_records.json 构建以 couple_id 为键的蛋白数据索引。
    结构：
      {
        couple_id: {
          "男方": { itemname: [ {指标, 结果, 单位, 异常, 参考范围}, ... ] },
          "女方": { itemname: [ ... ] }
        }
      }
    只保留 result 为有效数值的条目。
    """
    with open(protein_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    index: dict[str, dict] = {}
    for rec in records:
        cid = str(rec["couple_id"])
        grouped: dict[str, dict[str, list]] = {"男方": {}, "女方": {}}

        for t in rec.get("protein_tests", []):
            if not is_numeric(t.get("result")):
                continue
            gender_key = "男方" if t.get("sex") == "男" else "女方"
            item = t.get("itemname", "未知项目")
            grouped[gender_key].setdefault(item, [])
            entry = {
                "采样日期": t.get("sampletime", ""),
                "指标": t.get("name1", t.get("name2", "")),
                "结果": t["result"],
                "单位": t.get("unit", ""),
            }
            if t.get("abnormal"):
                entry["异常"] = t["abnormal"]
            if t.get("reference3"):
                entry["参考范围"] = t["reference3"]
            grouped[gender_key][item].append(entry)

        # 只保留有数据的性别分组
        index[cid] = {k: v for k, v in grouped.items() if v}

    print(f"[INFO] 蛋白索引: {len(index)} 个 couple")
    return index


# ─── 构建请求内容 ─────────────────────────────────────────────────────────────
def build_text_data(protein_data: Optional[dict]) -> dict:
    """组装发给 LLM 的文本数据字典（仅检验指标）。"""
    if protein_data:
        return {"lab_tests": protein_data}
    return {}


def build_user_content(text_data: dict,
                        video_paths: list[Path]) -> list[dict] | str:
    intro = "请根据以下夫妻实验室检验指标和胚胎培养视频帧，预测本次冷冻胚胎移植的临床妊娠结局。\n\n"

    if text_data:
        intro += json.dumps(text_data, ensure_ascii=False, indent=2) + "\n\n"

    if not video_paths:
        return intro.strip()

    content: list[dict] = [{"type": "text", "text": intro.strip()}]

    for vp in video_paths:
        embryo_no = no_from_path(vp)
        frames = extract_frames_b64(vp)
        if not frames:
            print(f"    [WARN] 无法读取视频帧: {vp.name}")
            continue

        content.append({
            "type": "text",
            "text": (
                f"【胚胎编号 no={embryo_no} 的培养视频帧】\n"
                f"以下 {len(frames)} 张图片为该胚胎从受精到囊胚期的培养过程，按时间顺序均匀采样。"
            )
        })
        for b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    return content


# ─── LLM 调用 ─────────────────────────────────────────────────────────────────
def call_llm(user_content, retries: int = 3) -> Optional[dict]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens":  8192,
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers,
                                 json=payload, timeout=180)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                cleaned = re.sub(r'[\x00-\x1f\x7f](?<!["\n\r\t])',
                                 lambda c: repr(c.group())[1:-1], m.group())
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    return json.loads(cleaned)
            print(f"    [WARN] 未找到 JSON: {raw[:120]}")
            return None
        except requests.exceptions.Timeout:
            print(f"    [WARN] 超时 (attempt {attempt}/{retries})")
            time.sleep(10)
        except json.JSONDecodeError as e:
            print(f"    [WARN] JSON 解析失败: {e}")
            return None
        except Exception as e:
            print(f"    [WARN] 调用异常: {e} (attempt {attempt}/{retries})")
            time.sleep(5)
    return None


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    print(f"加载病历数据: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"加载蛋白数据: {PROTEIN_PATH}")
    PROTEIN_INDEX = build_protein_index(PROTEIN_PATH)

    # 构建样本：每个 couple 找到 新鲜周期（有视频）+ 复苏周期（有 label + 移植记录）
    samples = []
    for patient in records:
        cid    = str(patient["couple_id"])
        cycles = patient.get("cycles", [])

        fresh_cycle    = None
        transfer_cycle = None

        for c in cycles:
            pid = str(c.get("pt_id", ""))
            if c.get("embryo_culture") and pid in VIDEO_INDEX:
                fresh_cycle = c
            label = c.get("label", {}).get("clinic_preg")
            if c.get("embryo_transfer") and label in ("阳性", "阴性"):
                transfer_cycle = c

        if fresh_cycle is None or transfer_cycle is None:
            continue

        samples.append({
            "couple_id":      cid,
            "fresh_pt_id":    str(fresh_cycle.get("pt_id")),
            "transfer_pt_id": str(transfer_cycle.get("pt_id")),
            "true_label":     transfer_cycle["label"]["clinic_preg"],
            "protein_data":   PROTEIN_INDEX.get(cid),
        })

    if MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]

    print(f"待预测样本: {len(samples)}\n")

    results = []
    text_records = []
    for idx, s in enumerate(samples, 1):
        cid  = s["couple_id"]
        fpid = s["fresh_pt_id"]
        has_protein = s["protein_data"] is not None
        print(f"[{idx:2d}/{len(samples)}] couple={cid}  fresh_pt_id={fpid}  蛋白={'有' if has_protein else '无'}", end="  ")

        video_paths = get_video_paths(fpid)
        print(f"视频: {len(video_paths)} 个", end="  ")

        text_data    = build_text_data(s["protein_data"])
        user_content = build_user_content(text_data, video_paths)

        # 记录视频路径及帧数
        video_info = [
            {
                "path":        str(vp),
                "embryo_no":   no_from_path(vp),
                "frames_sent": FRAMES_PER_VIDEO,
            }
            for vp in video_paths
        ]

        text_records.append({
            "couple_id":   cid,
            "fresh_pt_id": fpid,
            "video_info":  video_info,
            "text_input":  text_data,
        })

        llm_out = call_llm(user_content)

        base_entry = {
            "couple_id":      cid,
            "fresh_pt_id":    fpid,
            "transfer_pt_id": s["transfer_pt_id"],
            "true_label":     s["true_label"],
            "has_protein":    has_protein,
            "video_count":    len(video_paths),
            "frame_count":    len(video_paths) * FRAMES_PER_VIDEO,
        }

        if llm_out:
            entry = {
                **base_entry,
                "cot":        llm_out.get("cot", ""),
                "result":     llm_out.get("result", ""),
                "confidence": llm_out.get("confidence"),
            }
            print(f"result={entry['result']}  conf={entry['confidence']}")
        else:
            entry = {
                **base_entry,
                "frame_count": 0,
                "cot":         "",
                "result":      "",
                "confidence":  None,
            }
            print("失败")

        results.append(entry)
        time.sleep(REQUEST_INTERVAL)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存预测结果 → {OUT_PATH}  (共 {len(results)} 条)")

    with open(OUT_TEXT_PATH, "w", encoding="utf-8") as f:
        json.dump(text_records, f, ensure_ascii=False, indent=2)
    print(f"已保存文字输入 → {OUT_TEXT_PATH}  (共 {len(text_records)} 条)")


if __name__ == "__main__":
    main()
