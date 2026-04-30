#!/usr/bin/env python3
"""
仅使用胚胎培养视频帧/图片预测 IVF 冷冻胚胎移植临床妊娠结局。
数据源：full_test/sampling_test/medical_records.json（30条预筛样本）

视频/图片匹配逻辑：
  - Geri      : VIDEO_ROOT_GERI/{pt_id}/well{no:02d}_zid06.mp4
                按 well 编号升序取前 MAX_VIDEOS 个，每个视频均匀抽 FRAMES_PER_VIDEO 帧
  - Vitrolife : IMAGE_ROOT_VITROLIFE/{pt_id}/D{date}_..._WELL{no}/D{date}_..._WELL{no}_RUN{n}.JPG
                按 well 编号升序取前 MAX_VIDEOS 个 WELL，每个 WELL 均匀抽 FRAMES_PER_VIDEO 张图片
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
BASE        = Path("/aifs4su/zhuhan/chenjiale/AI4Health/full_test/sampling_test")
JSON_PATH   = BASE / "medical_records.json"
OUT_PATH    = BASE / "predictions_video_only.json"

VIDEO_ROOT_GERI      = Path("/aifs4su/hansirui/中山妇产数据/Geri仪器胚胎视频")
IMAGE_ROOT_VITROLIFE = Path("/aifs4su/hansirui/中山妇产数据/Vitrolife仪器胚胎视频"
                            "/有妊娠结局周期_图片（截止日期2026-04-28）")

# ─── API ─────────────────────────────────────────────────────────────────────
API_URL = "https://api3.xhub.chat/v1/chat/completions"
API_KEY = "sk-Co7aVxrhEYKOomDszhPgOvksE1sLEaOLl4MVV5g1AnOaTE1z"
MODEL   = "gemini-3.1-pro-preview"

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

请仅根据视频帧，预测本次冷冻胚胎移植（FET）是否会发生临床妊娠。

══════════════════ 视频帧说明 ══════════════════
- 每组帧标注了来源胚胎编号（no），帧按时间顺序均匀采样自培养视频
- 视频记录胚胎从受精（Day0）到囊胚期（Day5-7）的完整培养过程
- 重点观察：细胞分裂对称性、碎片率、囊胚形成质量、透明带完整性

══════════════════ 输出格式（严格）══════════════════
必须且只能输出以下 JSON，不得有任何额外文字或 markdown 代码块：
{
  "cot": "详细推理过程，结合视频帧观察综合分析",
  "result": "阳性",
  "confidence": 0.75
}
result 只能填写"阳性"或"阴性"，confidence 为 0.0~1.0 之间的浮点数。
"""


# ─── 索引构建 ────────────────────────────────────────────────────────────────
def build_pt_index(root: Path) -> dict[str, Path]:
    """文件夹名直接是 pt_id，建立 pt_id → folder 索引。"""
    index: dict[str, Path] = {}
    try:
        if not root.exists():
            print(f"[WARN] 目录不存在: {root}")
            return index
        for folder in root.iterdir():
            if folder.is_dir():
                index[folder.name] = folder
        print(f"[INFO] 索引 ({root.name}): {len(index)} 个文件夹")
    except PermissionError:
        print(f"[WARN] 无权限访问: {root}")
    return index


VIDEO_INDEX_GERI      = build_pt_index(VIDEO_ROOT_GERI)
IMAGE_INDEX_VITROLIFE = build_pt_index(IMAGE_ROOT_VITROLIFE)


# ─── Geri: 视频路径获取 ───────────────────────────────────────────────────────
def get_geri_video_paths(pt_id: str) -> list[Path]:
    """
    Geri 文件名格式：well{no:02d}_zid**.mp4，按 well 编号升序取前 MAX_VIDEOS 个。
    """
    folder = VIDEO_INDEX_GERI.get(str(pt_id))
    if folder is None:
        return []
    videos = []
    for f in folder.glob("*.mp4"):
        m = re.match(r"^well(\d+)_", f.name)
        if m:
            videos.append((int(m.group(1)), f))
    videos.sort(key=lambda x: x[0])
    return [f for _, f in videos[:MAX_VIDEOS]]


def no_from_geri_path(video_path: Path) -> str:
    """从文件名 well{no}_zid**.mp4 提取 well 编号。"""
    m = re.match(r"^well(\d+)_", video_path.name)
    return m.group(1) if m else video_path.stem


# ─── Vitrolife: 图片路径获取 ──────────────────────────────────────────────────
def get_vitrolife_well_dirs(pt_id: str) -> list[tuple[int, Path]]:
    """
    Vitrolife 图片目录结构：{pt_id}/D{date}_..._WELL{no}/D{date}_..._WELL{no}_RUN{n}.JPG
    返回按 well 编号升序排列的 (well_no, well_dir) 列表，取前 MAX_VIDEOS 个。
    """
    folder = IMAGE_INDEX_VITROLIFE.get(str(pt_id))
    if folder is None:
        return []
    wells = []
    for d in folder.iterdir():
        if not d.is_dir():
            continue
        m = re.search(r"_WELL(\d+)$", d.name)
        if m:
            wells.append((int(m.group(1)), d))
    wells.sort(key=lambda x: x[0])
    return wells[:MAX_VIDEOS]


# ─── 帧/图片提取与压缩 ────────────────────────────────────────────────────────
def extract_frames_b64(video_path: Path,
                        n_frames: int = FRAMES_PER_VIDEO) -> list[str]:
    """从视频均匀抽取 n_frames 帧，压缩后返回 base64 列表。"""
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


def extract_images_b64(well_dir: Path,
                        n_frames: int = FRAMES_PER_VIDEO) -> list[str]:
    """
    从 Vitrolife WELL 目录均匀抽取 n_frames 张 JPG，压缩后返回 base64 列表。
    图片按 RUN 编号升序排列（即时间顺序）。
    """
    imgs = sorted(well_dir.glob("*.JPG"),
                  key=lambda f: int(m.group(1)) if (m := re.search(r"_RUN(\d+)\.JPG$", f.name)) else 0)
    if not imgs:
        return []
    total = len(imgs)
    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        img = cv2.imread(str(imgs[idx]))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > IMAGE_LONG_SIDE:
            scale = IMAGE_LONG_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", img,
                              [cv2.IMWRITE_JPEG_QUALITY, IMAGE_JPEG_QUALITY])
        frames.append(base64.b64encode(buf).decode("utf-8"))
    return frames


# ─── 构建请求内容 ─────────────────────────────────────────────────────────────
def build_user_content_geri(video_paths: list[Path]) -> list[dict] | str:
    """Geri：从视频均匀抽帧。"""
    intro = "请根据以下胚胎培养视频帧，预测本次冷冻胚胎移植的临床妊娠结局。"
    if not video_paths:
        return intro

    content: list[dict] = [{"type": "text", "text": intro}]
    for vp in video_paths:
        embryo_no = no_from_geri_path(vp)
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
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def build_user_content_vitrolife(well_dirs: list[tuple[int, Path]]) -> list[dict] | str:
    """Vitrolife：从每个 WELL 目录均匀抽图片。"""
    intro = "请根据以下胚胎培养图片，预测本次冷冻胚胎移植的临床妊娠结局。"
    if not well_dirs:
        return intro

    content: list[dict] = [{"type": "text", "text": intro}]
    for well_no, well_dir in well_dirs:
        frames = extract_images_b64(well_dir)
        if not frames:
            print(f"    [WARN] 无法读取图片: {well_dir.name}")
            continue
        content.append({
            "type": "text",
            "text": (
                f"【胚胎编号 no={well_no:02d} 的培养图片】\n"
                f"以下 {len(frames)} 张图片为该胚胎从受精到囊胚期的培养过程，按时间顺序均匀采样。"
            )
        })
        for b64 in frames:
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
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
    print(f"加载数据: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # sampling_test 每条记录只含一个周期，pt_id 即视频文件夹名
    samples = []
    for patient in records:
        cid       = str(patient["couple_id"])
        equipment = patient.get("equipment", "")
        cyc       = patient["cycles"][0]
        pid       = str(cyc["pt_id"])
        label     = cyc.get("label", {}).get("clinic_preg")

        if label not in ("阳性", "阴性"):
            continue

        samples.append({
            "couple_id":  cid,
            "pt_id":      pid,
            "equipment":  equipment,
            "true_label": label,
        })

    if MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]

    # ── 断点续跑：加载已有结果，跳过已处理的 pt_id ──────────────────────────
    done_pids: set[str] = set()
    results: list[dict] = []
    if OUT_PATH.exists():
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_pids = {str(r["pt_id"]) for r in results}
        print(f"断点续跑：已完成 {len(done_pids)} 个样本，跳过")

    remaining = [s for s in samples if s["pt_id"] not in done_pids]
    print(f"待预测样本: {len(remaining)} / {len(samples)}\n")

    for idx, s in enumerate(remaining, 1):
        cid  = s["couple_id"]
        pid  = s["pt_id"]
        eq   = s["equipment"]
        print(f"[{idx:2d}/{len(remaining)}] couple={cid}  pt_id={pid}  equipment={eq}", end="  ")

        if eq == "Geri":
            video_paths = get_geri_video_paths(pid)
            media_count = len(video_paths)
            print(f"视频: {media_count} 个", end="  ")
            user_content = build_user_content_geri(video_paths)
        else:  # Vitrolife
            well_dirs = get_vitrolife_well_dirs(pid)
            media_count = len(well_dirs)
            print(f"WELL: {media_count} 个", end="  ")
            user_content = build_user_content_vitrolife(well_dirs)

        if media_count == 0:
            print("跳过（无数据）")
            entry = {
                "couple_id":   cid,
                "pt_id":       pid,
                "equipment":   eq,
                "true_label":  s["true_label"],
                "media_count": 0,
                "frame_count": 0,
                "cot":         "",
                "result":      "",
                "confidence":  None,
            }
        else:
            llm_out = call_llm(user_content)
            if llm_out:
                entry = {
                    "couple_id":   cid,
                    "pt_id":       pid,
                    "equipment":   eq,
                    "true_label":  s["true_label"],
                    "media_count": media_count,
                    "frame_count": media_count * FRAMES_PER_VIDEO,
                    "cot":         llm_out.get("cot", ""),
                    "result":      llm_out.get("result", ""),
                    "confidence":  llm_out.get("confidence"),
                }
                print(f"result={entry['result']}  conf={entry['confidence']}")
            else:
                entry = {
                    "couple_id":   cid,
                    "pt_id":       pid,
                    "equipment":   eq,
                    "true_label":  s["true_label"],
                    "media_count": media_count,
                    "frame_count": 0,
                    "cot":         "",
                    "result":      "",
                    "confidence":  None,
                }
                print("失败")

        results.append(entry)

        # 每完成一个样本立即写回文件
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(REQUEST_INTERVAL)

    n_done = sum(1 for r in results if r["result"])
    print(f"\n已保存 → {OUT_PATH}  (共 {len(results)} 条，成功预测 {n_done} 条)")


if __name__ == "__main__":
    main()
