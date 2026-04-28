#!/usr/bin/env python3
import json
import requests
import re
import time
from pathlib import Path
from typing import Optional

BASE      = Path("/aifs4su/zhuhan/chenjiale/AI4Health")
JSON_PATH = BASE / "medical_records.json"
OUT_PATH  = BASE / "pre_test/results/predictions_gemini-2.5-flash.json"

API_URL = "https://api3.xhub.chat/v1/chat/completions"
API_KEY = "sk-Co7aVxrhEYKOomDszhPgOvksE1sLEaOLl4MVV5g1AnOaTE1z"
MODEL   = "gemini-2.5-flash"

REQUEST_INTERVAL = 2.0
MAX_SAMPLES      = None  # 调试时改为 5

# ─── System Prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一位生殖医学专家，专门从事 IVF（试管婴儿）诊疗。
你将收到一份 JSON 格式的 IVF 患者完整病历数据，请预测本次胚胎移植是否会发生临床妊娠。

══════════════════════ 数据字段说明 ══════════════════════

▌ basic_info（表1 基本资料）
  woman_birth/man_birth        : 女方/男方生日
  woman_education/man_education: 教育程度
  woman_job/man_job            : 职业
  ivf_adddate                  : 初诊建档时间

▌ woman_first_visit（表2 女方初诊病历）
  age     : 初诊年龄
  zhusu   : 主诉
  bingshi : 现病史
  yjcycle : 月经周期（持续天数/间隔天数）
  fyjage  : 月经初潮年龄
  lmp     : 末次月经日期
  g/p/a   : 妊娠次数/分娩次数/流产次数
  jiwang  : 既往史
  geren   : 个人史
  jiazu   : 家族史
  fuzhu   : 辅助检查（超声卵巢储备情况）
  mh6_5/mh6_6/mh6_7 : 身高(cm)/体重(kg)/BMI
  jihua   : 诊疗计划
  chubu   : 初步诊断

▌ man_first_visit（表3 男方初诊病历）
  age          : 初诊年龄
  zhusu/bingshi: 主诉/现病史
  cgjc         : 初诊精液常规检查结果
  height/weight/bp : 身高/体重/血压
  ma1_1        : 既往病史-有无肝炎
  ma1_2        : 既往病史-有无结核
  ma1_3        : 既往病史-有无肾脏疾病
  ma1_4        : 既往病史-有无心血管疾病
  ma1_5        : 既往病史-有无泌尿系感染
  ma1_6        : 既往病史-其他疾病
  ma1_7        : 既往病史-病因
  ma1_8        : 既往病史-有无勃起功能障碍
  ma1_9        : 既往病史-有无早泄
  ma1_10       : 既往病史-有无慢性前列腺炎
  ma1_11       : 既往病史-有无手术史
  ma2_1        : 个人史-有无吸烟
  ma2_2        : 个人史-有无酗酒
  ma2_3        : 个人史-有无吸毒
  ma2_4        : 个人史-有无习惯用药
  ma2_5        : 个人史-有无药物过敏史
  ma2_6        : 个人史-有无重大精神病史
  ma2_7        : 个人史-过去健康状况
  ma2_8        : 个人史-现在健康状况
  ma2_9        : 个人史-有无出生缺陷
  ma2_10       : 个人史-有无冶游史
  ma3_1/ma3_2  : 是否近亲结婚/是否再婚
  ma4_1/ma4_2  : 家族遗传病史/家族不孕不育史
  ma7_1        : 阴茎长度(cm)
  ma7_2        : 前列腺情况
  ma7_3/ma7_4  : 左/右睾丸体积(ml)
  ma7_5/ma7_6  : 左/右睾丸质地
  ma7_7/ma7_8  : 左/右附睾
  ma7_9/ma7_10 : 左/右输精管
  ma7_11/ma7_12: 左/右侧精索静脉曲张
  ma7_14       : 精索静脉曲张程度
  ma7_13       : 精液情况
  zuizhong     : 最终诊断

▌ cycle_record（表4 女方周期病历）
  pt_id        : 周期编号
  ctype        : 周期类型（新鲜/复苏）
  pt_circle    : 第几个周期
  pt_age1/pt_age2 : 本周期女方/男方年龄
  startdate    : 周期开始日期
  tech         : 新鲜周期-技术方式；复苏周期-内膜准备方案
  coh          : 新鲜周期-COH促排方案；复苏周期-移植胚胎类型
  element1     : 不孕因素-女方因素
  element2     : 不孕因素-男方因素
  element3     : 不孕因素-主要因素
  element4     : 不孕因素-备注
  inage        : 不孕年限（年）
  mh6_1~4     : 体温/脉搏/呼吸/血压
  mh6_5~7     : 身高/体重/BMI
  mh6_8~21    : 营养/发育/精神/毛发/皮肤黏膜/淋巴结/乳房/肾/心/肺/肝/脾/脊柱四肢/其他
  mh7_1~18    : 妇科检查（外阴/阴道/宫颈/纳氏/肥大/子宫位置/大小/质地/活动度/压痛/附件左右/滴虫/霉菌/清洁度/探宫腔/长度/B超）
  mh5_*       : 婚育史（近亲结婚/再婚/妊娠/末次妊娠时间/孕产流各类型）
  mh1_*       : 既往史（肝炎/结核/肾脏/心血管/泌尿系感染/性传播疾病/阑尾炎/盆腔炎/手术史/其他）
  mh2_*       : 个人史（吸烟/酗酒/吸毒/习惯用药/药物过敏/重大精神刺激/健康状况/出生缺陷）
  mh3_1~5     : 月经史（初潮/持续天数/间隔/经量/痛经）
  mh4_1~2     : 家族遗传病史/不孕不育史
  diagnose     : 本周期最终诊断

▌ lab_tests（表5 检验报告）
  itemname  : 检验项目名称
  name1/name2 : 具体指标名称/别名
  result    : 检验结果值
  unit      : 结果单位
  abnormal  : 异常标识（↑偏高 ↓偏低 *异常 空=正常）
  reference3: 参考范围
  sex       : 检测对象性别
  sampletime: 送检时间

▌ exam_reports（表6 检查及妇科手术报告）
  classify     : 报告类型（超声/放射/心电图/手术）
  itemname     : 检查项目或手术名称
  checktime    : 检查/手术时间
  sight        : 检查所见或手术过程
  reportresult : 检查诊断结果或术后诊断

▌ monitoring_records（表7 周期内监测记录）
  adddate      : 监测日期
  mday         : 月经天数
  gnday        : Gn用药第几天
  gnrhday      : GnRH用药第几天
  imsize       : 子宫内膜厚度(mm)
  imtype       : 内膜类型（A型三线征/B型/C型）
  tt           : 尿检结果
  fsh/lh/e2/p/hcg : 内分泌检验值
  leftov_<10   : 左侧直径<10mm卵泡数
  leftov_10-14 : 左侧10-14mm卵泡数
  leftov_15-17 : 左侧15-17mm卵泡数
  leftov_18-22 : 左侧18-22mm卵泡数（成熟卵泡）
  leftov_>22   : 左侧>22mm卵泡数
  rightov_*    : 右侧同上各尺寸卵泡数
  nxtdate/detail: 下次就诊日期/内容
  nt           : 夜针（扳机）时间

▌ medication_records（表8 用药记录）
  name/name2   : 药品名称/别名
  type         : 药品分类（gn=促性腺激素 gnrh=GnRH类）
  contents     : 单位含量
  dose/unit2   : 每日剂量/单位
  freq/way     : 频率/给药途径
  date1/date2  : 用药起止日期

▌ ivf_surgery_records（表9 IVF手术记录）
  optype    : 手术类型（取卵记录/移植记录）
  datetime  : 手术时间
  name      : 手术名称
  mz        : 麻醉方式
  oprecord  : 手术过程记录
  opmemo    : 手术备注

▌ embryo_culture（表10 胚胎培养情况，每行一枚胚胎）
  no           : 胚胎编号
  d01/d02      : Day0评分（卵子成熟度，如MII/MI/GV）
  D0Ending     : Day0结局
  d1           : Day1评分（受精情况：2PN正常/0PN/1PN/多PN异常）
  d2           : Day2评分（卵裂情况）
  d3           : Day3评分（如"841"=8细胞4级1碎片）
  level1       : 卵裂期综合评级（1优质/2良好/3一般/4差/5退化）
  result       : 卵裂期结局（冷冻/继续培养/退化）
  d5/d6/d7     : Day5/6/7囊胚评分（如"4AA"=扩展囊胚AA级）
  level2       : 囊胚期综合评级（1优质/2良好/3可利用/4不可用）
  result2      : 囊胚期结局（冷冻/退化）
  tech         : 技术方式（常规IVF/ICSI）
  type         : 胚胎类型（卵裂期/囊胚期）

▌ embryo_stats（表11 胚胎培养统计）
  opu_date : 取卵时间
  ovum     : 获卵总数
  mii      : MII成熟卵数
  sj       : 总受精数
  ll       : 总卵裂数
  yp       : 优胚数
  d3ly     : D3可利用胚胎数
  d3knp    : D3可培养胚胎数
  d3th     : D3退化胚胎数
  pn       : 正常受精（2PN）数
  npn      : 多PN数
  pn0/pn1  : 0PN/1PN数
  isnp     : 形成囊胚数
  ynp      : 优质囊胚数
  lynp     : 可利用囊胚数
  d5np/d5ynp/d5lynp : D5形成/优质/可利用囊胚数
  d6np/d6ynp/d6lynp : D6形成/优质/可利用囊胚数
  ldll/ldnp: 冷冻卵裂胚数/冷冻囊胚数
  （其余字段为各PN分类统计，命名规则：pn+分类+指标）

▌ embryo_transfer（表12 胚胎移植情况）
  etdate       : 移植时间
  etcount      : 移植胚胎个数
  etypcount    : 移植优胚个数
  emtype       : 移植胚胎属性（冷冻/新鲜）
  em_type1     : 第1胚胎类型（卵裂期/囊胚期）
  etmsg1/etlevel1 : 移植时第1胚胎评分/评级
  d3msg1/d3level1 : 第1胚胎来源D3评分/评级
  npmsg1/nplevel1 : 第1胚胎来源囊胚期评分/评级
  pyday1       : 第1胚胎培养天数
  cohage1      : 第1胚胎来源取卵时女方年龄
  cohem1       : 第1胚胎来源促排方案
  emtech1      : 第1胚胎技术方式
  lddate1/jddate1 : 冷冻日期/解冻日期
  （第2个胚胎字段以2结尾，结构同上）

▌ pregnancy_monitoring（表13 妊娠监控记录）
  adddate  : 监测日期
  rn       : 宫内妊囊数
  tx       : 宫内胎心数
  gw       : 病灶数
  gwsite   : 异位妊娠描述
  us_result: B超结果
  isfinish : 是否毕业
  nxtdate  : 下次就诊日期
  detail   : 就诊内容

══════════════════ 输出格式（严格）══════════════════
必须且只能输出以下 JSON，不得有任何额外文字或 markdown 代码块：
{
  "cot": "详细推理过程",
  "result": "阳性",
  "confidence": 0.75
}
result 只能填写"阳性"或"阴性"，confidence 为 0.0~1.0 之间的浮点数。
"""


def build_input(patient: dict, cycle: dict) -> dict:
    return {
        "basic_info":           patient.get("basic_info", {}),
        "woman_first_visit":    patient.get("woman_first_visit", {}),
        "man_first_visit":      patient.get("man_first_visit", {}),
        "exam_reports":         patient.get("exam_reports", []),
        "cycle_record":         cycle.get("cycle_record", {}),
        "lab_tests":            cycle.get("lab_tests", []),
        "monitoring_records":   cycle.get("monitoring_records", []),
        "medication_records":   cycle.get("medication_records", []),
        "ivf_surgery_records":  cycle.get("ivf_surgery_records", []),
        "embryo_culture":       cycle.get("embryo_culture", []),
        "embryo_stats":         cycle.get("embryo_stats", []),
        "embryo_transfer":      cycle.get("embryo_transfer", {}),
        "pregnancy_monitoring": cycle.get("pregnancy_monitoring", []),
    }


def call_llm(patient_json: str, retries: int = 3) -> Optional[dict]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content":
             f"请根据以下 IVF 患者数据预测本次胚胎移植的临床妊娠结局：\n\n{patient_json}"},
        ],
        "temperature": 0.1,
        "max_tokens":  32768,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                return json.loads(m.group())
            print(f"    ⚠ 未找到 JSON，原始输出: {raw[:100]}")
            return None
        except requests.exceptions.Timeout:
            print(f"    ⚠ 超时 (attempt {attempt}/{retries})")
            time.sleep(10)
        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON 解析失败: {e}")
            return None
        except Exception as e:
            print(f"    ⚠ 调用异常: {e} (attempt {attempt}/{retries})")
            time.sleep(5)

    return None


def main():
    print(f"📂 加载数据: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    samples = []
    for patient in records:
        for cycle in patient.get("cycles", []):
            label = cycle.get("label", {}).get("clinic_preg")
            if cycle.get("embryo_transfer") and label in ("阳性", "阴性"):
                samples.append({
                    "couple_id":  patient["couple_id"],
                    "pt_id":      cycle.get("pt_id"),
                    "true_label": label,
                    "patient":    patient,
                    "cycle":      cycle,
                })

    if MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]

    print(f"🎯 待预测样本: {len(samples)}\n")

    results = []
    for idx, s in enumerate(samples, 1):
        print(f"[{idx:3d}/{len(samples)}] couple={s['couple_id']}  pt_id={s['pt_id']}", end="  ")

        inp = build_input(s["patient"], s["cycle"])
        inp_json = json.dumps(inp, ensure_ascii=False)

        llm_out = call_llm(inp_json)

        if llm_out:
            entry = {
                "couple_id":  s["couple_id"],
                "pt_id":      s["pt_id"],
                "true_label": s["true_label"],
                "cot":        llm_out.get("cot", ""),
                "result":     llm_out.get("result", ""),
                "confidence": llm_out.get("confidence"),
            }
            print(f"✅ result={entry['result']}  conf={entry['confidence']}")
        else:
            entry = {
                "couple_id":  s["couple_id"],
                "pt_id":      s["pt_id"],
                "true_label": s["true_label"],
                "cot":        "",
                "result":     "",
                "confidence": None,
            }
            print("❌ 失败")

        results.append(entry)
        time.sleep(REQUEST_INTERVAL)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 已保存 → {OUT_PATH}  (共 {len(results)} 条)")


if __name__ == "__main__":
    main()