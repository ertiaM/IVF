import json

INPUT_PATH  = "/aifs4su/zhuhan/chenjiale/AI4Health/full_test/medical_records.json"
OUTPUT_PATH = "/aifs4su/zhuhan/chenjiale/AI4Health/full_test/protein_records.json"

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

results = []

for record in data:
    couple_id = record.get("couple_id")
    protein_items = []

    for cycle in record.get("cycles", []):
        for lab in cycle.get("lab_tests", []):
            name1 = lab.get("name1") or ""
            if "蛋白" in name1:
                protein_items.append({
                    "pt_id":      lab.get("pt_id"),
                    "hisid":      lab.get("hisid"),
                    "sex":        lab.get("sex"),
                    "itemname":   lab.get("itemname"),
                    "name1":      lab.get("name1"),
                    "name2":      lab.get("name2"),
                    "result":     lab.get("result"),
                    "unit":       lab.get("unit"),
                    "abnormal":   lab.get("abnormal"),
                    "reference3": lab.get("reference3"),
                    "sampletime": lab.get("sampletime"),
                    "labtime":    lab.get("labtime"),
                })

    results.append({
        "couple_id":     couple_id,
        "protein_tests": protein_items,
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Done. {len(results)} records written to {OUTPUT_PATH}")
total = sum(len(r["protein_tests"]) for r in results)
print(f"Total protein test entries: {total}")
