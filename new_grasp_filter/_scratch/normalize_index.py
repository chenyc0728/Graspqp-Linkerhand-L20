import csv

ip = "new_grasp_filter/filtered_grasps/index.csv"
rows = list(csv.DictReader(open(ip, newline="", encoding="utf-8")))
fields = list(rows[0].keys())
for r in rows:
    r["grasp_file"] = r["grasp_file"].replace(chr(92), "/")
with open(ip, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("normalised", len(rows), "rows")
