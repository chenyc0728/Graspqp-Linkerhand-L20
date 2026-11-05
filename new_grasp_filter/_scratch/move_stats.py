from pathlib import Path

p = Path("new_grasp_filter/REPORT.md")
s = p.read_text(encoding="utf-8")

begin = "<!-- STATS:BEGIN -->"
end = "<!-- STATS:END -->"
i = s.index(begin)
j = s.index(end) + len(end)
block = s[i:j]
# remove block and the now-empty placeholder line above it
s_without = s[:i] + s[j:]
s_without = s_without.replace("（见本节末尾自动生成的统计表）\n\n", "")

marker = "## 4. 转换为 CMapDataset（阶段 3）"
assert marker in s_without
s_new = s_without.replace(marker, block + "\n\n---\n\n" + marker, 1)
p.write_text(s_new, encoding="utf-8")
print("moved stats block to end of section 3")
