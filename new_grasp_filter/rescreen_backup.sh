#!/usr/bin/env bash
# P1 任务 4：把"会被重筛覆盖"的索引先备份出来（放在 rescreen_all.sh 的最开头）。
#
#   filtered_grasps/index.csv          -> filtered_grasps/index_contactdb_<ts>.csv
#   filtered_grasps/index_grasps.csv   -> filtered_grasps/index_grasps_contactdb_<ts>.csv
#   filtered_grasps/index_ycb.csv      -> filtered_grasps/index_ycb_v1polluted_<ts>.csv
#   filtered_grasps/index_ycb_grasps.csv -> filtered_grasps/index_ycb_grasps_v1polluted_<ts>.csv
#
# 为什么：run_filter.py 每次都重写 filtered_grasps/index.csv / index_grasps.csv（HANDOFF §7-14），
# 而当前那两份是 **contactdb 的唯一索引**（source=object_contactdb，2911 条可行）；
# index_ycb*.csv 是 v1 污染的历史索引（merged_from 指向 index_v1_object.csv），只留档、不再使用。
#
# 行为：
#   * 备份后逐个比对 sha256，任何不一致 -> 退出码 1（配合 set -e 让上层立即停）；
#   * 可重复执行：已有同内容备份时不再复制（幂等）；
#   * 时间戳通过 TS 环境变量传入（rescreen_all.sh 用它保证全流程同一个 ts），
#     也会写到 new_grasp_filter/logs/rescreen_ts.txt 供后续步骤读取。
#
# 用法：bash new_grasp_filter/rescreen_backup.sh [--ts 20261110_120000] [--dry-run]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FG="${HERE}/filtered_grasps"
LOGS="${HERE}/logs"
mkdir -p "${LOGS}"

TS="${TS:-}"
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ts) TS="$2"; shift 2 ;;
    --ts=*) TS="${1#*=}"; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "[error] 未知参数: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${TS}" ]] || TS="$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "${TS}" > "${LOGS}/rescreen_ts.txt"

sha_of() { sha256sum "$1" | awk '{print $1}'; }

# 原始文件 : 备份文件 : 是否必须存在 (1/0)
PAIRS=(
  "${FG}/index.csv:${FG}/index_contactdb_${TS}.csv:1"
  "${FG}/index_grasps.csv:${FG}/index_grasps_contactdb_${TS}.csv:1"
  "${FG}/index_ycb.csv:${FG}/index_ycb_v1polluted_${TS}.csv:0"
  "${FG}/index_ycb_grasps.csv:${FG}/index_ycb_grasps_v1polluted_${TS}.csv:0"
)

echo "[backup] ts=${TS} dry_run=${DRY} dir=${FG}"
FAILED=0
RECORDS=""
for item in "${PAIRS[@]}"; do
  IFS=':' read -r SRC DST REQUIRED <<< "${item}"
  NAME="$(basename "${SRC}")"
  if [[ ! -f "${SRC}" ]]; then
    if [[ "${REQUIRED}" == "1" ]]; then
      echo "[error] 缺少必需索引 ${SRC}" >&2
      FAILED=1
    else
      echo "[warn] ${NAME} 不存在（可能已备份过）：跳过"
    fi
    RECORDS="${RECORDS}{\"src\":\"${SRC}\",\"dst\":\"${DST}\",\"status\":\"missing\",\"required\":${REQUIRED}},"
    continue
  fi
  SRC_SHA="$(sha_of "${SRC}")"
  SRC_ROWS="$(wc -l < "${SRC}")"
  STATUS="copied"
  if [[ -f "${DST}" ]]; then
    if [[ "$(sha_of "${DST}")" == "${SRC_SHA}" ]]; then
      STATUS="already_present"
    else
      echo "[error] 目标已存在但内容不同：${DST}" >&2
      FAILED=1
      STATUS="conflict"
    fi
  elif [[ "${DRY}" == "1" ]]; then
    STATUS="dry_run"
  else
    cp -p "${SRC}" "${DST}"
  fi
  if [[ "${STATUS}" == "copied" || "${STATUS}" == "already_present" ]]; then
    DST_SHA="$(sha_of "${DST}")"
    if [[ "${DST_SHA}" != "${SRC_SHA}" ]]; then
      echo "[error] sha256 校验失败 ${DST}" >&2
      FAILED=1
      STATUS="sha_mismatch"
    fi
  else
    DST_SHA=""
  fi
  echo "[backup] ${NAME} -> $(basename "${DST}")  ${STATUS}  rows=${SRC_ROWS}  sha=${SRC_SHA:0:12}"
  RECORDS="${RECORDS}{\"src\":\"${SRC}\",\"dst\":\"${DST}\",\"status\":\"${STATUS}\",\"required\":${REQUIRED},\"rows\":${SRC_ROWS},\"sha256_src\":\"${SRC_SHA}\",\"sha256_dst\":\"${DST_SHA}\"},"
done

python3 - "${LOGS}/rescreen_backup.json" "${TS}" "${FAILED}" "[${RECORDS%,}]" <<'PY'
import json, sys, time
out, ts, failed, records = sys.argv[1], sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4])
json.dump({"task": "P1-task4-index-backup", "ts": ts, "failed": bool(failed),
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "records": records},
          open(out, "w"), indent=2, ensure_ascii=False)
print(f"[write] {out}")
PY

if [[ "${FAILED}" != "0" ]]; then
  echo "[fatal] 索引备份失败，停止（后续重筛会覆盖 contactdb 索引，绝不能带病继续）" >&2
  exit 1
fi
echo "[done] 索引备份完成（ts=${TS}）"
