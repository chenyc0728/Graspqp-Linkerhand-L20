#!/usr/bin/env bash
# P1 全流程：备份索引 -> 同步 object1 mesh -> 重筛 ycb（object / object1）
#             -> 合并校验 -> 转 CMap（v6）-> 终检。
#
# 用法:
#   bash new_grasp_filter/rescreen_all.sh                 # 全量（很久；本机约 3-4 小时）
#   PILOT=1 bash new_grasp_filter/rescreen_all.sh         # 只跑 apple，产出 v6_pilot.pt
#   SKIP_MESH_REPLACE=1 bash ...                          # mesh 已同步过
#   FORCE_REPLACE_NEW_GEOMETRY=1 bash ...                 # 允许替换"新几何"物体（默认跳过）
#   START_STEP=3 bash ...                                 # 断点续跑，从第 3 步开始
#
# 步骤编号（START_STEP=N 从第 N 步开始）:
#   1 索引备份（rescreen_backup.sh，任务 4）
#   2 object1 mesh 同步（rescreen_mesh_replace.sh，任务 1）
#   3 重筛 object/ycb   -> <index-dir>/ycb_object_index.csv
#   4 重筛 object1/ycb  -> <index-dir>/ycb_object1_index.csv
#   5 合并前索引校验（rescreen_merge_check.py，任务 6）
#   6 合并转 CMap（convert_to_cmap.py）-> cmap_dataset_from_graspqp_v6*.pt
#   7 终检（verify_v6_rescreen.py，任务 7）
#
# 环境变量一览见本文件顶部；所有输出同时 tee 到日志文件。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
FG="${HERE}/filtered_grasps"
LOGS="${HERE}/logs"

PILOT="${PILOT:-0}"
SKIP_MESH_REPLACE="${SKIP_MESH_REPLACE:-0}"
FORCE_REPLACE_NEW_GEOMETRY="${FORCE_REPLACE_NEW_GEOMETRY:-0}"
START_STEP="${START_STEP:-1}"
PILOT_OBJECTS="${PILOT_OBJECTS:-apple}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
export TS FORCE_REPLACE_NEW_GEOMETRY

# 注意：grasp 副本与 index*.csv 必须放在**同一个** filtered_grasps 目录里 ——
# convert_to_cmap.iter_pairs() 按 index.csv 的**同级目录**解析行内相对路径
# （P0 约定：行内路径相对 `<output-dir>/filtered_grasps`）。所以重筛用独立的
# output-dir（rescreen[_pilot]/），里面的 filtered_grasps/ 自成一体，
# 既不会覆盖 contactdb 的 filtered_grasps/index.csv，也不用改 convert_to_cmap。
if [[ "${PILOT}" == "1" ]]; then
  OUT_ROOT="${HERE}/rescreen_pilot"
  INDEX_DIR="${OUT_ROOT}/filtered_grasps"
  OUT_PT="${HERE}/cmap_dataset_from_graspqp_v6_pilot.pt"
  RUN_TAG="pilot"
  OUT_LOG="${LOGS}/rescreen_pilot_${TS}.log"
  OBJ_ARGS=(--objects ${PILOT_OBJECTS})
  export MESH_OBJECTS="${MESH_OBJECTS:-${PILOT_OBJECTS}}"
else
  OUT_ROOT="${HERE}/rescreen"
  INDEX_DIR="${OUT_ROOT}/filtered_grasps"
  OUT_PT="${HERE}/cmap_dataset_from_graspqp_v6.pt"
  RUN_TAG="full"
  OUT_LOG="${LOGS}/rescreen_full_${TS}.log"
  OBJ_ARGS=()
fi
mkdir -p "${INDEX_DIR}" "${LOGS}"

# 全部输出进日志（同时保留在终端）
exec > >(tee -a "${OUT_LOG}") 2>&1

CUR_STEP=0
on_error() {
  local rc=$?
  echo ""
  echo "############################################################"
  echo "FAILED at step ${CUR_STEP}（rc=${rc}, $(date '+%Y-%m-%d %H:%M:%S')）"
  echo "已保留中间产物，可手工续跑：START_STEP=${CUR_STEP} TS=${TS} bash new_grasp_filter/rescreen_all.sh"
  echo "日志: ${OUT_LOG}"
  echo "############################################################"
  exit "${rc}"
}
trap on_error ERR

run_step() {
  local n="$1"; shift
  CUR_STEP="${n}"
  if (( n < START_STEP )); then
    echo "[skip] step ${n} (START_STEP=${START_STEP})"
    return 0
  fi
  echo ""
  echo "=== [step ${n}] 开始 $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "=== [step ${n}] 命令: $*"
  "$@"
  echo "=== [step ${n}] 结束 $(date '+%Y-%m-%d %H:%M:%S') ==="
}

# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate graspqp
cd "${REPO}"

echo "============================================================"
echo "P1 rescreen  跑法=${RUN_TAG}  ts=${TS}  start_step=${START_STEP}"
echo "  HERE=${HERE}"
echo "  output_root=${OUT_ROOT}"
echo "  index_dir=${INDEX_DIR}（副本与索引在同一目录，见文件顶部注释）"
echo "  output=${OUT_PT}"
echo "  pilot_objects=${PILOT_OBJECTS}（仅 PILOT=1 生效）"
echo "  SKIP_MESH_REPLACE=${SKIP_MESH_REPLACE} FORCE_REPLACE_NEW_GEOMETRY=${FORCE_REPLACE_NEW_GEOMETRY}"
echo "  git=$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "  python=$(python -V 2>&1)"
echo "  日志=${OUT_LOG}"
echo "============================================================"

# ---------------------------------------------------------------- step 1
step1() { bash "${HERE}/rescreen_backup.sh" --ts "${TS}"; }
run_step 1 step1

# ---------------------------------------------------------------- step 2
step2() {
  if [[ "${SKIP_MESH_REPLACE}" == "1" ]]; then
    echo "[step2] SKIP_MESH_REPLACE=1 -> 跳过 mesh 同步"
    return 0
  fi
  bash "${HERE}/rescreen_mesh_replace.sh"
}
run_step 2 step2

# ---------------------------------------------------------------- step 3
step3() {
  python -u "${HERE}/run_filter.py" \
    --grasp-path object/ycb --source-tag object_rescreen --device cpu "${OBJ_ARGS[@]}" \
    --output-dir "${OUT_ROOT}" \
    --output-index-dir "${INDEX_DIR}" --output-index-prefix ycb_object
}
run_step 3 step3

# ---------------------------------------------------------------- step 4
step4() {
  python -u "${HERE}/run_filter.py" \
    --grasp-path object1/ycb --source-tag object1_rescreen --device cpu "${OBJ_ARGS[@]}" \
    --output-dir "${OUT_ROOT}" \
    --output-index-dir "${INDEX_DIR}" --output-index-prefix ycb_object1
}
run_step 4 step4

# ---------------------------------------------------------------- step 5
CONTACTDB_INDEX="${FG}/index_contactdb_${TS}.csv"
step5() {
  if [[ ! -f "${CONTACTDB_INDEX}" ]]; then
    echo "[error] 找不到 contactdb 索引备份 ${CONTACTDB_INDEX}" >&2
    return 1
  fi
  local extra=()
  if [[ "${PILOT}" == "1" ]]; then
    # shellcheck disable=SC2206
    local objs=(${PILOT_OBJECTS})
    extra=(--required-objects "${objs[@]}")
  fi
  python -u "${HERE}/rescreen_merge_check.py" \
    --indexes "${INDEX_DIR}/ycb_object_index.csv" \
              "${INDEX_DIR}/ycb_object1_index.csv" \
              "${CONTACTDB_INDEX}" \
    --expected-sources object_rescreen object1_rescreen object_contactdb \
    --out "${LOGS}/rescreen_merge_check_${RUN_TAG}.json" "${extra[@]}"
}
run_step 5 step5

# ---------------------------------------------------------------- step 6
step6() {
  python -u "${HERE}/convert_to_cmap.py" \
    --inputs "${INDEX_DIR}/ycb_object_index.csv" \
             "${INDEX_DIR}/ycb_object1_index.csv" \
             "${CONTACTDB_INDEX}" \
    --prefixes ycb ycb contactdb \
    --output "${OUT_PT}"
}
run_step 6 step6

# ---------------------------------------------------------------- step 7
step7() {
  python -u "${HERE}/verify_v6_rescreen.py" \
    --dataset "${OUT_PT}" \
    --indexes "${INDEX_DIR}/ycb_object_index.csv" \
              "${INDEX_DIR}/ycb_object1_index.csv" \
              "${CONTACTDB_INDEX}" \
    --prefixes ycb ycb contactdb \
    --baseline "${HERE}/cmap_dataset_from_graspqp_v3.pt" \
    --jsonl-dir "${OUT_ROOT}/logs" \
    --out "${LOGS}/verify_v6_rescreen_${RUN_TAG}.json"
}
run_step 7 step7

echo ""
echo "============================================================"
echo "全部完成 $(date '+%Y-%m-%d %H:%M:%S')"
echo "  数据集: ${OUT_PT}"
echo "  索引:   ${INDEX_DIR}/ycb_object_index.csv, ycb_object1_index.csv"
echo "  contactdb 索引: ${CONTACTDB_INDEX}（备份自重筛前的 filtered_grasps/index.csv）"
echo "  校验:   ${LOGS}/rescreen_merge_check_${RUN_TAG}.json, ${LOGS}/verify_v6_rescreen_${RUN_TAG}.json"
echo "  日志:   ${OUT_LOG}"
echo "============================================================"
