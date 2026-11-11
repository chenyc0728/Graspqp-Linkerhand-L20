#!/usr/bin/env bash
# P1 任务 1：把 object/ycb 里 v3 修好的 mesh 同步到 object1/ycb（先备份，再替换，再核对）。
#
# 真正的逻辑在 rescreen_mesh_replace.py（需要 trimesh 做几何对比），本脚本只负责：
#   * 固定环境（conda graspqp）；
#   * 尊重 FORCE_REPLACE_NEW_GEOMETRY / TS 环境变量；
#   * 把 stdout 同时留在 logs/rescreen_mesh_replace_<ts>.log。
#
# 环境变量：
#   FORCE_REPLACE_NEW_GEOMETRY=1  允许替换"object1 不是修复前原件"的物体（默认跳过并报告）
#   MESH_OBJECTS="apple sponge"   只处理这些物体（默认全部 28 个）
#   TS=20261110_120000            统一时间戳（由 rescreen_all.sh 传入）
#
# 用法：bash new_grasp_filter/rescreen_mesh_replace.sh [--dry-run] [--no-geom]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
LOGS="${HERE}/logs"
mkdir -p "${LOGS}"

# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate graspqp

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOGS}/rescreen_mesh_replace_${TS}.log"
FORCE_ARGS=()
if [[ "${FORCE_REPLACE_NEW_GEOMETRY:-0}" != "0" && -n "${FORCE_REPLACE_NEW_GEOMETRY:-}" ]]; then
  FORCE_ARGS+=(--force-new-geometry)
  echo "[mesh] FORCE_REPLACE_NEW_GEOMETRY=${FORCE_REPLACE_NEW_GEOMETRY} -> 允许替换 new_geometry 物体"
fi
OBJ_ARGS=()
if [[ -n "${MESH_OBJECTS:-}" ]]; then
  # shellcheck disable=SC2206
  OBJ_ARGS+=(--objects ${MESH_OBJECTS})
  echo "[mesh] MESH_OBJECTS=${MESH_OBJECTS}"
fi

echo "[mesh] ts=${TS} log=${LOG}"
cd "${REPO}"
python -u "${HERE}/rescreen_mesh_replace.py" --ts "${TS}" "${FORCE_ARGS[@]}" "${OBJ_ARGS[@]}" "$@" \
  2>&1 | tee "${LOG}"
echo "[mesh] 完成，日志 ${LOG}"
