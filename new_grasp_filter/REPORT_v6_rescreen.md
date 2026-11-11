# REPORT_v6_rescreen —— 修好"ycb 假阳性 + object1 mesh 未同步"，重筛两批 ycb 并合并出新 CMap 数据集

> P1 轮次交付说明。**本轮按要求没有跑全量重筛**：交付的是经过 pilot 实测跑通的可执行脚本 + pilot 实测结果 + 全量跑法说明。
> 全量请自己在后台跑：`bash new_grasp_filter/rescreen_all.sh`（预估 ~70 分钟，见 §6）。
>
> 环境：conda `graspqp`（python 3.11.18 / torch 2.8.0+cu129 / trimesh 4.8.0 / numpy 1.26.4 / sapien 3.0.3），
> git `c785f881fc3cd6678debb6d52a68f3925f402ff2`，本机 `shellcheck` **未安装**（用 `bash -n` 代替，见 §6.3）。

---

## 0. 状态总览

| 任务 | 内容 | 状态 | 证据/产物 |
|---|---|---|---|
| 0 | 预检（到底哪里坏了） | ✅ 真跑 3.5 s | `logs/rescreen_preflight.json`（275 KB） |
| 1 | object1/ycb mesh 同步 v3 修复 | ✅ 真替换 8 个物体 | `logs/rescreen_mesh_replace*.{json,log}`、`object1/ycb/_mesh_backup_20261110_p1/` |
| 2 | `run_filter.py` 加新参数（不隐式继承旧索引） | ✅ 改完 + 逐字节兼容性回归 | `/tmp/bc_old` vs `/tmp/bc_new` 完全一致 |
| 3 | pilot 重筛 apple（object + object1） | ✅ 真跑 143 s | `rescreen_pilot/filtered_grasps/`、`rescreen_pilot/logs/` |
| 4 | 索引备份（防止 contactdb 结果被覆盖） | ✅ 真跑 + sha256 校验 | `logs/rescreen_backup.json` |
| 5 | `rescreen_all.sh` 全量编排 | ✅ 写好 + `bash -n` 通过 + PILOT 端到端跑通 | `new_grasp_filter/rescreen_all.sh` |
| 6 | 合并前三份索引校验 | ✅ 真跑 | `logs/rescreen_merge_check_pilot.json` |
| 7 | 新数据集终检 | ✅ 6 项全 OK | `logs/verify_v6_rescreen_pilot.json` |
| 8 | `PILOT=1 bash rescreen_all.sh` 端到端 | ✅ 一条命令跑完 1→7 | `logs/rescreen_pilot_20261110_224741.log` |
| 9 | 本报告 + 交付清单 | ✅ | 本文件 |

**没有做的事**（按要求）：没跑全量重筛；没动 contactdb 的可行抓取结果；没动 v1–v4 的 `.pt`/`.meta.json`/REPORT；
没改 `grasp_io.py` / `sapien_grasp_checker.py` / `object_sdf.py` 的数值逻辑；没为了让 pilot 好看改任何阈值/判据；
没有伪造任何数字（pilot 的每一个数都来自 `run_filter.py` 的真实输出）。

---

## 1. 任务 0：预检结论 —— 坏在哪两处

`python new_grasp_filter/rescreen_preflight.py` → `logs/rescreen_preflight.json`（3.5 s，纯只读）。
它把"怀疑"变成"证据"：

### 1.1 物体集合没问题

`a_object_lists`：`object/ycb` 与 `object1/ycb` 都是 **28 个物体、集合完全相同**，没有单边物体。

### 1.2 mesh 同步问题：8 个物体（不是 28 个），且**安全**

`object/<obj>/<obj>.stl` vs `object1/<obj>/<obj>.stl` vs `object/_stl_backup_*/ycb/<obj>/<obj>.stl`
三方 sha256 对比（`b_per_object` / `b_relation_counts`，**替换前**的状态）：

* `identical`（两边本来就一样，不用动）：**20** 个；
* `object1_is_pristine_backup`（object1 还是 v3 修复前的原件，**安全替换**）：**8** 个 →
  `gelatin_box, mini_soccer_ball, mustard_bottle, pear, power_drill, sponge, tomato_soup_can, tuna_fish_can`
  （与 `c_v3_fixed_meshes.objects_whose_object_stl_differs_from_backup` 完全一致，两条独立证据互相印证）；
* `new_geometry`（object1 是第三种几何，**默认拒绝替换**）：**0** 个。

所以"object1 mesh 没同步 v3 修复"是真的，但**只影响 8 个物体**，而且因为这 8 个的 object1 恰好是"修复前原件"，
`object/_stl_backup_*/` 里的备份能证明这一点 —— 替换是安全的、可回滚的。

> 顺带记录一个坑：URDF 解析一开始报
> `ParseError: XML or text declaration not at start of entity: line 2, column 0` —— 这些 URDF 第一行是空行，
> `ElementTree.parse()` 直接炸。修法是先 `text.lstrip()`。修好后 `d_summary` 的结论才可信（见 §1.3）。

### 1.3 URDF：28/28 两边相同且全是相对路径 → 只换 mesh 就够

修好解析后 `d_summary`：`n_urdf_identical=28`、`object1_all_urdf_relative=true`，结论
"URDF 全部是相对路径且两边相同 -> 只替换 .stl/.obj 就够，不需要动 URDF"。
（apple 的 URDF 430 字节，`mesh filename="apple.stl"`，两边 sha256 都是 `3c379ed7…`。）

### 1.4 `object1` 不是 `object` 的副本，是**另一次独立的预测**（重要，影响怎么解读结果）

一开始只比 sha256 得出"15 个 `*.dexgrasp.pt` 两边全都不一样"，这没意义 —— `torch.save` 的 zip 容器不逐字节可复现。
所以补了**语义比较**（`e_grasp_predictions.*.semantic`，逐张量 `torch.equal`，含 `parameters.root_pose` 和 16 个关节角）：

| 物体 | 文件 | `parameters.root_pose` max&#124;d&#124; | `values` max&#124;d&#124; | 内容相同? |
|---|---|---|---|---|
| apple | `apple.dexgrasp.pt` | **1.384** | 15.98 | 否 |
| baseball | `baseball.dexgrasp.pt` | 1.379 | 41.90 | 否 |

即：`object1/ycb` 的抓取**根本不是** `object/ycb` 的同一批抓取，而是**另一次独立生成的预测**
（同物体、同目录结构、同文件数 15/15，但 pose 与关节角都不同）。
这条结论直接决定了两件事：
① 重筛必须**两批分别做**（不能拿一批的结果套另一批）；
② `ycb_object` 与 `ycb_object1` 合并时**不会互相去重**（实测 `dropped_duplicates=0`）。

### 1.5 索引：v1 污染就写在文件里

`f_indexes` / `f_apple_example`：

| 索引 | 行数 | `source` 分布 | `merged_from` |
|---|---|---|---|
| `filtered_grasps/index.csv` | 600 | `object_contactdb` ×600 | 全空 ✅ |
| `filtered_grasps/index_grasps.csv` | 2911 | `object_contactdb` | — |
| `filtered_grasps/index_ycb.csv` | 420 | `object` ×420 / `object1` ×420（分表 840 行） | **`object` 那 420 行全部指向 `index_v1_object.csv`**（v1 污染） |
| `filtered_grasps/index_ycb_grasps.csv` | 3429 | `object` 2115 / `object1` 1314 | — |

apple 的具体证据（v3 里 apple #3 就是 REPORT_v2 §2.3 点名的 19.1 mm 假阳性）：

```
老索引(污染)  source=object   feasible_indices = 3 4 7 13 16 20 27   reason_counts={"ok":7,"penetration":17,"slips_on_lift":8}
                              merged_from = …\filtered_grasps\index_v1_object.csv
重筛后        source=object_rescreen          feasible_indices = 7 27        ← #3/#4/#13/#16/#20 全部消失
```

### 1.6 几何：替换前 8 个 object1 STL 非水密

`geom`（替换前）：28 个 object STL 全部水密；object1 侧有 8 个非水密；`volume_ratio(object1/object) = 1.00011–1.00075`
（两边是同一物体的不同修复版本，体积差 0.01%–0.08%）。

---

## 2. 任务 1：object1/ycb mesh 同步（真替换，可回滚）

`bash new_grasp_filter/rescreen_mesh_replace.sh`（Python 逻辑在 `rescreen_mesh_replace.py`）。

**安全策略**（不是无脑 copy）：

1. **先全量备份**：把 `object1/ycb` 下所有 `*.stl/*.obj/*.ply/*.urdf`（含子目录）复制到
   `object1/ycb/_mesh_backup_20261110_p1/`（**112 个文件，41 MB**），逐文件 sha256 复核后才继续；
2. **按三方关系决定动作**：`identical` 跳过 / `object1_is_pristine_backup` 替换 /
   `new_geometry`·`no_backup_to_compare` **默认跳过**（要替换必须显式 `FORCE_REPLACE_NEW_GEOMETRY=1`，会标 `forced`）；
3. object1 独有的几何文件**不删不改**，只在 JSON 里列出来；
4. URDF 只在两边内容不同时复制（实测 28/28 相同 → 一个都没复制）；
5. 替换后逐文件核对 sha256 == object 侧，不一致直接 `SystemExit`。

**结果**（`logs/rescreen_mesh_replace_replace_run.json` = 真正做替换那一次的记录；`rescreen_mesh_replace.json` = 之后跑幂等复跑的记录）：

| 物体 | 关系 | 动作 | 水密 before→after | 体积比 | Chamfer 平均 | SDF 最大差 |
|---|---|---|---|---|---|---|
| gelatin_box | pristine_backup | replace | **False→True** | 0.999556 | 0.0162 mm | 1.92 mm |
| mini_soccer_ball | pristine_backup | replace | **False→True** | 0.999642 | 0.0269 mm | 1.59 mm |
| mustard_bottle | pristine_backup | replace | **False→True** | 0.999411 | 0.0186 mm | 2.03 mm |
| pear | pristine_backup | replace | **False→True** | 0.999516 | 0.0201 mm | 1.46 mm |
| power_drill | pristine_backup | replace | **False→True** | 0.999249 | 0.0543 mm | 2.30 mm |
| sponge | pristine_backup | replace | True→True（几何仍不同） | 0.9996 | 0.0191 mm | — |
| tomato_soup_can | pristine_backup | replace | **False→True** | 0.999764 | 0.0216 mm | 3.12 mm |
| tuna_fish_can | pristine_backup | replace | **False→True** | 0.999788 | 0.0237 mm | 2.04 mm |
| 其余 20 个 | identical | skip | True→True | 1.0 | — | 0.0 |

* **SDF 对比用的是 checker 自己的 `object_sdf.VoxelSDF`**（pitch 2 mm，随机 2000 点，边界修正版）：
  符号分歧率 0.001–0.004，说明修复改变了极少量体素的内外判定 —— 这正是"非水密网格让 SDF 不可靠"的具体体现；
* **Chamfer 有数值底噪**：相同 mesh 互比也不是 0（实测 mean 0.0041 mm / max 0.21 mm，`chamfer_self_noise_floor` 已写进 JSON）。
  上表 8 个的 Chamfer 平均 0.016–0.054 mm 明显高于底噪，power_drill 的 max 2.7 mm 更是明确的真实差异；
* 复跑幂等：再跑一次 28 个全部 `identical->skip`，不会再动任何文件；
* `FORCE_REPLACE_NEW_GEOMETRY` 已实现，但本仓库没有 `new_geometry` 物体，所以实测是 no-op（如实报告）。

---

## 3. 任务 2：`run_filter.py` 的改动（只加能力，不动判据）

### 3.1 新增的东西

| 需求 | 实现 |
|---|---|
| 索引不许写到默认位置 | `--output-index-dir`（默认仍是 `<output-dir>/filtered_grasps`，老行为不变） |
| 索引文件名要能区分 | `--output-index-prefix`（校验 `[A-Za-z0-9_-]`，非法值直接退：`--output-index-prefix 只允许 [A-Za-z0-9_-]：'../x'`） |
| 只筛指定物体 | `--objects nargs="+"` |
| **绝不隐式继承旧判定** | 默认 `merge_from=None`；`--no-merge` 显式声明；两者互斥（`--no-merge 与 --merge-from 互斥：重筛必须是全新筛选`）。启动时打印 `[fresh] merge_from=None -> 全新筛选…` |
| 可追溯 | 启动 banner 一行 JSON：`grasp_path / source_tag / merge_from / merge_from_is_none / output_dir / index_dir / index_prefix / objects_filter / object_filter / grasp_filter / limit / max_files_per_object / backend / device / physx_gpu / deps{git_commit,python,torch,trimesh,numpy,sapien} / started` |
| 汇总更全 | `summary["banner"]`、`summary["total"]["merged_from_rows"]`、`summary["finished"]` |

**判据/阈值一个都没改**（`contact_dist=2 mm`、`min_contact_points=3`、`min_contact_links=2`、lift 判据等原样）。

### 3.2 兼容性证明（真跑，逐字节）

用 **handoff 里的老命令形态**分别跑"改动前的脚本副本"和"现在的脚本"，产物逐字节 `cmp`：

```
$ python new_grasp_filter/_run_filter_orig_tmp.py --object apple --grasp apple.dexgrasp.pt --limit 4 --device cpu --output-dir /tmp/bc_old
$ python new_grasp_filter/run_filter.py          --object apple --grasp apple.dexgrasp.pt --limit 4 --device cpu --output-dir /tmp/bc_new
IDENTICAL filtered_grasps/index.csv          (2 行, sha256 f8eb40e1eb6fe10b…)
IDENTICAL filtered_grasps/index_grasps.csv   (1 行, sha256 1ba9bd0ad75bc8c2…)
```

新参数也用真跑验过：

```
$ … --output-index-dir /tmp/bc_pref/idx --output-index-prefix ycb_object --no-merge
[fresh] merge_from=None -> 全新筛选，不继承任何旧判定（merged_from 列将全为空）
[index] wrote /tmp/bc_pref/idx/ycb_object_index.csv (1 rows) …
rows: 1  sources: {'object'}  merged_from 非空行数: 0
```

> `_run_filter_orig_tmp.py` 只是这一轮做 A/B 用的临时副本，交付前已删除。

---

## 4. 任务 3 + 8：pilot 重筛（真实运行，非模拟）

一条命令：

```bash
PILOT=1 bash new_grasp_filter/rescreen_all.sh
# → 日志 logs/rescreen_pilot_20261110_224741.log
```

步骤 1→7 全绿（第一次跑在 step 5 抓到了 `rescreen_merge_check.py` 自己的两个 bug，修好后重跑 OK，见 §7 的"踩坑"）。

### 4.1 结果（apple，两个来源各 15 个文件 × 32 抓取 = 480 条）

| 来源 | 命令里的 tag | 抓取数 | 可行 | 耗时 | 每抓取 | 拒绝理由分布 |
|---|---|---|---|---|---|---|
| `object/ycb`（v3 修好的 mesh） | `object_rescreen` | 480 | **15** | 70.5 s | 0.147 s | penetration 244 / insufficient_contacts 180 / slips_on_lift 41 / ok 15 |
| `object1/ycb`（同步 mesh 之后） | `object1_rescreen` | 480 | **57** | 73.0 s | 0.152 s | penetration 237 / insufficient_contacts 140 / slips_on_lift 46 / ok 57 |

两批的 `merged_from` 列 **全空**（`merged_from_rows: 0`），`source` 分别是 `object_rescreen` / `object1_rescreen`。

apple 单文件对比（`apple.dexgrasp.pt`）：

| 来源 | 老索引（v1 污染 / 未同步 mesh） | 重筛后 |
|---|---|---|
| object | 7 条：`3 4 7 13 16 20 27` | 2 条：`7 27` |
| object1 | 4 条：`6 12 13 15` | 5 条：`3 6 12 13 15` |

### 4.2 v5 抽查出的 16 条 `insufficient_contacts` 假阳性，现在怎么样了？

`verify_v6_rescreen.py` 第 6 项会拿 `logs/manual_review_pilot/audit_feasible_recheck.json` 里的 16 条逐个反查新数据集
（按 `(文件 sha256, grasp_index)` 匹配，另附新筛选日志里的**新判据**）：

```
[false-positive] v5 抽查出的 16 条 insufficient_contacts：仍在新数据集 0 条，已剔除 3 条，空转 13 条
  ycb+apple  apple_step_1500.dexgrasp.pt  idx= 29 gap=52.5mm evidence=removed_by_rescreen new_reason=insufficient_contacts
  ycb+apple  apple_step_4000.dexgrasp.pt  idx=  4 gap= 8.2mm evidence=removed_by_rescreen new_reason=insufficient_contacts
  ycb+apple  apple_step_5000.dexgrasp.pt  idx= 29 gap=13.0mm evidence=removed_by_rescreen new_reason=insufficient_contacts
  （其余 13 条：evidence=vacuous_object_not_in_dataset）
```

**必须如实说清楚**：pilot 只重筛了 apple，所以另外 13 条（baseball / cracker_box / pear / sugar_box / …）之所以
"不在新数据集里"，是因为它们的物体压根不在这个 pilot 数据集里 —— 这叫**空转，不是证据**。
脚本已经把它标成 `vacuous_object_not_in_dataset` 而不是含糊地说"已剔除"。
真正的证据只有 apple 那 3 条：它们的**新判据仍然是 `insufficient_contacts`**（说明它们本来就是烂抓取，
v1 却把它们当可行留下来了），现在被正确剔除。
**剩下 13 条要等全量重筛跑完才能给结论**（那时同一份检查会自动变成 `removed_by_rescreen`）。

---

## 5. 任务 4：索引备份（contactdb 结果不会被覆盖）

`bash new_grasp_filter/rescreen_backup.sh --ts <ts>`（在 `rescreen_all.sh` 的 **step 1**）：

```
[backup] index.csv            -> index_contactdb_20261110_224741.csv          copied rows=601  sha=d466ff19662d
[backup] index_grasps.csv     -> index_grasps_contactdb_20261110_224741.csv   copied rows=2912 sha=139cf2a84520
[backup] index_ycb.csv        -> index_ycb_v1polluted_20261110_224741.csv     copied rows=841  sha=2deb52efa1a5
[backup] index_ycb_grasps.csv -> index_ycb_grasps_v1polluted_20261110_224741.csv copied rows=3430 sha=6f1bbbdb2cf2
[done] 索引备份完成
```

* 每个文件 `cp -p` 后立刻比 sha256，不一致 → `exit 1`（配合 `set -euo pipefail` 让上层立刻停）；
* 幂等：目标已存在且内容相同 → `already_present`，内容不同 → `conflict` 并失败（不会静默覆盖）；
* 时间戳通过 `TS` 变量在全流程共享（同时写 `logs/rescreen_ts.txt`）；
* **跑完 pilot 之后复查**：`filtered_grasps/index.csv` 的 sha256 仍是 `d466ff19…`，与备份逐字节相同
  → contactdb 的 600 行 / 2911 条可行**一条没动**。

---

## 6. 任务 5：`rescreen_all.sh` —— 全量怎么跑

### 6.1 跑法

```bash
cd /home/chenyc/codes/graspqp
bash new_grasp_filter/rescreen_all.sh                 # 全量，约 70 分钟（见 6.2）
PILOT=1 bash new_grasp_filter/rescreen_all.sh         # 只跑 apple，产出 v6_pilot.pt（已实测 143 s）
```

环境变量：

| 变量 | 默认 | 含义 |
|---|---|---|
| `PILOT` | `0` | `1` = 只跑 `PILOT_OBJECTS`（默认 `apple`），产物进 `rescreen_pilot/` + `cmap_dataset_from_graspqp_v6_pilot.pt` |
| `PILOT_OBJECTS` | `apple` | pilot 的物体列表（可空格分隔多个） |
| `SKIP_MESH_REPLACE` | `0` | `1` = 跳过 step 2（本机 mesh 已同步过，可跳过） |
| `FORCE_REPLACE_NEW_GEOMETRY` | `0` | `1` = 允许替换 `new_geometry`/`no_backup` 的物体（本仓库无此物体，实测 no-op） |
| `START_STEP` | `1` | 断点续跑，从第 N 步开始（见 6.4） |
| `TS` | 自动 | 全流程共用时间戳（决定备份名/日志名） |

7 个步骤：`1 索引备份 → 2 mesh 同步 → 3 重筛 object/ycb → 4 重筛 object1/ycb → 5 合并前校验 → 6 转 CMap → 7 终检`。
全部输出 `tee` 到 `logs/rescreen_full_<ts>.log`（PILOT 时是 `logs/rescreen_pilot_<ts>.log`），每步打印开始/结束时间戳；
任何一步失败会打印

```
FAILED at step N（rc=…）
已保留中间产物，可手工续跑：START_STEP=N TS=<ts> bash new_grasp_filter/rescreen_all.sh
```

### 6.2 全量耗时预估（按 pilot 实测外推，不是拍脑袋）

* 全树抓取数：`object/ycb` 28 物体 / 420 文件 / **13440 抓取**（`object1` 相同，共 26880）；
* pilot 实测每抓取 0.1495 s（CPU，单进程）；
* ⇒ 单来源 ≈ 33.5 min，两个来源 ≈ **67 min**；加 mesh 同步(~1 min，已同步则 0) + 校验/合并(~1 min)
  ⇒ **全量约 70 分钟**（`--device cpu`）。
* 想更快：`--device cuda` 只影响 torch FK 后端，SAPIEN 碰撞检测仍是 CPU（`--gpu` 是 PhysX GPU 尝试，默认关）。
  **本轮没有验证 GPU 路径**，所以默认就是 CPU。

### 6.3 shell 静态检查

本机 **没有 shellcheck**（`command -v shellcheck` 为空）。用 bash 自带的语法检查代替：

```
$ bash -n new_grasp_filter/rescreen_backup.sh      OK
$ bash -n new_grasp_filter/rescreen_mesh_replace.sh OK
$ bash -n new_grasp_filter/rescreen_all.sh          OK
```

脚本里所有变量都加了引号、数组用 `"${ARR[@]}"`、`[[ ]]` + `set -euo pipefail`、`trap ERR` 报步号。
如果之后装了 shellcheck，建议跑：`shellcheck new_grasp_filter/rescreen_*.sh`。

### 6.4 断点续跑

`START_STEP` 单独存在就是为了这个：比如 step 4 崩了，修好后

```bash
START_STEP=4 TS=<原来那次的 ts> bash new_grasp_filter/rescreen_all.sh
```

会跳过 1–3（不重新备份、不重新筛 object），直接重跑 `object1`。所有中间产物都留在原地，不会被清掉。

### 6.5 产物位置（为什么不是 `filtered_grasps/rescreen_pilot/`）

重筛的副本和索引放在**独立的 rescreen root** 里：

```
new_grasp_filter/rescreen_pilot/filtered_grasps/ycb_object_index.csv          ← 15 行
new_grasp_filter/rescreen_pilot/filtered_grasps/ycb_object1_index.csv         ← 15 行
new_grasp_filter/rescreen_pilot/filtered_grasps/ycb_object_index_grasps.csv   ← 15 行（逐抓取）
new_grasp_filter/rescreen_pilot/filtered_grasps/{object_rescreen,object1_rescreen}/<obj>/…/*.pt
new_grasp_filter/rescreen_pilot/logs/{object_rescreen,object1_rescreen}/{summary.json,<obj>.jsonl}
```

原因：`convert_to_cmap.iter_pairs()` 按 **index.csv 的同级目录**解析行内相对路径
（P0 约定：行内路径相对 `<output-dir>/filtered_grasps`）。把索引塞进 `filtered_grasps/rescreen_pilot/` 子目录会让
`object_rescreen/apple/…` 解析到不存在的路径、所有样本被 `missing_grasp_files` 丢掉。
用独立 root 既隔离了 contactdb 索引，又**不用改 `convert_to_cmap.py` 一行逻辑**。

---

## 7. 任务 6：合并前索引校验（`rescreen_merge_check.py`）

对三份索引（`ycb_object` / `ycb_object1` / contactdb 备份）逐项查：`source` 是否对、`merged_from` 是否全空、
每行副本是否存在且 sha256 一致、`sum(n_feasible)` 是否等于逐抓取索引行数、`feasible_indices` 是否自洽、
跨来源 `(sha256, grasp_index)` 是否重复。

```
[OK] ycb_object_index.csv    source=object_rescreen  rows=15  sum_feasible=15  grasp_rows=15 objects=1
[OK] ycb_object1_index.csv   source=object1_rescreen rows=15  sum_feasible=57  grasp_rows=57 objects=1
[OK] index_contactdb_….csv   source=object_contactdb rows=600 sum_feasible=2911 grasp_rows=2911 objects=40
[OK] 跨来源去重预检: 检查 2983 条抓取，唯一 2983，重复 0
RESULT=OK
```

失败会 `exit 1`，`rescreen_all.sh` 立刻停在 step 5，**不会**用脏索引去合并。

**踩坑（值得记下来，因为它差点让我误判）**：

1. 一开始把"某行没有副本"当错误 —— 但 `run_filter` **只复制至少 1 条可行抓取的文件**，0 可行的行本来就没有副本
   （pilot 里 object_rescreen 有 7 个这样的文件、contactdb 有 27 个）。改成只对 `n_feasible>0` 的行要求副本，
   并把 `n_rows_zero_feasible_not_copied` 写进 JSON。
2. 逐抓取索引里的 `grasp_file` 是 **basename**（`apple.dexgrasp.pt`），文件级索引里是**完整相对路径**
   （`object_rescreen/apple/grasp_predictions/…/apple.dexgrasp.pt`）。一开始按完整路径做 key，
   导致 2983 条全部 fallback 成 `("", index)`、假报 "唯一 32 / 重复 20" + 一堆莫名其妙的跨来源重叠。
   改成按 basename 对齐后：**2983 唯一 / 0 重复 / 0 重叠**。
   （这条也说明 `ycb_object` 与 `ycb_object1` 确实是两批不同的抓取，见 §1.4。）

---

## 8. 任务 7：终检（`verify_v6_rescreen.py`，6 项）

```
[1/6] 1_structure:            OK   q=torch.float32 (22,)，robot=linkerhand，num_total 与 metadata 一致
[2/6] 2_prefixes:             OK   ycb 72 / contactdb 2911；无非法前缀；9 个"无抓取"物体(cube_*/cylinder_*/pyramid_*)一个都没混进来
[3/6] 3_diff_vs_baseline:     OK   新 2983 / v3 6340；物体 41 vs 68；按来源 object_rescreen 15 / object1_rescreen 57 / object_contactdb 2911
[4/6] 4_fk_roundtrip:         OK   见下
[5/6] 5_cmapdataset_load:     OK   真 CMapDataset.py：len=41、n_metadata=2983、robot=linkerhand、(1,27)/(1,512,3)；参考 allegro 521 条不受影响
[6/6] 6_v3_false_positives:   OK   见 §4.2
ALL OK = True
```

### 8.1 FK 回代（q → numpy FK → `hand_base_link` vs 原始 `root_pose`）

| 后端 | 20 条抽样 max&nbsp;rot | max&nbsp;trans | 判据 |
|---|---|---|---|
| numpy（CMap 导出用这个） | **3.95e-6 rad**（19/20 条 < 1e-6） | 1.68e-8 m | ✅ |
| pytorch_kinematics（本机 fork 有 ±1e-2 rad 偏置） | 2.04e-2 | 6.3e-9 m | 只记录，不修（REPORT_v4 §5） |

**关于那一条 3.95e-6**：它来自 apple 一个 `pitch = -88.2°`（接近 ±90° 万向锁）的抓取。
`q` 里的 rpy 是 float32 存的，在 `R = Rx·Ry·Rz` 且 `pitch→90°` 时 `1/cos(pitch) ≈ 32` 会放大存储量化误差。
我做了 float64 对照实验：把 rpy 换成 `R0` 在 float64 下反解出的精确值，再跑一次 FK →
**max rot 误差 5.79e-15**。也就是说这 3.95e-6 **完全来自 float32 存储**，不是转换/FK 逻辑错误。
判据因此写成（原样写进 JSON 的 `criteria`）：平移 < 1e-7 m；旋转 < 1e-6，**或** < 1e-5 且 float64 对照 < 1e-7。
这里没有"调阈值让结果变好看"——原始数字、原因、对照实验都写在同一份 JSON 里，可复核。

---

## 9. 交付清单

**新脚本（7 个，全部真跑过）**

| 文件 | 行数 | 作用 |
|---|---|---|
| `new_grasp_filter/rescreen_preflight.py` | 416 | 任务 0：只读预检，写 `logs/rescreen_preflight.json` |
| `new_grasp_filter/rescreen_mesh_replace.py` | 337 | 任务 1 的 Python 逻辑（备份→三方分类→替换→sha256→Chamfer/SDF 对比） |
| `new_grasp_filter/rescreen_mesh_replace.sh` | 44 | 任务 1 入口（conda 环境 + 日志 + `FORCE_REPLACE_NEW_GEOMETRY`/`MESH_OBJECTS`） |
| `new_grasp_filter/rescreen_backup.sh` | 110 | 任务 4：索引备份 + sha256 校验 + 幂等 |
| `new_grasp_filter/rescreen_all.sh` | 196 | 任务 5：7 步全流程编排（`PILOT/SKIP_MESH_REPLACE/FORCE_REPLACE_NEW_GEOMETRY/START_STEP`） |
| `new_grasp_filter/rescreen_merge_check.py` | 294 | 任务 6：合并前三份索引校验（失败即 `exit 1`） |
| `new_grasp_filter/verify_v6_rescreen.py` | 434 | 任务 7：新数据集 6 项终检（含 v5 假阳性定向复查、FK float64 对照） |

**改动的现有文件（只加参数/打印，不动数值逻辑）**：`new_grasp_filter/run_filter.py`（418 → 509 行）。

**跑出来的证据（全部可复查）**

```
new_grasp_filter/logs/rescreen_preflight.json                      275 KB  任务 0（含任务 1 结果 i_mesh_replace）
new_grasp_filter/logs/rescreen_mesh_replace.json                    56 KB  任务 1 幂等复跑（28/28 identical）
new_grasp_filter/logs/rescreen_mesh_replace_replace_run.json        60 KB  任务 1 真替换记录（7 个 replace + 8 个几何对比）
new_grasp_filter/logs/rescreen_mesh_replace_20261110_p1.log                 任务 1 终端日志
new_grasp_filter/logs/rescreen_backup.json                                  任务 4
new_grasp_filter/logs/rescreen_ts.txt                                       当前时间戳
new_grasp_filter/logs/rescreen_merge_check_pilot.json                       任务 6（PILOT）
new_grasp_filter/logs/verify_v6_rescreen_pilot.json                         任务 7（PILOT）
new_grasp_filter/logs/rescreen_pilot_20261110_224741.log                    任务 8 端到端全日志
new_grasp_filter/cmap_dataset_from_graspqp_v6_pilot.pt / .meta.json         PILOT 数据集（2983 条 / 41 物体）
new_grasp_filter/rescreen_pilot/filtered_grasps/…                            pilot 索引 + 抓取副本
new_grasp_filter/rescreen_pilot/logs/…                                       pilot 逐抓取 jsonl + summary
object1/ycb/_mesh_backup_20261110_p1/                                        任务 1 备份（112 文件 / 41 MB，回滚用）
new_grasp_filter/filtered_grasps/index_contactdb_20261110_224741.csv          任务 4 备份
new_grasp_filter/filtered_grasps/index_grasps_contactdb_20261110_224741.csv
new_grasp_filter/filtered_grasps/index_ycb_v1polluted_20261110_224741.csv
new_grasp_filter/filtered_grasps/index_ycb_grasps_v1polluted_20261110_224741.csv
```

**没动过的东西（已核对）**：`filtered_grasps/index.csv`（sha256 `d466ff19…`，600 行 / 2911 条可行，与备份逐字节相同）、
`index_grasps.csv`、`cmap_dataset_from_graspqp_v1/v2/v3/v4*.pt` 与 `.meta.json`、`REPORT.md/v2/v3/v4`、
`grasp_io.py`、`sapien_grasp_checker.py`、`object_sdf.py`、`convert_to_cmap.py`。

---

## 10. 已知限制 / 需要你知道的

1. **pilot 只证明 apple**。16 条假阳性里只有 3 条（都是 apple）拿到了"重筛确实剔除"的证据，另外 13 条是空转
   （物体不在 pilot 数据集里）。**要拿到全部 16 条的结论必须跑全量**，跑完同一份检查会自动显示
   `removed_by_rescreen` / `new_reason`。
2. **全量还没跑**（按要求）。跑完预计 ycb 部分的样本数会明显少于 v3 的 ycb 部分（v3 6340 里有 v1 污染和
   两批 3429 条可行抓取），具体数字以实际为准，我不会预设。
3. `power_drill` 的 Chamfer max 到 2.73 mm，是 8 个里最大的（该物体有细长结构，修复前后的三角面片差异更大）。
   如果你觉得这个差异不可接受，可以从 `_mesh_backup_20261110_p1/` 单个回滚该物体后重跑 step 2–7。
4. `new_geometry` / `no_backup_to_compare` 分支实现了但本仓库**没有实例**，所以 `FORCE_REPLACE_NEW_GEOMETRY`
   实测是 no-op；换机器/换数据时它才会真正起作用。
5. **GPU 路径没验证**：`--device cuda` 只换 FK 后端，SAPIEN 碰撞检测仍是 CPU；`--gpu`（PhysX GPU）本轮没试。
6. `pytorch_kinematics` 的 ±1e-2 rad 偏置依旧存在（上游 fork 问题），导出的 `q` 不受影响；`verify` 里用 numpy 判据。
7. 历史遗留：`new_grasp_filter/cmap_dataset_from_graspqp_v3_rejected.{pt,meta.json}` 还在（V5 时的副产物），
   本轮**没动**它（属于 v3 命名产物，按约束不碰）；要不要删由你决定。
