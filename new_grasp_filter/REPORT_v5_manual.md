# REPORT v5 — 人工审核工具（CMapDataset 逐条可视化 + 按键保留/丢弃）

> 本轮**只**做一件事：写一个人工筛选脚本，把 `cmap_dataset_from_graspqp_v3.pt` 里的每条抓取用 open3d 可视化，
> 由人按键决定保留/丢弃，最后把**被按 y 保留**的样本按与 v3.pt **完全相同**的结构写成新 `.pt`。
> 没有重跑 SAPIEN 筛选、没有改任何 mesh、没有改 v2/v3 的 `.pt` / `.meta.json`、没有改
> `grasp_io.py` / `sapien_grasp_checker.py` / `object_sdf.py` 的任何数值逻辑、没碰 `object1/`。

## 0. 交付物

| 文件 | 说明 |
|---|---|
| `new_grasp_filter/manual_review.py` | 审核工具本体（唯一新增的可执行脚本，~1200 行，含模块级文档） |
| `new_grasp_filter/verify_v5_manual.py` | 五项强制验证脚本（只读，不改任何东西） |
| `new_grasp_filter/manual_review_decisions.jsonl` | 决策日志（append-only、可续审、可撤销）— **由你第一次真实按键产生** |
| `new_grasp_filter/cmap_dataset_from_graspqp_v3_manual.pt` (+ `.meta.json`) | 只含 keep 的新数据集 — **由你第一次真实按键产生** |
| `new_grasp_filter/logs/manual_review_v5.json` | 汇总统计（状态索引；真实审核后会被完整统计覆盖） |
| `new_grasp_filter/logs/manual_review_pilot/` | 本轮**实测存档**：27 条真实按键产生的 `decisions.jsonl`、pilot `.pt`、`summary.json`、`verify_pilot.json`、`audit_feasible_recheck.json`、各会话 `*.log`、截图 |

> ⚠️ **关于「真实产物」**：本轮的硬约束是「只有真正被用户按键保留的样本才进新 `.pt`」。
> 所以我**没有**替你伪造交付路径上的 `manual_review_decisions.jsonl` / `..._v3_manual.pt`——它们必须由你按键生成。
> 为了证明工具链可用，我在 `logs/manual_review_pilot/` 里做了一次**真实交互 pilot**（真的开窗口、真的注入按键、真的落盘），
> 下面第 3–7 节的每条命令和每行输出都来自那次 pilot。

## 1. 数据源与输出结构

- 数据源固定为 `new_grasp_filter/cmap_dataset_from_graspqp_v3.pt`（**不**从 `filtered_grasps` 重新组装）。
  实读结构：6340 条 metadata / 68 个物体 / `num_upper_object = 285`。

```
{"info": {"linkerhand": {"robot_name": "linkerhand", "num_total": 6340,
                         "num_upper_object": 285, "num_per_object": {obj: n}},
          ...},                                            # 只有 linkerhand 一个机器人
 "metadata": [ (Tensor(22,) float32, "<prefix>+<obj>", "linkerhand"), ... ]}   # list[tuple]
```

- 输出 `.pt` 与 v3 **逐项同构**：顶层键仍只有 `info` / `metadata`；`metadata` 仍是 `list[tuple]`；
  `q` 仍是 `float32 (22,)`（**直接复用 v3 里的同一个 tensor 对象，不做任何数值改动**）；
  `robot_name` 仍是 `linkerhand`；顺序仍是 v3 的 metadata 顺序（不是审核顺序）。
- `info` 里 `num_total` / `num_upper_object` / `num_per_object` 按保留结果**重算**，并额外挂一个
  `info["linkerhand"]["manual_review"] = {num_before, num_after, num_rejected, num_unreviewed,
  kept_by_prefix, rejected_by_prefix, decisions_file}`。
  之所以能安全地挂：`Cmap_dataset/dataset/CMapDataset.py` 只用 `payload["info"].keys()` 判断有哪些机器人，
  不会遍历 info 的字段（已实测，见 §5 第 4 项）。若你希望 `info` 一字不差，`.meta.json` 里也有一份同样的统计。
- `.meta.json` 记录：`n_samples / n_objects / per_object / per_prefix / per_source / num_kept / num_rejected /
  num_unreviewed / diff_vs_source（逐物体 before/kept/rejected/unreviewed）/ objects_fully_rejected /
  fk_backend / object_geom / decisions_file / app_version`。

## 2. 交互与命令行

按键（`manual_review.py --help` 里也在，逐字一致）：

| 键 | 动作 |
|---|---|
| `y` / 空格 | 保留（keep） |
| `n` / `d` | 丢弃（reject） |
| `s` | 暂缓（**不写** decisions；本次会话往前走，下次启动还会出现） |
| `u` | 撤销上一条决策（弹出 decisions.jsonl 最后一行 + 回到那条样本） |
| `r` | 重置视角 |
| `v` | 切换视角 auto / front / top / side |
| `q` / `Esc` | 退出（已做决策全保留；`.pt` 只含 keep） |

- **每次按键立刻**追加一行 decisions.jsonl（`flush()` + `os.fsync()`，实测 `kill -9` 后 3 条决策全在）。
- 断点续审：启动时读已有 decisions，**默认只审未决策的**（keep/reject 都算已决策），并打印
  `[resume] 行数=27 已 keep=21 已 reject=6 未审=6313`；`--review-all` 强制重审（重审时最新决策覆盖旧的）。
- 屏幕上看得到：物体点云/网格（橙色）、手部 22 个 link 的**视觉** mesh、`hand_base_link` 与物体坐标系、
  `q` 的 xyz/rpy/关节角、当前决策状态、`meta#` 进度。标题栏里还有实时 HUD（open3d 0.20 没有 `add_3d_label`，
  见 §8）。
- CLI（`--help` 全文见 `logs/manual_review_pilot/help.txt`）：
  `--dataset --output [--prefixes ycb contactdb] [--objects ...] [--start N] [--limit N] [--shuffle] [--seed 0]
  [--decisions ...] [--view auto|front|top|side] [--record-dir ...] [--dry-run] [--review-all] [--offscreen]`
  外加：`--index-csvs`（显式指定索引 csv）、`--uid-mode {sha,positional}`、`--fk-backend {numpy,pk}`、
  `--object-geom {auto,points,mesh}`、`--point-size`、`--width/--height`、`--save-every`、`--summary-json`、
  `--rejected-output`。
- 无 `DISPLAY` 时**硬失败**，不静默降级：
  `[error] 没有 DISPLAY（无法交互）。请用 ssh -X/ssh -Y/VNC，或用 --dry-run（只重放 decisions）／--offscreen --record-dir <dir>（只出图）。`
  `--offscreen` 必须配 `--record-dir`，否则 `[error] --offscreen 需要配合 --record-dir <dir>（只出图，不交互）`。

## 3. 实测：pilot 交互（真实按键）

```bash
# 会话 1：ycb+apple 前 25 条（注入键序 y,y,n,y,y,y,y,y,n,y,y,y,y,y,y,y,y,n,y,y,y,y,y,n,y,q）
DISPLAY=:10.0 python new_grasp_filter/manual_review.py --objects ycb+apple --limit 25 \
  --decisions new_grasp_filter/logs/manual_review_pilot/decisions.jsonl \
  --output    new_grasp_filter/logs/manual_review_pilot/cmap_dataset_from_graspqp_v3_manual_pilot.pt \
  --summary-json new_grasp_filter/logs/manual_review_pilot/summary.json
# 会话 2：contactdb+sphere_large 两条都丢（键序 n,n,q），续审同一 decisions 文件
DISPLAY=:10.0 python new_grasp_filter/manual_review.py --objects contactdb+sphere_large \
  --decisions .../decisions.jsonl --output .../cmap_dataset_from_graspqp_v3_manual_pilot.pt
# 收尾：重放（--dry-run 也能写产物）
python new_grasp_filter/manual_review.py --dry-run --decisions .../decisions.jsonl --output .../_manual_pilot.pt \
  --rejected-output .../cmap_dataset_from_graspqp_v3_rejected_pilot.pt
```

原始输出（`logs/manual_review_pilot/s1_apple.log` / `s2_sphere.log`，节选）：

```
[load] cmap_dataset_from_graspqp_v3.pt: metadata=6340  info.num_total=6340  robot=linkerhand
[uid] 与 convert_to_cmap 的去重键一致（6340 条，例 77ae16debc6beea347e4346ee925c50c6ac7048d81b6dba6308fd4ca90c949db:3）
[resume] decisions=.../decisions.jsonl  行数=0  已 keep=0  已 reject=0  未审=6340
[queue] {'after_filters': 131, 'pending_after_resume_filter': 131, 'queued': 25}  本次待审 25 条
[hand] fk_backend=numpy  links=22  tris=321230  urdf=.../assets/cmap_left_repaired.urdf
[view] 键位: y/space=keep  n/d=reject  s=skip  u=undo  r=reset  v=view  q/Esc=quit
[1/25] meta#0 ycb+apple robot=linkerhand undecided
t=(-0.201,-0.005,-0.027) rpy=(-2.953,+1.414,-0.030)
[1/25] keep   ycb+apple      77ae16debc6beea347e4346ee925c50c6ac7048d81b6dba6308fd4ca90c949db:3
...
[25/25] meta#24 ycb+apple robot=linkerhand undecided
[25/25] keep   ycb+apple      0358ac4c30579ae5284ffd77c39b743fcf5446f83629f199f3bf1c53855a0d23:20
[write] .../cmap_dataset_from_graspqp_v3_manual_pilot.pt  n=21 / 6340  keep_prefix={'ycb': 21}  rejected=4

object_name                       before   kept  rejected  unreviewed
ycb+apple                            131     21         4         106
TOTAL (reviewed objects)             131     21         4         106
TOTAL (whole dataset)               6340     21         4        6315
[2/2] reject contactdb+sphere_large   68908530ad15c97b4dd65815f427c657a316d130770d348efc23e743eed22adb:28
[write] ..._manual_pilot.pt  n=21 / 6340  keep_prefix={'ycb': 21}  rejected=6
TOTAL (whole dataset)               6340     21         6        6313
```

pilot 结果：**27 条决策 = 21 keep + 6 reject**，输出 21 条（1 个物体 `ycb+apple`），
`unreviewed = 6313`。`--objects ycb+apple` 的 131 条里还剩 106 条未审 —— 这正是续审该看到的状态。

## 4. 行为验证（都是真按键，不是"应该没问题"）

| 场景 | 命令要点 | 结果（原始日志） |
|---|---|---|
| 键位全覆盖 `y,n,space,s,u,v,y,r,n,q` | `--limit 6 --record-dir tests/shots` | `tests/keys_all.log`：meta#0 keep → meta#1 reject → meta#2 keep → meta#3 **skip（不写入）** → `[undo] 撤销 uid=...:7（原决策 keep）` 并回到 meta#2 → 再 `y` 覆盖为 keep → `v/r` 只换视角不产生决策（日志里出现两次 `[3/6]`、`[4/6]` 行）→ `n` 把 meta#3 判 reject → `q` 退出并 `[write] n=2 ... rejected=2` |
| `kill -9` 后续审 | 先 `y,n,y` 再杀掉 | 3 行决策全在、**没有半个 `.pt`**；重启 `[resume] 行数=3 已 keep=2 已 reject=1 未审=6337` → 从 `meta#3` 继续 → `[write] n=3` |
| `--review-all` 重审 | 对已决策的 3 条执行 `s,s,d,q` | `[1/3] meta#0 ... decided=KEEP` / `[2/3] meta#1 ... decided=REJECT`；`s` 不写；`d` 把 meta#2 的 keep **覆盖**为 reject，输出从 n=3 变 n=2 |
| 无 DISPLAY / `--offscreen` 误用 | 见 §2 末 | `tests/errors.log` 两条 `[error]` |

## 5. 五项强制验证（`verify_v5_manual.py`，对象是 pilot 产物）

```bash
python -u new_grasp_filter/verify_v5_manual.py \
  --manual  new_grasp_filter/logs/manual_review_pilot/cmap_dataset_from_graspqp_v3_manual_pilot.pt \
  --decisions new_grasp_filter/logs/manual_review_pilot/decisions.jsonl \
  --out new_grasp_filter/logs/manual_review_pilot/verify_pilot.json
```

原始 stdout：

```
[uid] 与 convert_to_cmap 的去重键一致（6340 条，例 77ae16debc6beea347e4346ee925c50c6ac7048d81b6dba6308fd4ca90c949db:3）
[1/5] structure: OK
[2/5] bit-identical: OK
[3/5] fk roundtrip: OK
!!! Using debug objects !!!
!!! Using fixed object pcs !!!
[4/5] CMapDataset load: OK
[5/5] dry-run determinism: OK

ALL OK = True
```

| # | 验证 | 数字（`verify_pilot.json`） |
|---|---|---|
| 1 | **结构相等** | 顶层键 `{info, metadata}` 都是；`metadata` 都是 `list[tuple(22,)float32, str, str]`；`robot_name` 唯一 `linkerhand`；`q` dtype `{float32}` / shape `{22}`；前缀 `{ycb}`（都在 `{ycb,contactdb}` 内）；`num_total` 与 `len(metadata)` 一致（21）；`num_per_object == 实际计数`（`ycb+apple: 21`）；`num_upper_object = 21`；`manual_review` 块在位；`output_uids == decisions 里的 keep 集合`；rejected 样本 **0** 条混入；输出顺序是 v3 顺序的子序列（真实下标严格递增） |
| 2 | **逐位相同** | 随机 20 条：`torch.equal(q_manual, q_v3)` 全 True，`max_abs_diff = 0.0` |
| 3 | **q → FK 往返** | 20 条：旋转 `max|dR| = 3.06e-07`、平移 `max|dt| = 1.60e-08 m`（numpy 路径）；pytorch_kinematics 路径 `max|dR| = 1.98e-02`（已知 fork 偏置，见 §8，**记录而不修**） |
| 4 | **真的能被 `CMapDataset.py` 读** | `len(ds) = 1`（== 物体数）；`ds[0]` → `robot_name='linkerhand'`、`initial_q (1,27)`、`object_pc (1,512,3)`；参照 allegro 数据集 521 条 metadata 不受影响（`robots_in_metadata=['allegro']`，结构校验通过） |
| 5 | **`--dry-run` 两次张量完全一致** | 两次独立进程重放的 `metadata` 逐张量 `torch.equal` 全 True、object_name 序列相同、`num_per_object` 相同；并且与**已交付的 `.pt`** 也逐张量相同（`matches_shipped_file=True`）。比的是张量不是 md5 —— `torch.save` 的 zip 容器不保证逐字节可复现 |

## 6. 几何正确性：三方对拍

1. **u ↔ 渲染**：`q` 用的是 v3 里那个 tensor 本身，没有任何再量化。
2. **渲染 ↔ 原始根位姿**：`q → (R, t_root = q[:3] + R@(0.03,0,0.10))` 与 SAPIEN `root_pose` 对拍，
   7 个样本（含下面那些"离得很远"的）`max|dR| ≤ 1.4e-07`、`max|dt| ≤ 1.7e-08 m`。
3. **渲染 ↔ SAPIEN checker 的几何**：`sapien_grasp_checker._place_hand` 摆的是 `Troot @ fk`，
   我的可视化摆的是同一个 `Troot @ fk`，物体都在世界原点、都用 `object/ycb/<obj>/<obj>.stl` 原尺寸（checker 里**没有**任何 scale）。
   直接对拍 4 个样本（手部离物体最近距离）：

   | meta# | 我的测量（指尖/指腹到物体表面） | SAPIEN `geo_closest_gap_mm` | checker 判定 |
   |---|---|---|---|
   | #0 | 17.77 mm | 18.94 mm | `insufficient_contacts` |
   | #4 | 98.99 mm | 79.17 mm（全 22 link 的最小值） | `insufficient_contacts` |
   | #9 | 100.91 mm | 76.24 mm | `insufficient_contacts` |
   | #7 | **0.00 mm** | 0.00 mm（`geo_penetration_mm 14.04`、196 个接触点） | `ok` |

   （我只看 5 个指尖/指腹 link，checker 扫全 22 link，所以我的数只会**更小**；两者对"有没有接触"的判断完全一致。）

## 7. 【重要发现，本轮不修】ycb 部分存在"完全没接触"的可行假阳性

做 §6 对拍时发现：pilot 里被判 keep 的 apple 样本有一半手离苹果 **2–10 cm**。用**未改动**的
`SapienGraspChecker`（默认 `CheckerConfig()`）做了一次**只读抽查**（100 条随机 v3 样本，`seed=0`，
不改任何数据集），存于 `logs/manual_review_pilot/audit_feasible_recheck.json`：

```
抽样 100 条  复现 ok=30  未复现=70 (70%)
未复现原因分布: {'slips_on_lift': 54, 'insufficient_contacts': 16}
按前缀: ycb ok=11 / insufficient_contacts=16 / slips_on_lift=24   (n=51)
        contactdb ok=19 / slips_on_lift=30 / insufficient_contacts=0   (n=49)
```

- `insufficient_contacts` 的 16 条**全部集中在 ycb**，几何最近间隙 4.1 / 5.3 / 7.8 / 8.2 / 11.3 / 13.0 / 15.8 /
  19.7 / 24.3 / 24.9 / 25.7 / 39.4 / 41.2 / 45.6 / 52.5 / 54.0 mm —— 16 条全部 >2mm，12 条 >10mm。
  这是**与仿真无关的静态几何矛盾**：`contact_dist=2mm` 的判据不可能让 4–54 mm 的间隙通过。
- `slips_on_lift` 的 54 条**都有真实接触**（间隙 0.0 mm），它依赖 PhysX 提离仿真的机器/数值细节，
  本次抽查**不下结论**。
- 溯源（只读线索）：`filtered_grasps/index_ycb.csv` 的每一行都带 `merged_from = ...\index_v1_object.csv`，
  即 ycb 的判定是从 **v1 的 object 索引**继承来的；而 `REPORT_v2.md §2.3` 明确写过
  "apple 第 3 号抓取真实情况是手在物体外 19.1 mm、0 个接触点 …… v2 修正后该抓取被 `insufficient_contacts`
  正确拒绝"——我抽到的 apples#0 正是这条（18.94 mm / `insufficient_contacts`），而它**在 v3 里**。
  contactdb 部分是 v4 在本机重跑出来的（`index.csv`，2911 条），抽查 0/49 无接触，干净。

结论：**ycb 那 3429 条里估计有 ~31%（16/51，95% 置信区间约 18–44%）是"手根本没碰到物体"的假阳性**。
这恰好是人工审核要解决的问题，本轮**不重跑筛选**（HANDOFF §8.2 的 deferred 项），只在此登记事实与证据。
建议：审核顺序上先扫 ycb（`--prefixes ycb`），或先跑
`python new_grasp_filter/manual_review.py --prefixes ycb --decisions .../ycb_decisions.jsonl`。

## 8. 已知限制 / 与"理想实现"的偏差

1. **需要 X11**：open3d 的 legacy `VisualizerWithKeyCallback` 必须有 `DISPLAY`（本机实测 `:10.0` 可用；
   `ssh -X` / VNC 都行）。没有 DISPLAY 直接硬失败，不会悄悄退化成 `--dry-run`。
2. **open3d 0.20 的 legacy visualizer 没有 `add_3d_label` / 没有标题设置接口**，所以屏幕上的 HUD 分两处：
   终端逐条打印（`t/rpy/关节角/决策计数`）+ X11 窗口标题（`XStoreName` + `_NET_WM_NAME` 尽力而为，无 X11 时静默无效）+
   `--record-dir` 时把 HUD 用 PIL 烧进 PNG。所有 HUD 文本都是 ASCII（open3d/PIL 默认字体没有中文字形）。
3. **`--fk-backend` 默认 `numpy`**：`pytorch_kinematics`（本仓库的 fork）有约 ±0.01 rad/revolute 关节偏置，
   用它摆出来的手会整体偏 ~4 cm（§5 第 3 项的数字就是它）。所以默认用 numpy FK（与 `grasp_io.fk_batch` 同源），
   `--fk-backend pk` 保留用于复现。**这是唯一一处偏离"随手用现成库"的地方，理由已量化。**
4. **`s` 只在"下次启动"时重复出现**：一次会话内跳过会继续往前走（decisions 里没记，所以下次一定还在）。
5. **`--objects` 只接受 v3 里的全名**（`ycb+apple` / `contactdb+mug`）；`--prefixes` 先过滤。
6. **内存**：`ObjectGeometry` 只缓存 2 个物体的几何（点云 `<prefix>/<obj>.pt` 优先，缺了退回 `<obj>.stl`），
   全量 6340 条会不断换物体，单条加载 ~0.1–0.4 s。
7. **v3 里存在 771 组 `(object_name, q)` 逐位重复**（同一物体的不同 grasp 文件里有相同 q）。所以
   「用 (object, q) 唯一反查 v3 下标」的做法不成立，`verify_v5_manual.py` 用的是「按 v3 顺序贪心匹配 +
   另用 `--dry-run` 重放与交付文件逐张量相同」来定位与验证顺序。
8. `--rejected-output` **默认不写**（给了路径才写），避免无意覆盖仓库里已有的同名文件。

## 9. 你接下来要做的

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate graspqp
# 1) 交互审核（先 ycb，再 contactdb；随时 q 退出，随时重跑续审）
python new_grasp_filter/manual_review.py \
  --dataset new_grasp_filter/cmap_dataset_from_graspqp_v3.pt \
  --output  new_grasp_filter/cmap_dataset_from_graspqp_v3_manual.pt
# 2) 审完（或任何时刻）验证五项
python -u new_grasp_filter/verify_v5_manual.py --out new_grasp_filter/logs/verify_v5_manual.json
```

- 全量 6340 条按每条 1.5–3 s（看图+按键）大概是 **2.6–5.3 小时**；建议按物体分批
  （`--objects ycb+apple --limit 50`），随时 `q`，下次自动续审。
- 想先看一眼"图形长什么样、不产生任何决策"：`--record-dir /tmp/peek --offscreen --limit 5`（只出图，不写 .pt）。
- 交付路径上的 `manual_review_decisions.jsonl` / `cmap_dataset_from_graspqp_v3_manual.pt` / `logs/manual_review_v5.json`
  会在你第一次真实按键退出时自动生成（`logs/manual_review_v5.json` 目前是一份状态索引，写明了 pilot 与验证的位置）。

## 10. 未改动清单（自检）

`grasp_io.py` / `sapien_grasp_checker.py` / `object_sdf.py` / `run_filter.py` / `repair_stls.py` /
`convert_to_cmap.py` / `validate_cmap_dataset.py` / `verify_cmap_v3.py` / `visualize_filtered.py` 全部未改；
`cmap_dataset_from_graspqp.pt` / `_v2.pt` / `_v3.pt` / 各 `.meta.json` 的 mtime 与 md5 未变
（`_v3.pt` = `f136a319d62f06d6a9854a9c55216c15`）；`REPORT.md/v2/v3/v4` 未改；`object1/` 只读未碰；
`Cmap_dataset/dataset/CMapDataset.py` 未改。
`git status` 里的修改项均为上一轮（v4/ContactDB）遗留，本轮只新增
`manual_review.py`、`verify_v5_manual.py`、`REPORT_v5_manual.md`、`logs/manual_review_v5.json`、`logs/manual_review_pilot/`。
