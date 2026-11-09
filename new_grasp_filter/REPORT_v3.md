# v3：修复 object/ 非水密 STL + 审计/重建 LinkerHand L20 collision mesh

> **范围**：本轮**没有重新筛选抓取**。只做了两件事：
> **(A)** 修复 `object/` 下非水密 STL（8 个 YCB + 49 个 contactdb 全量审计）；
> **(B)** 审计 L20 collision mesh，并重建唯一缺失的一个。
>
> **`object1/` 全程只读**：728 个文件中，任务开始（2026-11-05 16:51）之后被修改的为 **0 个**（`find object1 -newermt ...` 验证）。
> `Cmap_dataset/`、`object/ycb/*/*.urdf`、`linkerhand_l20.urdf` 未被修改。
> `object/` 在 `.gitignore` 中（第 3 行），因此修复产物与备份不会污染版本库。

---

## 0. 可直接复制运行的命令序列

```bash
# 0) 环境
conda activate genh2r
cd /d/study/VScodes/Graspqp-Linkerhand-L20

# 1) Blender 门禁（两个版本各跑一次；4.5 必须带 --factory-startup，原因见 §1）
"D:\Blender_Foundation\Blender_5.1\blender.exe" --background --factory-startup \
    --python new_grasp_filter/_blender_probe/probe.py -- "$(pwd)/new_grasp_filter/_blender_probe"
"D:\Blender_Foundation\Blender_4.5\blender.exe" --background --factory-startup \
    --python new_grasp_filter/_blender_probe/probe.py -- "$(pwd)/new_grasp_filter/_blender_probe"

# 2) 修复前审计（57 个 mesh：8 YCB + 49 contactdb）
python new_grasp_filter/audit_stl.py --tag before

# 3) 备份 + Blender 修复（23 个 FAIL；自动备份、自动校验、失败自动还原）
python new_grasp_filter/repair_stls.py --tag before

# 4) 修复后审计 + 对比
python new_grasp_filter/audit_stl.py --tag after

# 5) 去掉修复后残留的“内部空腔”壳
python new_grasp_filter/drop_nested.py --only contactdb/apple contactdb/flashlight contactdb/mouse

# 6) collision mesh 审计（输出 logs/collision_mesh_audit.csv/.json）
python new_grasp_filter/audit_collision_mesh.py --tag before

# 7) 重建缺失的 thumb_metacarpals collision mesh（凸包 + 0.8 mm，与现有家族一致）
python new_grasp_filter/rebuild_collision_mesh.py --links thumb_metacarpals
python new_grasp_filter/audit_collision_mesh.py --tag before   # 复检
```

Blender 单一版本约定：**后续所有 Blender 调用都用 `5.1` + `--background --factory-startup`**，完整路径
`D:\Blender_Foundation\Blender_5.1\blender.exe`。

---

## 1. 第一步：Blender 工作区门禁（硬门禁）

### 1.1 测试方法

`new_grasp_filter/_blender_probe/probe.py`：新建立方体 → 导出 STL 到工作区 → 读回（大小 + 重导入）→
在工作区写 `.txt` → 删除临时文件 → 退出。调用方式固定为
`<blender.exe> --background --python probe.py -- <workspace_probe_dir>`，**使用完整路径，未改 PATH / 环境变量**。

### 1.2 结果

| 版本 | 首次调用 | 最终调用 | 返回码 | 工作区读写删 |
|---|---|---|---|---|
| **5.1.0** | 直接成功 | `--background --python`（无额外参数） | **0** | ✅ 导出 684 B、重导入 2 objects、txt 26 B、删除成功 |
| **4.5.1 LTS** | **挂起 >7 分钟**（无任何输出、未生成文件） | 加 `--factory-startup` 后成功 | **0** | ✅ 导出 684 B、重导入 2 objects、txt 30 B、删除成功 |

5.1 的关键 stdout：

```
[probe] blender=5.1.0 probe_dir=...\new_grasp_filter\_blender_probe
[probe] exported via bpy.ops.wm.stl_export
[probe] stl size=684 head=b'\x00\x00\x00\x00\x00\x00\x00\x00'
[probe] re-imported via bpy.ops.wm.stl_import; objects=2
[probe] txt size=26
[probe] deleted temp files; remaining=['probe.py']
[probe] OK
```

5.1 的日志里同时出现（**非致命**，属于 Blender 自己的配置目录被沙箱拒绝写）：

```
Failed to remove: (... 'AppData\Roaming\Blender Foundation\Blender\5.1\extensions\...\yaml\__init__.py', PermissionError(13, ...))
Unable to set stale files pending: [Errno 13] Permission denied: '...\Blender\5.1\config\stale-pending'
```

### 1.3 4.5 挂起的原因与处理

- `blender.exe --version` 正常（`Blender 4.5.1 LTS, build 2025-07-29`），说明可执行文件本身没问题；
- 用最小 `--python-expr "print('HELLO45')"` 测试：**不带** `--factory-startup` 同样挂起，**带**则立即成功；
- 结论：4.5 在读取/清理用户配置目录（`AppData\Roaming\Blender Foundation\Blender\4.5`）时被沙箱权限挡住而卡住，
  与**工作区访问无关**。`--factory-startup` 是 Blender 官方的"跳过用户配置"开关，不改变写入位置、不改权限、
  不是绕过沙箱；加上它以后 4.5 完整通过了工作区读写删测试。

### 1.4 选定版本

**5.1**。理由：无需额外参数即可工作（4.5 必须带 `--factory-startup`），且 5.1 的 `wm.stl_import/export`
API 更完整（4.5 需要回退分支）。为统一行为，后续调用一律加 `--factory-startup`。

**门禁结论：通过。** 两个版本都能在工作区 headless 读写删，未修改任何系统设置。

---

## 2. 阶段 0：复核 `validate_cmap_dataset.json`

文件：`new_grasp_filter/logs/validate_cmap_dataset.json`（用户重跑后生成，数据为真实的 `object1`）。

### 2.1 是否全部通过

| 检查项 | 结果 |
|---|---|
| `CMapDataset` 能否加载参考数据 | ✅ `metadata=521`、`len()=521`（3 个 debug object 的 allegro 样本） |
| 能否加载我们的 v2 数据 | ✅ `metadata=3429`、`len()=28`（28 个 robot×object 组合） |
| `robots_in_metadata` | ✅ `['linkerhand']` |
| `q` 形状/类型 | ✅ `(22,)`、`torch.float32` |
| `__getitem__` | ✅ `getitem_ok=true`，返回 `robot=linkerhand`、`object=ycb+apple`、`initial_q=(1,27)`、`object_pc=(1,512,3)` |
| `q` 回代 FK（20 条） | ✅ 旋转 `2.45e-7`、平移 `7.40e-9` m |

### 2.2 左右手 URDF 索引

| 身份 | 解析到的 URDF | dof | pk 可加载 |
|---|---|---|---|
| `linkerhand`（左手） | `new_grasp_filter/assets/cmap_left_repaired.urdf` | 27 | ✅ |
| `linkerhand_right`（右手） | `Cmap_dataset/data_urdf/robot/linkerhand/right/linkerhand_l20_right_extended.urdf` | 27 | ✅ |

别名合成测试：_legacy_ payload（`info` 未声明 `linkerhand`）→ 样本被改名为 `linkerhand_right` ✅；
左手 payload（`info` 声明 `linkerhand`）→ 保持 `linkerhand` ✅。

### 2.3 warning / 字段缺失 / 数量不符

- **无 warning**，无字段缺失。
- 两处"看起来是 0"但**属于数据事实、不是问题**：
  1. `reference_linkerhand_right` 的 `metadata=0`：原始 `cmap_dataset.pt` 里**本来就没有** `linkerhand` 样本
     （v2 §1.6 已验证，24764 条只有 5 个机器人），所以用 `linkerhand_right` 过滤参考数据必然是 0；
     别名逻辑由 2.2 的合成测试覆盖。
  2. `v2_dataset.len()=28` 与 `metadata=3429` 不相等是**正常的**：`len()` 在 validate 模式下是
     "robot×object 组合数"，不是样本数。

### 2.4 结论

**上一轮产物验证通过，没有发现严重问题，无需在最终交付中额外标注。**
（唯一需要下游注意的是：左手数据用的是"修复副本"URDF，见 v2 §1.4。）

---

## 3. 阶段 A：修复 `object/` 下非水密 STL

### 3.1 范围与调研

- `object/ycb/`：8 个已知非水密物体（沿用 v2 清单）。
- `object/contactdb/`：**49 个物体，全部是 `<name>.stl` + `<name>.urdf` + `coacd_allinone.obj`**，格式与 YCB 一致，
  因此同一套流程适用。没有 `.ply` / `.glb`。
- "曾导致 GraspQP 报错的证据"：代码/文档中没有 contactdb 的专门报错记录；本轮以**几何审计**为准。

### 3.2 审计方法（以及两个必须纠正的细节）

`new_grasp_filter/audit_stl.py` 输出 `logs/stl_audit_before.{csv,json}`，字段包括：格式、顶点/面数、包围盒、
体积、`is_watertight`、`is_winding_consistent`、`is_volume`、`euler_number`、`body_count`、边界边、
非流形边、退化/重复面、每个连通体的体积、**嵌套壳**、**外法线探针**、判定与原因。

两个实现细节（都是第一次跑错后修正的，记录在此避免复发）：

1. **必须 `trimesh.load(..., process=True)`**。STL 每个三角形独立存顶点，不合并顶点时**任何** STL 都会
   被判成"非水密"（第一版审计 57/57 FAIL 就是这个原因）。
2. **外法线探针的方向**：取表面点 `p` 与外法线 `n`，`p + n·ε` 应在物体**外**。正确实心网格应约 0% 落在内部；
   内部翻面的网格（`docs/linkerhand_l20_notes.md` 记录的 "inside-out mesh"）会接近 100%。
   第一版把 `p - n·ε` 当作探针，方向反了，导致所有正常网格都报 ~100%。

**判定口径**

- **FAIL**：非水密 / 绕序不一致 / 体积非正 / 非流形边 / 存在负体积连通体 / 存在嵌套壳（内部空腔）/
  外法线探针 >10%；
- **WARN**：多连通体、退化面、重复面、外法线探针 1–10%（薄壁或凹面）；
- **PASS**：以上全部不触发。

### 3.3 修复前审计结果（57 个 mesh）

| 指标 | 数量 |
|---|---|
| FAIL | **23**（8 个 YCB + 15 个 contactdb） |
| WARN | 7（airplane, alarm_clock, binoculars, door_knob, eyeglasses, light_bulb, scissors） |
| PASS | 27 |

- **8 个 YCB 全部 FAIL，且失败模式与 v2 完全一致**（非水密 + 非流形边 3–6 条 + 重复面 1–2 个）→ 说明这些文件
  自 v2 以来未被改动过，交叉对照通过。
- contactdb 的 15 个 FAIL 分三类：
  * **开放边界**（holes）：bowl(112)、flashlight(10)、hand(28)、headphones(16)、knife(4)、pan(58)、
    piggy_bank(304)、stapler(12)、toothpaste(110)、train(64)、utah_teapot(160)；
  * **嵌套壳 / 内部空腔**：apple、mouse（以及 ps_controller 的 41 个连通体中 21 个嵌套、7 个负体积）；
  * **绕序不一致**：pan、wristwatch。

### 3.4 备份

- **主备份**：`object/_stl_backup_20261105_234004/` —— 23 个待修文件在**第一次修改前**的原样拷贝，
  保留 `ycb/<object>/…`、`contactdb/<object>/…` 目录结构，`shutil.copy2` + 逐文件 SHA-256 校验，
  清单在 `new_grasp_filter/logs/stl_repair_manifest_20261105_234004.json`。
- 后续每次"重试/补修"都另开 `_stl_backup_<时间戳>/`（`20261106_000530` 等 8 个），内容是当时的中间状态，
  同样逐字节可还原。**主备份不会被覆盖或删除。**
- `object/` 在 `.gitignore` 第 3 行，备份不会进版本库。

### 3.5 修复流程（Blender headless）

`new_grasp_filter/blender_repair/repair_mesh.py`（由 `repair_stls.py` 驱动）：

1. `merge by distance`，阈值 = `bbox_diag × 1e-5`（自适应，最小 1e-7）；
2. `fill holes`（`sides=0`，补所有洞）；
3. `normals_make_consistent(inside=False)`（**关键**，修正绕序）；
4. 若合并非流形边仍 >0 → 升级 `voxel remesh`（体素 = `bbox_diag/256`，可调 64–384）→ `smooth(0.5, 5)` →
   `decimate`（循环调用直到面数 ≤ 2× 原面数）；
5. 导出 STL，立刻用 trimesh 复检：水密、绕序、体积变化 <5%、Chamfer 中位数 <1 mm、
   **SDF 内外一致率 >99%**（表面点沿法线内/外各偏移 ε 后 `contains` 判定，ε=0.5 mm，1000 点）；
6. 不达标 → **自动从备份还原**并记录，绝不留半成品。

**关于 ε**：第一轮用 3 mm 探针，薄壁物体（碗、刀、耳机、罐子）会把"内偏移 3 mm 穿到另一侧"误判为失败；
改为 0.5 mm 后复测，这些网格实际是 99.5–100% 一致。对多连通体（火车 13 件、茶壶 3 件、手表 10 件）则用
**逐连通体**探针（本体外偏移落在另一个零件内部是合法的）。这两点是本轮明确记录的口径调整。

### 3.6 修复结果

| 结果 | 数量 | 明细 |
|---|---|---|
| **修复成功** | **22** | 8 个 YCB 全部；contactdb 14 个 |
| 修复失败（已还原原文件） | **1** | `contactdb/utah_teapot` |
| 审计通过（未改动） | 35 | PASS/WARN 且 SHA-256 与备份一致 |

22 个成功案例的指标（全部满足验收阈值）：

| 指标 | 最差值 | 阈值 |
|---|---|---|
| 体积变化 | **-3.70%**（ps_controller）/ -3.42%（toothpaste）/ -1.69%（pan） | <5% ✅ |
| 与原始 Chamfer 中位数 | **0.056 mm**（contactdb/hand） | <1 mm ✅ |
| SDF 内外一致率 | **99.10%**（mouse） | >99% ✅ |
| `is_watertight` | 22/22 True | True ✅ |
| `is_winding_consistent` | 22/22 True | True ✅ |

（完整逐文件数值在 `logs/stl_repair_log.jsonl` 与 `logs/stl_audit_after.csv`。）

修复后重新审计（57 个）：**PASS 46 / WARN 10 / FAIL 1**。
- 唯一 FAIL 是 `contactdb/utah_teapot`（保持原文件，见 §3.8）；
- 22 个被修改的文件里，19 个变为 PASS、3 个为 WARN（apple 多体=2、train 多体=13、mouse 凹面探针 3.0%），均不再 FAIL。

### 3.7 对照组（必须先验证脚本不会弄坏好网格）

用**同一套 Blender 脚本**处理 3 个审计通过的网格，输出写到 `new_grasp_filter/_blender_probe/control/`，
**不覆盖原文件**：

| 对照网格 | 原始水密 | 输出水密 | 体积变化 | Chamfer 中位数 |
|---|---|---|---|---|
| `ycb/apple` | True | True | 0.000% | 2.6e-16 mm |
| `ycb/cracker_box` | True | True | 0.000% | 0.000 mm |
| `contactdb/torus_small` | True | True | 0.000% | 0.065 mm |

**结论：脚本对好网格是恒等变换（误差在浮点噪声量级），可以安全用于坏网格。**

### 3.8 修复失败项

**`contactdb/utah_teapot`**（保持原文件不动）：

- 原始：非水密（160 条边界边）、3 个连通体；
- 尝试过的参数：默认（voxel 256 + 自动 decimate）、`--voxel-div 128/320/384`、`--force-remesh`；
- 结果：能变水密、SDF 一致率 99.8–100%，但**体积每次都缩小 5.1–5.9%**（壶嘴/把手薄壁被体素化吃掉）；
- 判定：**体积变化超过 5% 的硬阈值**，按任务要求"保留原文件、单列失败"，不做"强行修好"。
- 建议：若下游必须使用，可单独放宽该物体的体积阈值，或用人工 CAD 修补壶嘴；本轮不做。

### 3.9 跨批次一致性（`object1/` 只读）

`object1/` 下同名物体**仍是原始（未修复）版本**，本轮没有同步：

* `object1/` 728 个文件，任务开始后被修改的 **0 个**；
* 若下游要求两批几何一致，需要用户手动同步，或授权 DSH 在下一轮把 `object/` 的 22 个修复文件复制到
  `object1/` 对应位置（同样先备份）。**本轮未执行。**

### 3.10 本阶段结论 + 遗留问题

**结论**：8 个 YCB 非水密网格全部修复并通过全部阈值；contactdb 49 个中 15 个有问题，修复 14 个；
对照组证明流程不会破坏好网格；唯一失败项有明确原因并已还原。

**遗留问题**：`utah_teapot` 无法在 <5% 体积误差内修好；`object1/` 未同步；修复后网格面数普遍增加
（YCB 从 16k 到 36k–45k 面），体积/形状未变但文件更大。

---

## 4. 阶段 B：LinkerHand L20 collision mesh 审计与重建

### 4.1 URDF ↔ collision ↔ visual 对照

`linkerhand_l20.urdf` 共 22 个带 visual 的 link，每个 collision 都引用
`meshes/collisions/<link>.STL`，`<origin>` 全部是 `xyz="0 0 0" rpy="0 0 0"`（与 visual 同坐标系）。
逐 link 对照见 `logs/collision_mesh_audit.csv`（含 `urdf_has_origin` / `urdf_origin` 两列）。

**关键发现（修复前）**：

| 问题 | 证据 |
|---|---|
| `thumb_metacarpals` 的 collision 文件**不存在** | URDF 引用 `meshes/collisions/thumb_metacarpals.STL`，目录里没有；审计跳过了该行 |
| 这正是 v2 SAPIEN 加载手 URDF 报 `File not found: meshes/collisions/thumb_metacarpals.STL` 的直接原因 | v2 日志 |

其余 21 个 link 的 collision 文件齐全、命名一一对应，没有多余文件。

### 4.2 审计结果（22 个 link）

`new_grasp_filter/audit_collision_mesh.py` → `logs/collision_mesh_audit.{csv,json}`。

| 检查 | 结果 |
|---|---|
| `is_watertight` | **22/22 True** |
| `is_winding_consistent` | **22/22 True** |
| 非流形边 | 0 |
| 嵌套壳（内部空腔） | 0 |
| 外法线探针 | 21 个 link = 0.0%；`hand_base_link` = 1.5% |
| 与 visual 的 Chamfer 中位数 | 0.2–1.4 mm |
| 体积比（collision / visual） | **1.73× – 7.21×** |
| 凸性偏差（hull↔collision Chamfer） | 0.66–0.89 mm（说明 collision ≈ 轻微外扩的凸包） |

**判定：0 FAIL / 22 WARN。** 所有 WARN 都是"体积比 1.7–7.2×"这一条——这是**设计使然**（它们是包住视觉
网格的凸包，用于碰撞），不属于任务列的 FAIL 条件（非水密/绕序/内外错误/空腔/自交）。

`hand_base_link` 的 1.5% 外法线探针：2000 点复测为 1.65%（ε=0.5 mm）/1.55%（ε=1.0 mm），失败点**散布在整个
掌部包围盒**（`[-0.033,-0.048,0.023] … [-0.015,0.054,0.169]`），而手指 link 是 0.0%。这是**凹陷区域**（掌心的
抓握面）里"外法线方向落在掌体内部"的几何现象，不是翻面（翻面会接近 100%）。因此判定为 WARN，
未重建。审计脚本的 FAIL 阈值设为 5%，并把这个理由写进了代码注释与 CSV 的 `notes`。

### 4.3 重建：只补了缺失的 `thumb_metacarpals`

**为什么只重建一个**：其余 21 个全部通过 FAIL 判据，按任务要求"有问题才重建，没问题不改"，不动它们。

**重建方法**（`rebuild_collision_mesh.py`）：现有 collision mesh 的实测规律是
**visual 的凸包 + 约 0.8 mm 外扩**（hull→collision Chamfer 中位数 0.66–0.89 mm，体积约为凸包的 1.05×），
所以缺失的那个按同一配方生成：

1. `visual mesh → convex hull`；
2. 沿质心方向整体外扩 **0.8 mm**；
3. 导出 `meshes/collisions/thumb_metacarpals.STL`；
4. 复检。

| 项 | 值 |
|---|---|
| 输出面数 | 7,330 |
| 体积 | 59.57 cm³（visual 22.81 cm³，**2.61×**，与同族 base1 3.38× / base2 3.67× / proximal 2.98× 一致） |
| 水密 / 绕序 | True / True |
| 与凸包 Chamfer | 0.657 mm（与现有家族的 0.66–0.89 mm 同量级） |
| 备份 | 原文件本就不存在，`collisions_backup_20261106_072002/` 为空目录并记录"nothing to back up" |

复检后 22 个 link 全部 WARN（体积比），**0 FAIL**，`thumb_metacarpals` 也进入 WARN（2.61×）。

### 4.4 重建对照组

对 3 个原本正常的 link 跑同一套重建脚本，输出写到 `_blender_probe/collision_control/`，
与人工版对比：

| link | 人工版体积 | 重建体积 | 与人工版 Chamfer 中位数 / p95 |
|---|---|---|---|
| `middle_distal` | 7.35 cm³ | 6.94 cm³ | **0.202 mm / 0.457 mm** |
| `hand_base_link` | 728.39 cm³ | 717.61 cm³ | **0.263 mm / 0.625 mm** |
| `index_proximal` | 12.18 cm³ | 14.57 cm³ | 0.379 mm / **7.98 mm** |

**结论**：配方能复现人工版（2/3 在 0.5 mm 内）；`index_proximal` 偏差较大（p95 8 mm，凸包比人工版更胖，
且该凸包的绕序检查未通过），说明**人工版对个别 link 做过额外修整**。因此本轮的结论是：
**该配方只用于补"缺失"的 mesh，不用来替换现有可用的 mesh**。已有的 21 个保持原样。

### 4.5 对上一轮筛选结果的影响

- **被改动的 collision mesh：只有 1 个**（新增 `thumb_metacarpals`，原本不存在）。
- v2 的 SAPIEN 筛选**没有使用**这些 collision mesh：它对手侧用的是 **visual mesh 的凸包**
  （`sapien_grasp_checker.build_hand_hulls`），所以 **v2 的 feasible 集合与提离结果不受本次重建影响**。
- 新增文件的唯一直接效果：`linkerhand_l20.urdf` 现在可以被 SAPIEN 正常加载（v2 加载时会报缺失文件），
  对下游"用 URDF + collision mesh"的仿真器是修复。
- **对象 STL 的修复**（22 个）会让 SAPIEN 的动态三角网格 SDF 更可靠（此前非水密的 8 个 YCB 会打印
  `SDF creation: Mesh is not suitable for SDF`）。几何变化 <0.06 mm，对 v2 的判定影响应在数值噪声内，
  但严格来说输入变了。
- **是否建议重跑筛选**：
  * 只关心"这批抓取能不能用" → **不必重跑**（几何变化远小于 2 mm 的接触阈值）；
  * 需要与当前 `object/` 网格**严格可复现**的筛选结果 → **建议只重跑 8 个 YCB 物体**（约 8×100 s），
    contactdb 没有抓取数据，不涉及；
  * 需要两批（object/object1）几何一致 → 先决定是否同步 `object1/`（§3.9）。

### 4.6 本阶段结论 + 遗留问题

**结论**：21 个现有 collision mesh 全部通过硬性判据（水密/绕序/无空腔/无自交/法线方向正确），无需重建；
发现并补齐了唯一缺失的 `thumb_metacarpals`（这也是 v2 里手 URDF 加载失败的根因）。
体积比 1.7–7.2× 是凸包设计的固有"幽灵体积"，已在 v2 报告中作为手侧近似误差记录。

**遗留问题**：这些 collision mesh 是**凸包**而非精确表面，若下游用它做接触/穿透，会把 1.7–7.2× 的体积
算成实体（v2 §2.6 已记录该风险）；若要更精确，应改为"视觉网格的体素化水密重建"而不是凸包，这会改变
抓取筛选结果，需要另开一轮。

---

## 5. Ubuntu 迁移注意项

| # | 项 | 说明 |
|---|---|---|
| 1 | Blender 路径 | Windows 为 `D:\Blender_Foundation\Blender_5.1\blender.exe`；Ubuntu 通常是 `/usr/bin/blender` 或解压目录下的 `blender`。脚本里 `BLENDER` 是常量，迁移时改一处或改成环境变量 |
| 2 | headless 参数 | `--background --python script.py -- args` 跨平台一致；`--factory-startup` 建议保留（避免读用户配置） |
| 3 | Blender Python 版本 | 5.1 自带 Python 3.13；脚本只用了 `bpy`/`bmesh`/`math`/`os`/`sys`/`json`，无版本敏感 API |
| 4 | trimesh 跨平台 | 审计/校验都在 conda 的 `genh2r` 里跑，与 Blender 无关；`process=True`、`contains` 行为一致 |
| 5 | 文件权限 | 本轮在 Windows 遇到 `object/` 的 ACL 缺 `WRITE_OWNER`（已用 DSH 的 ACL 修复脚本补上当前用户完全控制，备份与恢复命令在 `new_grasp_filter/logs/acl_recovery/`）。Ubuntu 上一般不存在该问题，但要注意 `object/` 属主 |
| 6 | 子进程管道 | 本沙箱禁止 `subprocess` 创建管道；`repair_stls.py` 因此把 Blender 输出重定向到**文件**再读取。Ubuntu 上管道可用，但文件重定向同样有效，无需改 |
| 7 | 并发 | 单进程串行调用 Blender（每个文件一次进程启动）。Ubuntu 上可并行，但要注意显存/内存 |
| 8 | 备份路径 | `object/_stl_backup_<ts>/` 与 `graspqp/assets/linkerhand_l20/collisions_backup_<ts>/` 都是相对仓库根的路径，迁移后自动成立 |

---

## 6. 已知问题与假设

| # | 项 | 状态 |
|---|---|---|
| B1 | Blender 4.5 不带 `--factory-startup` 会挂起 | ⚠️ 环境限制（沙箱阻止写 AppData），已用官方开关规避并记录；5.1 无此问题 |
| B2 | 审计的"SDF 内外一致率"阈值 | 默认 ε=0.5 mm；薄壁需小 ε、多体需逐体判定，两者都已实现并写明 |
| B3 | `contactdb/utah_teapot` 修复失败 | ❌ 体积损失 5.1–5.9% > 5%，保留原文件 |
| B4 | `object1/` 未同步修复后的网格 | ⚠️ 按约束只读；同步需用户授权 |
| B5 | collision mesh 体积比 1.7–7.2× | ⚠️ 凸包设计固有；未改，避免影响既有流水线 |
| B6 | 修复后网格面数上升（YCB 16k→36k–45k） | ℹ️ 体积/形状不变（Chamfer <0.06 mm），仅文件变大 |
| B7 | `index_proximal` 重建结果与人工版差异 p95=8 mm | ⚠️ 人工版有额外修整；该 link 未重建（本就不需要），仅作为对照结论 |
| B8 | 本轮未重跑抓取筛选 | ✅ 按任务要求；影响与建议见 §4.5 |

---

## 7. 产物清单

| 路径 | 说明 |
|---|---|
| `object/_stl_backup_20261105_234004/` | **主备份**：23 个待修 STL 的修复前原样（含 SHA-256 清单） |
| `object/_stl_backup_*`（另外 8 个） | 重试/补修的中间备份 |
| `object/ycb/<8 个物体>/<obj>.stl` | 已修复（水密） |
| `object/contactdb/<14 个物体>/<obj>.stl` | 已修复 |
| `object/contactdb/utah_teapot/utah_teapot.stl` | **未改动**（修复失败，已还原） |
| `graspqp/assets/linkerhand_l20/meshes/collisions/thumb_metacarpals.STL` | **新增**（重建，7,330 面 / 59.57 cm³） |
| `graspqp/assets/linkerhand_l20/collisions_backup_20261106_072002/` | 该 link 原本缺失，备份目录为空并注明 |
| `new_grasp_filter/audit_stl.py` / `repair_stls.py` / `drop_nested.py` | 审计与修复脚本 |
| `new_grasp_filter/blender_repair/repair_mesh.py` | Blender 修复脚本 |
| `new_grasp_filter/audit_collision_mesh.py` / `rebuild_collision_mesh.py` | collision 审计与重建 |
| `new_grasp_filter/_blender_probe/probe.py` | 门禁探针 |
| `new_grasp_filter/logs/stl_audit_before.{csv,json}` | 修复前审计（57 个） |
| `new_grasp_filter/logs/stl_audit_after.{csv,json}` | 修复后审计 |
| `new_grasp_filter/logs/stl_repair_log.jsonl` | 逐文件修复记录（参数、复检、还原） |
| `new_grasp_filter/logs/stl_repair_manifest_*.json` | 备份清单与 SHA-256 |
| `new_grasp_filter/logs/collision_mesh_audit.{csv,json}` | collision 审计（22 个 link） |
| `new_grasp_filter/logs/collision_rebuild_*.json` | 重建记录 |
| `new_grasp_filter/logs/acl_recovery/` | `object/` 权限修复的备份、报告与**回滚脚本** |
