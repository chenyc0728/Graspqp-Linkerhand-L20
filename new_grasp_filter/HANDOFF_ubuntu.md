# 交接文档：GraspQP × LinkerHand L20 抓取筛选 → CMapDataset（Ubuntu 续作）

> **用途**：把 Windows 侧三轮（v1/v2/v3）的全部上下文压缩成一份可直接带走的说明。
> 在 Ubuntu 上开新的 DSH 会话时，把本文件交给它，并附上下面的启动提示词即可继续。
>
> **最后更新**：2026-11-10（Windows 侧 v1–v3 收尾；同日按 Ubuntu 新状态更新：contactdb 已筛选、待合并）

---

## 0. 给新会话的启动提示词（可直接复制）

```text
这是一个 GraspQP + LinkerHand L20 抓取筛选项目，历史工作在 Windows 上完成（v1/v2/v3），
交接文档在 new_grasp_filter/HANDOFF_ubuntu.md，请先完整读它，再读 REPORT_v3.md
（以及 REPORT_v2.md 的 §2/§3；本轮不要重跑筛选、不要改网格）。

本机环境：Ubuntu 服务器，conda activate graspqp（完整 GraspQP 栈可用），
没有 isaaclab，仿真使用 SAPIEN；Blender 只有 5.1（仅当需要修网格时才用）。

当前状态（详见 §2）：
- ycb 两批已筛选合并：cmap_dataset_from_graspqp_v2.pt（3429 条，robot_name=linkerhand）；
- contactdb 的网格修复后已重新生成抓取并完成筛选，结果在当前
  filtered_grasps/index.csv + index_grasps.csv（source=object_contactdb）；
  ycb 的旧索引已备份为 index_ycb.csv / index_ycb_grasps.csv / index_ycb_v1_project.csv；
- 9 个面太少的 contactdb 物体（cube_*/cylinder_*/pyramid_*）没有抓取，网格在 object/contactdb_nograsp/。

**本轮唯一任务（P0，见 §8）**：
1) 跑 §11 的交接检查清单；
2) 把 contactdb 的可行抓取转成 CMap 格式，并与已有的 ycb CMap 数据集**拼接**成一个新文件。
   注意 contactdb 的 object_name 前缀必须是 `contactdb+<obj>`（ycb 是 `ycb+<obj>`）。
   详细要求（输入、输出、去重、校验）见 §8 的任务卡。

不要做 §8 里"暂缓"列表中的事项（重跑 ycb、同步 object1、修 utah_teapot、改 collision mesh 等），
除非我明确要求。
```

---

## 1. 项目一句话

用 **SAPIEN**（无 isaaclab）验证 GraspQP 为 **LinkerHand L20 左手**生成的抓取（`*.dexgrasp.pt`），
筛出物理可行的样本，按 **CMapDataset**（GenDexGrasp / D(R,O) Grasp）格式导出，供后续训练使用。

---

## 2. 当前状态快照（截至 2026-11-10，Ubuntu）

| 项 | 状态 |
|---|---|
| ycb 数据源 | `object/ycb`（旧，28 物体 × 15 文件 × 32 = 13,440）；`object1/ycb`（真实新数据，同样 13,440） |
| ycb 筛选结果 | `filtered_grasps/index_ycb.csv`：object 405/420 + object1 381/420，可行 **3429**（2115 + 1314） |
| ycb CMap 产物 | `cmap_dataset_from_graspqp_v2.pt`：**3429 条 / 28 物体 / `linkerhand`** |
| contactdb 数据源 | 网格修复后**重新生成**了抓取；筛选命令用 `--grasp-path object1/contactdb --source-tag object_contactdb`（见 §7-13，实际目录以服务器为准） |
| contactdb 筛选结果 | 当前 `filtered_grasps/index.csv` + `index_grasps.csv`（`source=object_contactdb`）。**行数未知，需在 Ubuntu 上读 csv 确认** |
| contactdb 无抓取物体 | `cube_small/medium/large`、`cylinder_small/medium/large`、`pyramid_small/medium/large`（9 个，面太少）；网格已移到 `object/contactdb_nograsp/`，`object/contactdb/` 下已删除 |
| 合并副本 | `filtered_grasps/{object,object1}/…/*.pt` 786 个（ycb，已校验）；contactdb 副本在 `filtered_grasps/object_contactdb/…` |
| 旧产物 | `cmap_dataset_from_graspqp.pt`（v1，1276 条，**已过时**）；`cmap_dataset_from_graspqp_v2.pt`（ycb，当前基线） |
| 物体网格 | `object/`：8 个 YCB + 14 个 contactdb 已修复为水密（v3）；`object1/`：**未同步**，仍是原始非水密版本 |
| 手 collision mesh | 21 个原有（未改）+ 1 个新建 `meshes/collisions/thumb_metacarpals.STL`（v3） |
| 备份 | `object/_stl_backup_20261105_234004/`（23 个修复前原件）；`object/contactdb_nograsp/`（9 个无抓取物体的网格）；`graspqp/assets/linkerhand_l20/collisions_backup_20261106_072002/` |

**索引文件命名（容易搞混，务必先 `head -1` 看列、再看 `source` 取值）**

| 文件 | 含义（推断） | 用途 |
|---|---|---|
| `index.csv` | **当前** contactdb 的逐文件索引（`source=object_contactdb`） | 本轮转换 contactdb 用这个 |
| `index_grasps.csv` | **当前** contactdb 的逐抓取索引 | 同上 |
| `index_ycb.csv` | ycb 的逐文件索引备份（object + object1） | 本轮转换 ycb 用这个 |
| `index_ycb_grasps.csv` | ycb 的逐抓取索引备份 | 校验/抽样用 |
| `index_ycb_v1_project.csv` | v1 时代的旧索引（1276 条，**不要**参与合并） | 仅留档 |
| `index_v1_object.csv` | 更早的 v1 object 索引（Windows 侧产物） | 仅留档 |

> 若 `index_ycb.csv` 的行数不是 840、或 `source` 列只有一种取值，说明备份方式与上表不同：
> 先打印行数与 `Counter(source)` 再决定用哪个，**不要盲合并**（见 §7-14）。

## 3. Ubuntu 环境与 Windows 的差异（**先读**）

| 项 | Windows（历史） | Ubuntu（现在） | 需要做的事 |
|---|---|---|---|
| conda 环境 | `genh2r` | **`graspqp`**（完整 GraspQP 栈） | 所有命令用 `conda activate graspqp` |
| GraspQP 依赖 | 只有 roma / pytorch_kinematics / torch / sapien | 全套可用（warp、pytorch3d、graspqp 包等） | 可直接 `import graspqp`；本项目的脚本**不依赖**它，仍可独立跑 |
| isaaclab | 无 | **仍然没有** | 继续用 SAPIEN 3.x |
| Blender | 4.5 + 5.1 并存，4.5 需要 `--factory-startup` | **只有 5.1** | 改 `repair_stls.py` 里的 `BLENDER` 常量；`--factory-startup` 可保留（无害） |
| 路径分隔符 | `\` | `/` | 代码已全部 `pathlib`；唯一硬编码路径是 `BLENDER` |
| 子进程管道 | 沙箱禁止，已改成写日志文件 | 正常可用 | 现有"写文件再读"的方式在 Linux 同样工作，**不必改** |
| `object/` 写权限 | 曾因 Windows ACL 缺 `WRITE_OWNER` 被拒，已修 | 无此问题 | 忽略；`logs/acl_recovery/` 只是历史记录 |
| PhysX GPU | 不可用（`gpu simulation is not initialized`） | **未验证**，可能可用 | 见 §11 检查项；即使可用，单抓取场景 GPU 反而更慢（v2 §5） |

### 必须改的一处常量

`new_grasp_filter/repair_stls.py:41`：

```python
BLENDER = Path(r"D:\Blender_Foundation\Blender_5.1\blender.exe")   # Windows
BLENDER = Path("/usr/bin/blender")                                  # Ubuntu（用 `which blender` 确认）
```

`new_grasp_filter/rebuild_collision_mesh.py`、`audit_*.py` 不含 Blender 路径，无需改。

---

## 4. 目录地图

```
Graspqp-Linkerhand-L20/                     # 仓库根（Ubuntu 上路径会不同，脚本全部相对根目录）
├── object/ycb/<obj>/                        # 旧数据 + 物体资源（<obj>.stl / .urdf / coacd_allinone.obj / grasp_predictions）
│   └── _stl_backup_<ts>/                    # v3 备份（.gitignore 已忽略 object/）
├── object/contactdb/<obj>/                  # 40 个 contactdb 物体（网格 + 重新生成的 grasp_predictions）
├── object/contactdb_nograsp/<obj>/          # 9 个面太少、无法生成抓取的物体（仅网格，无抓取）
├── object1/ycb/<obj>/                       # 真实新数据（只读！未同步 v3 修复）
├── object1/contactdb/<obj>/                 # contactdb 抓取的筛选输入（--grasp-path object1/contactdb；见 §7-13）
├── graspqp/assets/linkerhand_l20/
│   ├── linkerhand_l20.urdf                  # 左手 L20，21 revolute（16 驱动 + 5 mimic）
│   ├── meshes/*.STL                         # 视觉网格（空心壳）
│   └── meshes/collisions/*.STL              # 碰撞网格（凸包 + ~0.8mm），22 个（v3 补齐 thumb_metacarpals）
├── Cmap_dataset/
│   ├── CMapDataset_filtered/cmap_dataset.pt # 参考格式（24764 条，5 个机器人，**无 linkerhand**）
│   ├── data_urdf/robot/linkerhand/left/…    # 左手 extended URDF（**原文件有 bug，见 §7-4**）
│   └── dataset/CMapDataset.py               # 已加 ROBOT_ALIASES（可逆，见 §7-5）
└── new_grasp_filter/                        # 本项目全部产物
    ├── HANDOFF_ubuntu.md                    # ← 本文件
    ├── REPORT.md / REPORT_v2.md / REPORT_v3.md
    ├── grasp_io.py                          # IO + 位姿/FK（torch/roma/pk 默认，numpy fallback）
    ├── sapien_grasp_checker.py              # SAPIEN 场景、接触、提离测试、判据
    ├── object_sdf.py                        # 物体体素 SDF（穿透/接触几何量）
    ├── run_filter.py                        # 批量筛选入口
    ├── convert_to_cmap.py                   # → CMap 格式
    ├── validate_grasp_io.py                 # numpy vs roma/pk 校验
    ├── compare_backends.py                  # 后端一致性
    ├── validate_cmap_dataset.py             # 用 CMapDataset.py 实机验证
    ├── visualize_filtered.py                # 单/批量可视化（sapien/open3d）
    ├── audit_stl.py / repair_stls.py / drop_nested.py        # v3 物体网格审计/修复
    ├── blender_repair/repair_mesh.py         # Blender 修复脚本
    ├── audit_collision_mesh.py / rebuild_collision_mesh.py   # v3 collision 审计/重建
    ├── make_report.py / make_report_v2.py    # 报告统计再生成
    ├── benchmark_cpu_gpu.py                  # CPU/GPU 基准
    ├── filtered_grasps/                      # 当前 index.csv/index_grasps.csv（contactdb）
    │                                         # + index_ycb.csv/index_ycb_grasps.csv（ycb 备份）
    │                                         # + object/ object1/ object_contactdb/ …copies
    ├── logs/                                 # 逐抓取 JSONL、审计 CSV/JSON、基线、修复日志
    ├── vis_output/                           # 可视化截图
    └── assets/                               # 手凸包缓存、修复版 CMap 左手 URDF
```

---

## 5. 核心约定（写代码前必须对齐）

### 5.1 `.pt` 抓取文件

`keys = [values, parameters, grasp_velocities, full_grasp_velocities, grasp_velocities_off, contact_idx, grasp_type, contact_links, root_frame]`

- `values (32,)`：能量；
- `parameters`：16 个驱动关节名 → `(32,)`，另加 `root_pose (32,7) = [tx,ty,tz, qx,qy,qz,qw]`（**xyzw**）；
- `contact_idx (32,12)`：128 个接触候选里的下标；
- `root_pose` 描述 **URDF `root_link`** 的位姿（不是掌心）；
- 关节顺序 = `grasp_io.DRIVE_NAMES`（= 当前 URDF 链顺序），**按名字取用最安全**；
- 5 个 mimic：`*_dip = 0.8917 * *_pip`，`thumb_ip = 1.1619 * thumb_mcp`。

### 5.2 坐标系与 CMap

- 物体在原点、单位米、scale=1；`<obj>.urdf` 只有一个 link。
- CMap `q` = `[x, y, z, roll, pitch, yaw, 16 joints]`，欧拉乘序 **`R = Rx(roll) @ Ry(pitch) @ Rz(yaw)`**（与常见 URDF rpy 相反，已在 CMap 自己的 URDF 上验证）。
- **`q` 的平移/旋转描述"手基座"（`hand_base_link`），不是 `root_link`**：
  `t_base = t_root + R_root @ (-0.03, 0, -0.10)`（`grasp_io.ROOT_TO_HAND_BASE`）。
- `robot_name`：**`linkerhand` = 左手**（本项目数据）；CMap 历史上那个右手身份通过 `ROBOT_ALIASES` 映射成 `linkerhand_right`。

### 5.3 筛选判据（`CheckerConfig`，相对判据）

| 参数 | 值 | 说明 |
|---|---|---|
| `contact_dist` | 2 mm | 手表面点到物体表面 ≤2mm 记为接触 |
| `min_contact_points` / `min_contact_links` | 3 / 2 | |
| `max_penetration` | 15 mm | **只是"是否值得做提离仿真"的闸门，不是物理容差** |
| `lift_height/steps/timestep` | 60mm / 40 / 1/240s | 运动学抬升 |
| `lift_min_rise` | 25 mm | 物体随手上移 ≥25mm 判可行 |

SAPIEN 侧工程要点（血泪教训）：
- **PhysX 不产生 mesh–mesh 接触** → 手必须用凸包（`build_hand_hulls`），物体用动态三角网格（SDF）；
- **静态刚体 `set_pose` 不会推动物体** → 提离必须用 `build_kinematic` + `set_kinematic_target`；
- SAPIEN 的 `separation` 对 mesh-convex 有 ~6mm 偏置 → 毫米级穿透一律用 `object_sdf.VoxelSDF`（负=内部），
  且网格外做了线性外推（v2 修过一个致命 bug，见 §7-2）。

### 5.4 CMap 的 `object_name` 前缀（contactdb 合并的关键）

CMap 用 `<数据集>+<物体名>` 作为 `object_name`：

| 来源 | source tag | CMap `object_name` 示例 |
|---|---|---|
| ycb（旧/新） | `object` / `object1` | `ycb+apple`、`ycb+lemon` |
| contactdb | `object_contactdb` | **`contactdb+alarm_clock`**、`contactdb+banana` |

`convert_to_cmap.py` 目前把前缀写死成 `ycb+`；合并 contactdb 时必须改为按来源映射
（建议：`--prefixes` 参数或"source 含 contactdb → contactdb+"的规则），否则 contactdb 样本会被错误标成 `ycb+…`。

另外：**`cube_*` / `cylinder_*` / `pyramid_*` 这 9 个物体没有抓取**，最终数据集里不会出现
`contactdb+cube_small` 等 9 个名字——这是预期行为，不是丢数据（网格在 `object/contactdb_nograsp/`）。
CMap 参考数据集里虽然有这些物体，但我们的 LinkerHand 数据不覆盖它们。

---

## 6. 三轮工作结论（每轮 3 行）

**v1（REPORT.md）**：首个 SAPIEN 筛选流水线；发现"普遍穿模"——后来证明是 SDF 边界 bug 的假象；产出 1276 条数据集（已过时）。

**v2（REPORT_v2.md）**：
- 用 `roma` + `pytorch_kinematics` 替换/校验 numpy 实现（误差 ~1e-7），保留 `--legacy-numpy`；
- `run_filter.py` 支持 `--grasp-path/--source-tag/--merge-from`；合并 object + object1；
- 修正 CMap 的 hand-base 偏移（v1 少了 10.4 cm）；
- CMap `linkerhand` 语义统一为左手，旧右手身份 → `linkerhand_right`（只改 `CMapDataset.py` 的内存映射）；
- GPU 结论：PhysX GPU 在 Windows 不可用；torch CUDA 只在大 batch 有优势 → 筛选用 CPU。

**v3（REPORT_v3.md）**：
- Blender 门禁双版本通过（5.1 直接可用；4.5 需要 `--factory-startup`）；
- 修复 `object/` 下 22 个非水密 STL（8 YCB + 14 contactdb），体积变化 ≤3.7%、Chamfer ≤0.06mm、SDF ≥99.1%；
  `utah_teapot` 修不好（体积 -5.8%），已还原并记录；
- L20 collision mesh 审计：22 个全部"水密/绕序/无空腔/无自交"，仅体积比 1.7–7.2×（凸包设计）；
  **补齐了唯一缺失的 `thumb_metacarpals.STL`**（这正是 v2 里手 URDF 加载失败的根因）。

---

## 7. 已知问题与坑（编号引用，**别重复踩**）

1. **`object1/` 与 `object/` 现在是两批不同数据**（早期 `object1` 曾是 `object` 的逐字节副本，已由用户替换）。
   不要再用"内容相同"的假设做去重/复用；当前 `index.csv` 的 840 行已逐行 SHA-256 对得上各自源文件。
2. **v1 的体素 SDF 边界 bug**：网格外点被裁剪到边界值，导致"远处的手"也报 5–15mm 穿透 → v1 的
   "普遍穿模"结论不成立。修复在 `object_sdf.query`（裁剪后加"到裁剪位置的世界距离"）。**不要再退回旧实现。**
3. **`trimesh.proximity.closest_point` 在 4.11.3 返回的是无符号距离**；`signed_distance` 才是带符号（正=内部）。
   审计里必须用 `process=True` 加载 STL，否则所有 STL 都会被判"非水密"。
4. **CMap 的左手 extended URDF 原文件是坏的**：`virtual_robot` 指向不存在的 `base_link`，`pytorch_kinematics`
   直接 KeyError。项目用 `grasp_io.cmap_left_urdf()` 生成**修复副本**（`new_grasp_filter/assets/cmap_left_repaired.urdf`，
   只改这一个引用 + mesh 绝对路径），**没有改 CMap 原文件**。
5. **`CMapDataset.py` 的别名规则**：仅当数据集 `info` **未声明** `linkerhand` 时才把它改名为 `linkerhand_right`；
   我们的 v2 数据在 `info` 里声明了 `linkerhand`（左手），所以不会被误改名。回滚方法写在 v2 REPORT §3.2。
6. **`CMapDataset.py` 依赖的 `utils/hand_model.py` 在仓库里不存在**：`validate_cmap_dataset.py` 用 shim 注入
   （基于 pytorch_kinematics，实现左手/右手 URDF 解析）。若上游补齐该模块，需要保持同样的左右手映射。
7. **手侧碰撞用"视觉网格凸包"**：指尖 phantom ~3–4mm、手掌最大 ~18mm。提离测试可能被 phantom 接触"帮一把"；
   已用"精确网格接触门"挡住"完全没接触"的假阳性，但无法完全排除"真实接触 + phantom 额外支撑"。
8. **`thumb_metacarpals` collision mesh 是 v3 新补的**：凸包 + 0.8mm、2.61× visual。若下游换用自己的碰撞体，
   注意它会影响手 URDF 的加载（此前缺失会导致 SAPIEN `File not found`）。
9. **`object1/` 未同步 v3 的网格修复**：同名物体仍是原始非水密版本。两批几何不一致；下游若要求一致需手动同步。
10. **`filtered_grasps` 里曾残留 20 个 stale 副本**（旧 `object1` 内容的遗留，index 未引用），本轮已删除，
    现在 786 个副本与 index 完全一致。
11. **PhysX GPU**：Windows 上 `enable_gpu()` 需 `cuda.dll` 别名且 `step()` 失败。Ubuntu 未验证；若要用，
    必须先 `ctypes.CDLL("libcuda.so", RTLD_GLOBAL)` → `sapien.physx.enable_gpu()` → **再创建任何 Scene**。
12. **CMap 参考数据里没有 `linkerhand` 样本**（24764 条只有 5 个机器人），所以"改名"不涉及历史样本迁移。
13. **contactdb 抓取的实际路径**：用户描述是"重新生成了 `object/contactdb` 的 grasp_predictions"，
    但筛选命令用的是 `--grasp-path object1/contactdb`。两者可能只有一处有文件。
    **先在 Ubuntu 上 `ls`/`find` 确认**，再决定转换时用哪个索引；不要假设。
14. **索引文件被覆盖**：`run_filter.py` 每次都重写 `filtered_grasps/index.csv` 与 `index_grasps.csv`。
    contactdb 那一轮把 ycb 的索引覆盖了，旧内容已备份成 `index_ycb*.csv`。
    合并前必须先确认每个备份的行数与 `source` 取值；`index_ycb_v1_project.csv` 是 v1 旧数据（1276 条），
    **不能**与新 ycb 索引一起合并，否则会重复计数。
15. **9 个 contactdb 物体没有抓取**：`cube_small/medium/large`、`cylinder_small/medium/large`、
    `pyramid_small/medium/large` 因面数太少无法生成抓取，已从 `object/contactdb` 删除（网格在
    `object/contactdb_nograsp/`）。合并后的 CMap 数据集里对应 9 个 `contactdb+*` 名字缺失属正常。

---

## 8. 当前唯一任务（P0）与暂缓事项

> **本轮只做 P0，其余一律不做**（用户 2026-11-10 明确要求）。做完 P0 就停下汇报。

### 8.1 任务卡：contactdb 抓取 → CMap 格式，并与 ycb 的 CMap 数据集拼接

**背景**：contactdb 的网格修复后重新生成了抓取，并用
`run_filter.py --grasp-path object1/contactdb --source-tag object_contactdb --device cpu`
完成了 SAPIEN 筛选。现在要把这批 contactdb 的 feasible 抓取转成 CMap 格式，
再和已有的 ycb 数据集（`cmap_dataset_from_graspqp_v2.pt`，3429 条）**合并成一个文件**。

**输入（都在 `new_grasp_filter/filtered_grasps/`，先在 Ubuntu 上确认存在与行数）**

| 用途 | 文件 | 期望 |
|---|---|---|
| ycb 逐文件索引 | `index_ycb.csv` | 840 行（source = `object` 420 + `object1` 420） |
| ycb 逐抓取索引 | `index_ycb_grasps.csv` | 3429 行 |
| contactdb 逐文件索引 | `index.csv`（当前） | source 全为 `object_contactdb`；行数待读 |
| contactdb 逐抓取索引 | `index_grasps.csv`（当前） | 行数 = contactdb feasible 数 |
| 抓取副本 | `filtered_grasps/object/…`、`object1/…`、`object_contactdb/…` | index 的 `grasp_file` 相对路径 |

**不要用** `index_ycb_v1_project.csv` / `index_v1_object.csv`（v1 旧数据，会造成重复）。

**实现要求（对 `convert_to_cmap.py` 的增量修改，保持向后兼容）**

1. 支持多输入与按来源的前缀：
   - 建议新增 `--inputs <csv> [<csv> ...]` 与 `--prefixes ycb contactdb`（与 `--inputs` 一一对应）；
   - 不传 `--inputs` 时保持现有单输入行为（默认读 `<input>/index.csv`，前缀 `ycb`）；
   - `object_name = f"{prefix}+{obj}"` ⇒ ycb 为 `ycb+apple`，contactdb 为 `contactdb+alarm_clock`。
2. `q` 的生成不变：`grasp_io.root_pose_to_cmap_q(root_pose, joints)`
   （**手基座**语义 + `Rx@Ry@Rz`，不要回退到 v1 的 `root_link` 直转）。
3. 去重：按 `(sha256, grasp_index)`（ycb 的 object/object1 是不同数据，不冲突；contactdb 若重复运行可自愈）。
4. `info` 聚合：`num_total`、`num_upper_object`（单物体最大样本数）、`num_per_object` 覆盖两个前缀。
5. 输出：**新文件** `new_grasp_filter/cmap_dataset_from_graspqp_v3.pt`（ycb+contactdb），
   **不要覆盖 v2**；同时写 `*.meta.json`（`per_source`、`per_object`、`dropped_duplicates`、`prefixes`）。
6. 校验（必须实际运行，不能只看代码）：
   - `torch.load` 成功；顶层 keys 恰为 `['info','metadata']`；`q` 为 `float32 (N,22)`；`robot_name == 'linkerhand'`；
   - `N == 3429 + contactdb_feasible`，物体数 == 28 + 有可行抓取的 contactdb 物体数；
   - 随机抽 20 条 contactdb 样本做 `q` 回代 FK（复用 `validate_cmap_dataset.check_q_roundtrip` 的思路），
     旋转误差应 ~1e-7、平移 ~1e-8；
   - 确认 contactdb 样本的 `object_name` **全部**以 `contactdb+` 开头、ycb 的全部以 `ycb+` 开头；
   - 确认没有 9 个无抓取物体（`cube_*`/`cylinder_*`/`pyramid_*`）的条目。
7. 用 `CMapDataset.py` 实机加载新文件（`validate_cmap_dataset.py --dataset <新文件>`）：
   metadata 可遍历、`__getitem__` 返回 `robot=linkerhand`；参考数据的过滤不受影响。
8. 验证脚本注意：`validate_cmap_dataset.py` 的 validate 模式会读
   `Cmap_dataset/PointCloud/object/<prefix>/<obj>.pt`；contactdb 的 PointCloud 有 50 个文件（覆盖我们 49 个），
   所以 `__getitem__` 对 contactdb 行可用，无需额外 shim。
9. 记录：写 `new_grasp_filter/REPORT_v4_contactdb.md`（简短：输入行数、各来源样本数、物体覆盖、
   字段对照沿用 v2 §4.1、校验输出、已知问题），并在 `logs/` 留一份 JSON 摘要。**不要重写 v1/v2/v3 报告。**

**验收标准**

- 新 `.pt` 能 `torch.load`，结构与 `cmap_dataset.pt` 对齐（`info` + `metadata`，`metadata` 为 3 元组列表）；
- ycb 部分与 v2 **逐样本一致**（同一 `(sha,index)` 的 `q` 完全相同，可直接比较）；
- contactdb 样本前缀正确、数量与 `index.csv` 的 `n_feasible` 之和一致；
- REPORT_v4 里有输入/输出计数、校验结果与遗留问题。

**已知陷阱（务必先看）**

- `index.csv`/`index_grasps.csv` 每次 `run_filter` 都会被覆盖，合并前先确认它们确实只含 contactdb；
- `validate_cmap_dataset.py` 目前默认读 `filtered_grasps/index.csv` 做回代校验，合并后需要加
  `--index` 参数指向对应的逐文件索引，否则会把 contactdb 的 q 和 ycb 的文件配错；
- 9 个无抓取物体不是"丢数据"（见 §5.4、§7-15）。

### 8.2 暂缓事项（本轮**不要**做）

| # | 事项 | 为什么暂缓 |
|---|---|---|
| D1 | 重跑 ycb（8 个修复网格）或 object1 的筛选 | 现有结果可用；重跑约 2–3 小时，等用户明确要求 |
| D2 | 同步 `object1/` 的网格修复 | 会改动只读目录，需用户授权 |
| D3 | 修 `contactdb/utah_teapot` | 需要人工 CAD/调参，非本轮范围 |
| D4 | 用 GraspQP 官方 `HandModel` 交叉验证 SAPIEN | 属于方法学验证，非数据产出 |
| D5 | 把 collision mesh 从凸包升级为精确表面 | 会改变筛选结果，需要重跑 |
| D6 | 力闭合检查 | 判据增强，非本轮范围 |

## 9. 常用命令（Ubuntu 版）

```bash
conda activate graspqp
cd <repo root>            # 例如 ~/Graspqp-Linkerhand-L20

# --- 1) 环境与产物自检
python new_grasp_filter/validate_grasp_io.py          # numpy vs roma/pk 误差表
python new_grasp_filter/validate_cmap_dataset.py      # 用 CMapDataset.py 加载 v2 数据集
python new_grasp_filter/audit_stl.py --tag before     # 物体网格审计（ycb 8 + contactdb 现有 40；9 个已移到 contactdb_nograsp）
python new_grasp_filter/audit_collision_mesh.py --tag before

# --- 2) 可视化（人工检查）
python new_grasp_filter/visualize_filtered.py \
  --grasp-file object1/ycb/apple/grasp_predictions/linkerhand_l20/12_contacts/graspqp/default/apple.dexgrasp.pt \
  --index 3 --backend open3d                 # 交互；无显示器加 --offscreen
python new_grasp_filter/visualize_filtered.py \
  --index-csv new_grasp_filter/filtered_grasps/index_ycb.csv \
  --object apple --source object1 --backend open3d --offscreen
# contactdb 用当前的 index.csv（source 填 object_contactdb）

# --- 3) 【P0】contactdb → CMap 并与 ycb 拼接（先扩展 convert_to_cmap.py 的 --inputs/--prefixes）
# 先确认索引：
python - <<'PY'
import csv
from collections import Counter
for f in ["index.csv", "index_grasps.csv", "index_ycb.csv", "index_ycb_grasps.csv"]:
    try:
        rows = list(csv.DictReader(open("new_grasp_filter/filtered_grasps/" + f)))
        print(f, len(rows), Counter(r.get("source", "") for r in rows))
    except FileNotFoundError:
        print(f, "MISSING")
PY

python new_grasp_filter/convert_to_cmap.py \
    --inputs new_grasp_filter/filtered_grasps/index_ycb.csv \
             new_grasp_filter/filtered_grasps/index.csv \
    --prefixes ycb contactdb \
    --output new_grasp_filter/cmap_dataset_from_graspqp_v3.pt

python new_grasp_filter/validate_cmap_dataset.py \
    --dataset new_grasp_filter/cmap_dataset_from_graspqp_v3.pt

# --- 4) 重新筛选（暂缓；先改 BLENDER 路径不需要）
python new_grasp_filter/run_filter.py --grasp-path object1/ycb --source-tag object1 \
    --merge-from new_grasp_filter/filtered_grasps/index_v1_object.csv --device cpu

# --- 5) 转 CMap + 验证（ycb，既有）
python new_grasp_filter/convert_to_cmap.py \
    --input new_grasp_filter/filtered_grasps \
    --output new_grasp_filter/cmap_dataset_from_graspqp_v2.pt
python new_grasp_filter/validate_cmap_dataset.py

# --- 5) Blender 相关（Ubuntu：只有 5.1）
which blender                                  # 确认路径，然后改 repair_stls.py:41
blender --background --factory-startup --python new_grasp_filter/blender_repair/repair_mesh.py -- \
    --input object/ycb/pear/pear.stl --output /tmp/pear_test.stl
python new_grasp_filter/repair_stls.py --tag before
python new_grasp_filter/drop_nested.py --only contactdb/apple contactdb/flashlight contactdb/mouse
python new_grasp_filter/rebuild_collision_mesh.py --links thumb_metacarpals
```

---

## 10. 关键数字（便于核对）

| 指标 | 值 |
|---|---|
| 抓取总数（每批） | 13,440（28 物体 × 15 文件 × 32） |
| feasible | object **2,115** / object1 **1,314** / 合计 **3,429** |
| contactdb feasible | **待读** `filtered_grasps/index.csv` 的 `n_feasible` 之和（Ubuntu 上确认） |
| contactdb 物体覆盖 | 40 个有抓取（49 − 9 个面太少的）；合并后 CMap 物体数 = 28 + 40 = **68**（以实际 feasible 为准） |
| 合并副本 `.pt` | 786（object 405 + object1 381），SHA-256 全等 |
| CMap 数据集 | 3429 条、28 物体、`linkerhand`、`q=(22,) float32` |
| CMap 结构校验 | `metadata` 为 list[tuple(3)]；`__getitem__` OK；q 回代 FK 旋转 2.4e-7 / 平移 7.4e-9 |
| v3 修复 | 22/23（`utah_teapot` 失败）；体积变化 ≤3.7%、Chamfer ≤0.056mm、SDF ≥99.1% |
| collision mesh | 22 个（21 原有 + 1 新建）；全部水密/绕序一致；体积比 1.73–7.21× |
| 全量筛选耗时 | CPU 约 2.3 小时（v2 实测 8222 s，object1） |
| 单抓取耗时 | ~0.2 s（CPU）/ ~1.7 s（CUDA，小 batch 反而慢） |

---

## 11. 交接检查清单（到 Ubuntu 后**先跑这些**）

```bash
conda activate graspqp
cd <repo root>

# 1) 依赖版本（记录进报告，便于复现）
python - <<'PY'
import sys, torch, sapien, trimesh, numpy, scipy
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("sapien", sapien.__version__, "trimesh", trimesh.__version__, "numpy", numpy.__version__)
import roma, pytorch_kinematics
print("roma", roma.__version__)
PY

# 2) 数据完整性：对**每一个**索引文件校验副本哈希（ycb 备份 + contactdb 当前）
python - <<'PY'
import csv, hashlib
from pathlib import Path
fg = Path("new_grasp_filter/filtered_grasps")
for name in ["index_ycb.csv", "index.csv"]:
    rows = list(csv.DictReader(open(fg / name)))
    bad = n = 0
    for r in rows:
        if not int(r.get("copied") or 0):
            continue
        p = fg / r["grasp_file"]
        n += 1
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
            bad += 1
    from collections import Counter
    print(f"{name}: rows={len(rows)} sources={dict(Counter(r.get('source','') for r in rows))} "
          f"copies={n} hash_mismatch={bad}")
PY
# 期望：index_ycb.csv 840 行（object 420 + object1 420）、0 处不符；
#       index.csv 只含 object_contactdb，且 0 处不符

# 2b) contactdb 无抓取的 9 个物体：在 contactdb_nograsp 里、不在 contactdb 的抓取目录里
ls object/contactdb_nograsp | wc -l            # 期望 9
for o in cube_small cube_medium cube_large cylinder_small cylinder_medium cylinder_large \
         pyramid_small pyramid_medium pyramid_large; do
  test -d object/contactdb_nograsp/$o || echo "MISSING backup: $o"
done

# 3) 手模型可加载（指针：thumb_metacarpals 必须存在）
ls graspqp/assets/linkerhand_l20/meshes/collisions | wc -l     # 期望 22
python - <<'PY'
import sapien
sapien.set_log_level("error")
s=sapien.Scene(); l=s.create_urdf_loader(); l.fix_root_link=True; l.collision_is_visual=False
a=l.load("graspqp/assets/linkerhand_l20/linkerhand_l20.urdf")
print("hand links", len(a.get_links()), "joints", len(a.get_active_joints()))
PY

# 4) SAPIEN GPU 是否可用（可选；不可用就保持 CPU）
python - <<'PY'
import sapien
try:
    sapien.physx.enable_gpu(); print("gpu enabled", sapien.physx.is_gpu_enabled())
except Exception as e:
    print("gpu unavailable:", type(e).__name__, e)
PY

# 5) 数据集可加载 + CMapDataset 验证（v2 基线；合并后加 --dataset 指向新文件）
python new_grasp_filter/validate_cmap_dataset.py --dataset new_grasp_filter/cmap_dataset_from_graspqp_v2.pt
```

**预期**：
- `index_ycb.csv`：840 行（object 420 + object1 420）、副本 hash 0 处不符（786 个）；
- `index.csv`：只含 `object_contactdb`、副本 hash 0 处不符（数量以服务器为准）；
- 手 **23 个 link / 21 个主动关节**（v3 补齐 collision mesh 后实测可加载）；
- `validate_cmap_dataset.py`（v2 基线）：`n_metadata=3429`、`len=28`、`getitem_ok=true`、`q_roundtrip` 旋转 ~1e-7；
- `object/contactdb_nograsp/` 有 9 个目录，`object/contactdb/` 下没有这 9 个物体。

---

## 12. 交接注意事项（给接手的 DSH）

- **本轮只做 §8.1 的 P0**（contactdb→CMap 合并）；§8.2 的暂缓项一律不要主动做。
- **不要重跑全量筛选**：现有 ycb 3429 条是可靠的；contactdb 的筛选结果已在 `index.csv` 里。
- **`object1/` 只读**（用户明确要求过）；任何脚本都不得写入该目录。
- **改代码前先看 REPORT_v2 的 §2.3/§2.5 与 REPORT_v3 的 §3.2**：那里记录了三个会导致"全盘结论反转"的实现陷阱
  （体素 SDF 边界、process=True、探针方向）。
- **报告是累积的**：v1/v2/v3 的数字有先后修订关系（例如 v1 的"穿模"结论已被 v2 §2.3 推翻），以最新报告为准。
- 所有阈值集中在 `CheckerConfig` / `sapien_grasp_checker.py`、`audit_stl.py` 的判定段，改阈值请同步更新 REPORT。
- 产物路径固定：`new_grasp_filter/{filtered_grasps,logs,vis_output}`；不要新建平行目录。
```

---

## 13. 迁移到 Ubuntu 时要带哪些文件

| 路径 | 体积 | 是否必须 | 说明 |
|---|---|---|---|
| `object/`（ycb + contactdb） | 173 MB | ✅ | 物体网格（v3 已修 22 个）+ `_stl_backup_20261105_234004/`（34 MB，修复前原件） |
| `object1/` | 与 object 同量级 | ✅ | 真实新数据（ycb + contactdb 抓取输入），**只读** |
| `object/contactdb_nograsp/` | 小 | ✅ | 9 个无抓取物体的网格备份（重建/复核用） |
| `object1/contactdb/` | 与 ycb 同量级 | ✅ | contactdb 抓取的筛选输入（`--grasp-path object1/contactdb`） |
| `filtered_grasps/index_ycb*.csv` | < 2 MB | ✅ | ycb 的索引备份（合并必需的输入） |
| `filtered_grasps/object_contactdb/` | 视数量 | ✅ | contactdb 的可行抓取副本 |
| `graspqp/assets/linkerhand_l20/` | 约 40 MB | ✅ | 手模型（含 v3 新增的 `thumb_metacarpals.STL`） |
| `new_grasp_filter/*.py` + `*.md` | < 1 MB | ✅ | 全部代码与三份报告 |
| `new_grasp_filter/filtered_grasps/` | 94 MB | ✅ | 786 个副本 + 两个 index.csv（**数据集可由此重建**） |
| `new_grasp_filter/cmap_dataset_from_graspqp_v2.pt` (+ `.meta.json`) | 1.3 MB | ✅ | 最终产物 |
| `new_grasp_filter/logs/` | 193 MB | 建议 | 逐抓取 JSONL（可追溯判定的唯一证据）；若空间紧张，至少带 `index*.csv`、`*_audit_*.csv/json`、`summary.json`、`stl_repair_*`、`collision_*` |
| `new_grasp_filter/assets/` | 2.3 MB | ✅ | 手凸包缓存 + 修复版 CMap 左手 URDF（删了会重新生成，但 CMap 左手 URDF 是必需的） |
| `new_grasp_filter/_blender_probe/` | 61 MB | ❌ | 含门禁探针、控制组输出、Blender 日志；只保留 `probe.py` 与 `control/` 即可 |
| `new_grasp_filter/vis_output/` | 1.4 MB | ❌ | 可视化截图，可重生成 |
| `Cmap_dataset/` | — | ✅ | 参考数据集、CMap 左右手 URDF、`CMapDataset.py`（含 v2 的 `ROBOT_ALIASES` 改动） |

**最小可复现集合**：`object/`（或其中用到的 28 个 ycb 物体）+ `object1/` + `graspqp/assets/linkerhand_l20/` +
`Cmap_dataset/` + `new_grasp_filter/`（去掉 `_blender_probe/`、`vis_output/`、可选 `logs/`）。

**Ubuntu 上需要重建/重设的**：
1. `repair_stls.py:41` 的 `BLENDER` 路径；
2. 若 `tags`/权限不同，`chmod` 确保 `filtered_grasps/`、`logs/` 可写；
3. 若用 GPU，先按 §11-4 验证 SAPIEN PhysX GPU，再决定 `--device cuda`（否则保持 `cpu`）。
