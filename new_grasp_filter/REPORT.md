# GraspQP → SAPIEN → CMapDataset：LinkerHand L20 抓取筛选报告

> 本目录（`new_grasp_filter/`）中的**全部**代码、日志、结果均为本次新增文件，任务过程中**没有修改**
> 任何已有文件。`git status` 里另外几项（`.gitignore`、`linkerhand_l20.urdf` 的 `M`，
> `linkerhand_l20/collisions/` 的删除，`meshes/collisions/` 的未跟踪）**在任务开始前就已存在**，
> 与本次工作无关；`object/` 与 `Cmap_dataset/` 完全未动。
> （证据：这些文件的 mtime 为 2026-11-04 17:08–17:27，早于本次会话开始的 18:26；
> `new_grasp_filter/` 下所有文件均创建于 18:27 之后。）

---

## 0. 直接可复制的命令序列

```bash
# 0) 环境（本机只有 genh2r；任务描述里的 graspqp 环境即 genh2r）
conda activate genh2r
cd /d/study/VScodes/Graspqp-Linkerhand-L20

# 1) 冒烟测试：单个物体、单个文件、前 8 个抓取
python new_grasp_filter/run_filter.py --object apple --grasp apple.dexgrasp.pt --limit 8

# 2) 全量筛选（28 个物体 / 420 个 .pt / 13440 个抓取）
python new_grasp_filter/run_filter.py

# 3) 转 CMapDataset 格式
python new_grasp_filter/convert_to_cmap.py \
    --input  new_grasp_filter/filtered_grasps \
    --output new_grasp_filter/cmap_dataset_from_graspqp.pt
```

辅助命令：

```bash
# 只跑 smoke（1 个物体 1 个文件的前 8 个抓取）
python new_grasp_filter/run_filter.py --object apple --grasp apple.dexgrasp.pt --limit 8

# 由已有日志/索引重新生成 REPORT 里的统计与 sanity check（不重新仿真）
python new_grasp_filter/make_report.py

# 可选：把 uncertain 的抓取也写进 CMap 数据集
python new_grasp_filter/convert_to_cmap.py --include-uncertain
```

---

## 1. 调研（阶段 0）

### 1.1 运行环境

| 项 | 值 | 说明 |
|---|---|---|
| conda 环境 | `genh2r` | 本机 `conda env list` 会因 `multiprocessing` 权限问题崩溃；改用 `D:\anaconda3\envs\genh2r\python.exe` 直接调用，未安装/升级任何包 |
| Python | 3.10.20 | |
| SAPIEN | **3.0.3**（CPU `PhysxCpuSystem`） | 因此全部代码使用 SAPIEN 3.x API（`create_actor_builder` / `scene.get_contacts` / `PhysxRigidDynamicComponent`） |
| torch | 2.7.0+cu128（仅 CPU 使用） | 只用于 `torch.load` 与 `torch.save` |
| 其他 | numpy 1.26.4, trimesh 4.11.3, scipy 1.15.2 | |
| 缺失 | `roma`、`pytorch_kinematics`、GraspQP 全栈 | 均**未安装**。位姿/四元数/FK 全部用 numpy 自行实现（`grasp_io.py`），并在报告里给出校验方式 |

> 结论：**没有向任何环境安装任何包**，也没有创建新环境。

### 1.2 物体资源

每个 `object/ycb/<object>/` 包含：

| 文件 | 内容 |
|---|---|
| `<object>.stl` | 物体网格，单位 **米**，水密（apple: 8194 顶点 / 16384 面 / 体积 2.47e-4 m³） |
| `<object>.urdf` | 单 link `object`，`<mesh filename="<object>.stl" scale="1 1 1"/>`，无 inertial、无关节 |
| `coacd_allinone.obj` | CoACD 重网格版本；包围盒/体积与 STL 完全一致（apple 均为 75.4×74.9×71.9 mm、2.48e-4 m³） |
| `grasp_predictions/linkerhand_l20/12_contacts/graspqp/default/*.pt` | 15 个文件/物体：`<object>.dexgrasp.pt` + `*_step_500..7000.dexgrasp.pt`，每个 32 个抓取 |

Sapien 加载物体**不需要额外变换**：世界系 = 物体系，物体固定在原点、单位米、scale=1。
（`ObjectModel` 在 GraspQP 里先找 `coacd/remeshed.obj`，再 `coacd/decomposed.obj`，否则 glob `*.obj`；
本数据集只有 `coacd_allinone.obj`，所以优化时用的是它。）

> **注意（重要）**：`coacd_allinone.obj` 是多个凸块的并集，内部面穿过物体内部，因此
> `trimesh.contains` / signed distance 在它上面是**无意义**的（原点处 signed distance≈0，
> 实测）。本项目的碰撞与几何查询一律使用水密的 `<object>.stl`。

### 1.3 `.pt` 抓取文件结构

以 `apple.dexgrasp.pt` 为例（`torch.load(..., weights_only=False)`）：

| key | 类型/shape | 含义（推断 + 代码验证） |
|---|---|---|
| `values` | float32 (32,) | 每个抓取的 GraspQP 能量（按升序更有可能是好抓取，但下文会看到它与可行性弱相关） |
| `parameters` | dict | 16 个驱动关节名 → float32 (32,)，外加 `root_pose` |
| `parameters[root_pose]` | float32 (32,7) | `[tx,ty,tz, qx,qy,qz,qw]`，**xyzw** 四元数 |
| `contact_idx` | int64 (32,12) | 12 个接触候选点在 128 个候选中的下标 |
| `grasp_velocities` | dict 16×(32,) | 由接触法向求出的**闭合**关节位移 `delta_theta`（DexGraspNet 语义） |
| `full_grasp_velocities` | dict 16×(32,) | 用全部 128 个候选求的闭合位移（数值可到 ±9 rad，未裁剪，不建议直接执行） |
| `grasp_velocities_off` | dict 16×(32,) | 带 offset 的变体 |
| `grasp_type` / `root_frame` / `contact_links` | str / str / None | `'all'` / `'gripper_root'` / None |

**root_pose 的四元数序与坐标系**：`scripts/vis/visualize_result.py:636` 用
`roma.unitquat_to_rotmat(root_pose[..., [4,5,6,3]])`，即把存储序重排为 `(w,x,y,z)`，所以存储序是
**xyzw**；随后 `hand_params = [t, R.mT.flatten(1,2), joints]`，而 `HandModel.set_parameters` 里
`global_rotation` 重建回 `R`，网格变换为 `x_world = R @ x_link + t`。
→ **`.pt` 的 root_pose 把 URDF 根 link（`root_link`）放到物体/世界系**。

**关节顺序**：`graspqp/hands/linkerhand_l20.py` 里
`DRIVE_NAMES = [pinky_mcp_roll, pinky_mcp_pitch, pinky_pip, ring_..., middle_..., index_..., thumb_cmc_yaw, thumb_cmc_roll, thumb_cmc_pitch, thumb_mcp]`，
且 `parameters` 是以这些**名字**为 key 存储的，因此按名字取用不存在歧义。
5 个耦合关节按 URDF `<mimic>` 展开（`dip = 0.8917*pip`，`thumb_ip = 1.1619*thumb_mcp`），
与 `linkerhand_l20.py::_build_full_angles` 完全一致（已逐行复刻到 `grasp_io.build_full_angles`）。

### 1.4 LinkerHand L20 手模型

* 文件：`graspqp/assets/linkerhand_l20/linkerhand_l20.urdf`（`<robot name="linkerhand_l20_left">`），
  mesh 目录 `graspqp/assets/linkerhand_l20/meshes/*.STL`（视觉网格，米）。
* 关节：**21 个 revolute** = 16 驱动 + 5 mimic（`*_dip`、`thumb_ip`）。链顺序 = 文件顺序 =
  `pinky(roll,pitch,pip,dip) → ring → middle → index → thumb(yaw,roll,pitch,mcp,ip)`。
* 根：`root_link --fixed--> hand_base_link`，固定变换 `xyz=(-0.03, 0, -0.10)`；
  FK 在零位时 `index_distal` 位于 `z=+0.1358 m`、`thumb_distal` 位于 `y=-0.125 m`，
  与 `docs/linkerhand_l20_notes.md` 记录的"fingertips at z≈+0.136 m / +Y 手指展开"一致 → **FK 与仓库文档自洽**。
* 与 CMap 的右手 URDF（`Cmap_dataset/data_urdf/robot/linkerhand/right/...`）对比：
  CMap 版是**右手镜像**（`index_mcp_roll` 的 y 从 -0.0267 变为 +0.0267，thumb 链 base1/base2 顺序也不同）。
  本数据集由 GraspQP 的**左手**模型生成，因此坐标系按左手解释。

**关节顺序假设**：`.pt` 的关节按 `DRIVE_NAMES`（= 当前 URDF 链顺序）。该假设由
`linkerhand_l20.py::_check_joint_layout` 在仓库内强制保证；本机无 `pytorch_kinematics`，
改用自写 FK 复算，并用"最低能量抓取的 4 个指尖曲面到物体表面距离 = 0.00 mm"这一独立现象交叉验证
（见 §2.4）。若未来 URDF 链顺序变化，`grasp_io.CHAIN_JOINT_NAMES` 必须同步。

### 1.5 CMapDataset 目标格式

`Cmap_dataset/CMapDataset_filtered/cmap_dataset.pt` 顶层只有两个 key：

```
{'info': {robot_name: {'robot_name', 'num_total', 'num_upper_object', 'num_per_object'}},
 'metadata': [ (q: float32[22], object_name: str, robot_name: str), ... ]}   # 24764 条
```

* `metadata` 是 **list of tuple**，`q` 为 `torch.float32`、shape `(22,)`；`object_name` 形如
  `ycb+apple` / `contactdb+alarm_clock`；`robot_name` ∈ {allegro, barrett, ezgripper,
  robotiq_3finger, shadowhand}（本数据集新增 `linkerhand`，与
  `data_urdf/robot/urdf_assets_meta.json` 里的 key 一致）。
* `q` 的 22 维 = `[x, y, z, roll, pitch, yaw, j1..j16]`：
  前 3 维是平移（实测 allegro 的 `q[:,0:3]` 在 ±0.19 内，米），
  `q[:,3]∈[-π,π]`、`q[:,4]∈[-1.56,1.56]`、`q[:,5]∈[-π,π]`（欧拉角），后 16 维是关节角。
* 欧拉角约定：CMap 的 extended URDF 虚拟链是
  `virtual_joint_x → y → z → roll → pitch → yaw → fixed → base_link`，各关节 origin 为单位阵，
  因此 `R = Rx(roll) @ Ry(pitch) @ Rz(yaw)`（**与常见 URDF rpy 的乘序相反**）。
  该约定已用 `grasp_io.UrdfModel` 在 CMap 的 `linkerhand_l20_right_extended.urdf` 上数值验证：
  给定任意旋转，解出的 rpy 代回链上 FK 得到的 `base_link` 旋转**误差 0.0**（见 `convert_to_cmap.matrix_to_rpy_cmap`）。
* `info[robot]['num_upper_object']` 在原数据里等于该机器人"单物体最大样本数"（allegro=185 = `ycb+orange` 的 185 条），
  本转换沿用该语义。

---

### 1.6 本阶段结论 + 遗留问题

**结论**：数据格式、坐标系、关节顺序、CMap 目标结构全部可从现有代码/数据中确定，无需猜测：

* `.pt` = `root_pose`（xyzw 四元数，作用在 `root_link`）+ 16 个按名字存储的驱动关节角；
* 手模型是**左手** L20（URDF 名 `linkerhand_l20_left`，21 个 revolute，5 个 mimic）；
* CMap 的 `q` = 3 平移 + 3 欧拉角（`Rx@Ry@Rz`）+ 16 关节，已在 CMap 自己的 URDF 上验证乘序。

**遗留问题**：

1. CMap 的 `linkerhand` 是**右手** URDF，而 GraspQP 数据是左手；`q` 只是按关节名对齐，
   下游用 CMap 右手模型复现时需要镜像（本次未做）。
2. 数据生成时使用的手模型版本无法确认（工作区 URDF 在生成之后被修改过），
   这直接关系到 §5.1 的穿模问题。

## 2. 验证方案（阶段 1）

### 2.1 实现结构

| 文件 | 作用 |
|---|---|
| `grasp_io.py` | 读取 `.pt`、四元数/欧拉角、URDF 解析与批量 FK、mimic 展开、资产路径 |
| `object_sdf.py` | 物体**体素 SDF**（2 mm），用于毫米级穿透/接触判定 |
| `sapien_grasp_checker.py` | SAPIEN 场景、接触、提离测试、判据与阈值 |
| `run_filter.py` | 批处理、逐抓取 JSONL 日志、二进制拷贝 + SHA-256 校验、`index.csv` |
| `convert_to_cmap.py` | 转 CMap 格式 |

### 2.2 SAPIEN 场景设计（以及为什么这样设计）

对每个物体构建**一个可复用场景**（每个抓取只重置位姿，重建成本 1 次/物体）：

1. **手**：`linkerhand_l20.urdf` 的 22 个 link，每个 link 用其视觉网格的**凸包**作为碰撞体，
   `build_kinematic()` 建成**运动学刚体**，由 `set_kinematic_target()` 逐帧驱动。
   * 为什么不用三角网格：PhysX **不产生 triangle-mesh vs triangle-mesh 接触**（实测：
     静态手网格 + 动态物体网格重叠也无任何 contact）；动态三角网格（SDF）与**静态凸体**之间
     才产生接触（实测通过）。所以至少一侧必须是凸体。
   * 为什么手用凸包而不是物体用凸包：物体用动态三角网格时是**精确几何**，代价只是手侧近似。
   * 凸包近似误差（凸包表面到视觉网格的距离，随机采样 400 点/测）：
     指尖 2.97–4.34 mm，指节 4.75–6.30 mm，`hand_base_link` 最大 18.47 mm（均值 3.11 mm），
     `thumb_metacarpals_base2` 最大 10.69 mm。→ 手侧近似在**毫米级**，会影响接触点位置，
     但不会造成 10 mm 级别的假穿透（见 §2.4 与 §5 的已知问题）。
2. **物体**：`<object>.stl` 的**动态三角网格**（SAPIEN 3 的 SDF collision），质量由密度 1000 自动计算
   （apple 0.248 kg）。
3. **接触**：`scene.get_contacts()`；contact_offset 3 mm。

**静态 vs 运动学（踩过的坑）**：最初用 `build_static()` + `set_pose()` 驱动手，提离测试里物体
**完全不动**（自由落体）。用"静态地板 + 小球"最小实验复现：静态地板每次 `set_pose` 上升 1.5 mm，
小球穿过地板掉到 z=-63 mm；改成 `build_kinematic()` + `set_kinematic_target()` 后，小球随地板升到
z=+75 mm（= 地板 60 + 半厚 5 + 半径 10，完全正确）。→ **提离测试必须用运动学刚体**，已修正。

### 2.3 穿透深度：为什么用体素 SDF 而不是 SAPIEN 的 `separation`

用"静态球 vs 动态苹果网格"做了标定（球半径 20 mm，已知间隙）：

| 标称间隙 | SAPIEN 报出的 `separation`（contact_offset=3 mm） |
|---|---|
| +20 mm | 无接触 |
| +5 mm | 无接触 |
| 0 mm | **+5.96 mm** |
| −3 mm | +3.29 … +4.63 mm |
| −10 mm | −2.75 … −2.27 mm |

即 `separation ≈ 真实间隙 + 6 mm`（≈ 2×contact_offset），且 mesh-convex 接触在浅穿透时根本不生成。
把 contact_offset 降到 0.5–2 mm 后接触生成更不稳定（−3 mm 时仍无接触）。
→ **SAPIEN 的 `separation` 无法用于毫米级穿透判据**，只能用于"有没有接触"。
因此：穿透深度用**精确网格 + 2 mm 体素 SDF**（`object_sdf.VoxelSDF`，误差：对随机表面点
均值 1.9 mm、最大 15 mm；与 `trimesh.contains` 的符号一致率 94.7%，不一致点都在表面 ±2 mm 内）。
SAPIEN 仍然负责**物理**（接触、提离稳定性），这正是任务要求"优先用 Sapien 做物理验证"的部分。

### 2.4 位姿解码的交叉验证

用"最低能量抓取（apple, E=1.942）"检查 4 个指尖 link 的曲面到物体表面最近距离：
`index/middle/ring/pinky_distal` 均为 **0.00–0.03 mm**。若四元数序、FK、root link 约定任一错误，
不可能出现 4 个指尖同时贴合。→ 位姿解码正确。
（同一抓取的 `contact_idx` 选中的 link 是 middle/ring/pinky 的 proximal/middle，
与几何上贴合/穿入的 link 一致，见 §5。）

### 2.5 判据与阈值（`CheckerConfig`）

| 参数 | 值 | 依据 |
|---|---|---|
| `sdf_pitch` | 2 mm | 精度/速度折中（单物体建 0.1–6 s，查询 <1 ms/抓取） |
| `contact_dist` | 2 mm | 手表面点到物体表面 ≤2 mm 记为接触（SDF 误差约 2 mm） |
| `min_contact_links` | 2 | 至少两个不同 link 接触（单点接触不构成抓取） |
| `min_contact_points` | 3 | 去重后的手表面采样点中至少 3 个在 2 mm 内 |
| `max_penetration` | **15 mm** | **不是物理容差**：仅作为"是否值得做提离仿真"的界限，见 §5 已知问题 |
| `lift_height` / `lift_steps` / `timestep` | 60 mm / 40 / 1/240 s | 0.167 s 内手上升 60 mm（≈0.36 m/s）；自由落体约 136 mm，区分度足够 |
| `lift_min_rise` | 25 mm | 明显高于"被挤压抖动"量级（实测失败抓取的 rise ≤ 0 mm，成功 47–92 mm） |
| `explosion_dist` | 1 m | 数值爆炸 → `uncertain` |
| `contact_offset` | 3 mm | 仅用于 SAPIEN 接触生成 |

**判定流程（每个抓取）**

1. 按 `root_pose`/关节角摆好手，重置物体；
2. 精确几何：手表面采样点（每 link ≤300 点）到物体 SDF 的距离 → 接触点/接触 link/**穿透深度**；
3. 接触不足 / link 不足 → `insufficient_contacts` / `too_few_contact_links`；
4. 穿透 > 15 mm → `penetration`（跳过提离仿真，`lift_skipped=1`）；
5. SAPIEN settle 5 步，记录接触（advisory）；
6. **提离测试**：开启物体重力，手在 40 步内上升 60 mm（运动学），测物体 z 位移；
7. 判据：`rise ≥ 25 mm` → `feasible`；否则 `slips_on_lift`；仿真发散 → `uncertain`。

---

### 2.6 本阶段结论 + 遗留问题

**结论**：

* SAPIEN 3.0.3 可以在这台 Windows 机器上完整跑通"手-物接触 + 提离"，
  但必须满足两个工程约束：**(a)** 至少一侧是凸体（PhysX 不产生 mesh-mesh 接触）；
  **(b)** 手必须用 `kinematic` + `set_kinematic_target` 驱动（`static` + `set_pose` 不会推动物体）。
* SAPIEN 的 `separation` 对 mesh-convex 有 ~6 mm 偏置，毫米级穿透必须改用精确网格 + 体素 SDF。

**遗留问题**：

1. 手侧凸包会把掌部放大（最大 18 mm），`sapien_*` 接触只作参考。
2. 未做力闭合检查；提离测试只验证"能否承住"，不验证接触力是否可达。
3. 单进程 CPU，全量约 50 分钟；未做多进程（Windows + Sapien 场景隔离成本高）。

## 3. 结果与统计（阶段 2）

> 数据由 `run_filter.py` 全量跑出：28 个物体 / 420 个 `.pt` / 13440 个抓取。
> 逐抓取日志：`new_grasp_filter/logs/<object>.jsonl`；汇总：`logs/summary.json`；
> 机读索引：`filtered_grasps/index.csv`。



### 3.1 `filtered_grasps/` 的组织与拷贝语义

```
new_grasp_filter/filtered_grasps/
  index.csv
  <object_name>/linkerhand_l20/12_contacts/graspqp/default/<original_file_name>.pt
```

* **拷贝单元是文件，判定单元是抓取**。一个 `.pt` 内含 32 个抓取；只要其中有 ≥1 个
  `feasible`（或 `uncertain`，由 `--include-uncertain` 控制，默认开），就整文件
  `shutil.copy2` 到目标目录，并**用 SHA-256 校验与源文件逐字节一致**。
* `index.csv` 的 `feasible_indices` / `uncertain_indices` 给出该文件内真正通过的下标，
  下游（含 `convert_to_cmap.py`）只用这些下标。这样既满足"文件二进制一致"，又不会把
  未通过的抓取混进 CMap 数据集。
* 实测：**405 / 420** 个文件被拷贝（15 个文件内没有任何可行抓取），
  **405 个副本全部通过 SHA-256 与源文件比对（0 处不一致）**。
* 由于一个文件含 32 个抓取，只要 1 个通过就整文件拷贝，所以"文件级"筛选几乎是全通过；
  真正有意义的是 `feasible_indices`（逐抓取）。

### 3.2 `.pt` 字段与原始 GraspQP 输出的关系

**筛选产物就是原始文件的逐字节副本**，没有新增/派生字段。判定信息全部在外部日志里：

| 文件 | 字段 |
|---|---|
| `logs/<object>.jsonl` | `object, grasp_file, grasp_index, energy, verdict, reason, metrics{geo_*, sapien_*, lift_*}, contact_links[], sapien_contacts[]` |
| `filtered_grasps/index.csv` | `object, grasp_file, n_grasps, n_feasible, n_uncertain, verdict, feasible_indices, uncertain_indices, reason_counts, copied, sha256` |

`metrics` 中关键量的物理含义：

* `geo_min_distance_mm`：手表面采样点到物体表面的最近距离（负=穿透），
* `geo_penetration_mm`：最深的穿透深度（>0 表示手在物体内部），
* `geo_contact_points` / `geo_contact_links`：≤2 mm 的采样点数 / 不同 link 数，
* `sapien_*`：SAPIEN 的接触数（含其 6 mm 偏置，仅参考），
* `lift_rise_mm` / `lift_drift_mm`：提离测试中物体的 z 位移 / 水平漂移。

---

<!-- STATS:BEGIN -->
### 3.3 全量统计

* 验证抓取总数：**13440**（28 个物体 / 420 个 `.pt`，每个 32 个抓取）
* `feasible`：**2115**  (15.7%)
* `uncertain`：**0**
* `infeasible`：**11325**
* 拷贝文件数：**405** / 420（全部经 SHA-256 校验）
* 有可行抓取的物体：**28** / 28

**失败原因分布**

| reason | count | share |
|---|---:|---:|
| `penetration` | 6042 | 45.0% |
| `slips_on_lift` | 4449 | 33.1% |
| `ok` | 2115 | 15.7% |
| `insufficient_contacts` | 624 | 4.6% |
| `too_few_contact_links` | 210 | 1.6% |

**关键指标分布**

| 指标 | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 穿透深度 geo_penetration_mm | 0.0 | 2.0 | 6.0 | 13.5 | 25.4 | 33.2 | 61.4 |
| 接触 link 数 | 0.0 | 6.0 | 10.0 | 16.0 | 20.0 | 21.0 | 22.0 |
| 提离位移 lift_rise_mm | -341.2 | -218.1 | -139.7 | -85.5 | 51.2 | 61.7 | 193.4 |

> 提离测试实际执行了 7203 个抓取；其余因穿透 >15 mm 被跳过（`lift_skipped=1`，这类抓取不可能通过任何合理阈值）。

* 能量 vs 判定：feasible 的 energy 中位数 6.46，infeasible 的 6.30 → 能量**不能**单独作为可行性判据。

**逐物体统计**

| object | grasps | feasible | rate | files copied | top reason |
|---|---:|---:|---:|---:|---|
| apple | 480 | 73 | 15.2% | 15 | `penetration` (243) |
| baseball | 480 | 68 | 14.2% | 15 | `slips_on_lift` (224) |
| bleach_cleanser | 480 | 34 | 7.1% | 15 | `penetration` (302) |
| cracker_box | 480 | 8 | 1.7% | 8 | `penetration` (443) |
| foam_brick | 480 | 29 | 6.0% | 15 | `penetration` (286) |
| gelatin_box | 480 | 73 | 15.2% | 15 | `penetration` (240) |
| hammer | 480 | 132 | 27.5% | 15 | `slips_on_lift` (292) |
| lemon | 480 | 37 | 7.7% | 14 | `slips_on_lift` (274) |
| master_chef_can | 480 | 66 | 13.8% | 15 | `penetration` (304) |
| mini_soccer_ball | 480 | 76 | 15.8% | 15 | `penetration` (281) |
| mustard_bottle | 480 | 87 | 18.1% | 15 | `penetration` (331) |
| orange | 480 | 73 | 15.2% | 15 | `slips_on_lift` (189) |
| peach | 480 | 84 | 17.5% | 15 | `slips_on_lift` (244) |
| pear | 480 | 70 | 14.6% | 15 | `slips_on_lift` (168) |
| phillips_screwdriver | 480 | 99 | 20.6% | 15 | `slips_on_lift` (274) |
| pitcher_base | 480 | 11 | 2.3% | 8 | `penetration` (402) |
| plum | 480 | 64 | 13.3% | 15 | `slips_on_lift` (285) |
| potted_meat_can | 480 | 93 | 19.4% | 15 | `penetration` (211) |
| power_drill | 480 | 61 | 12.7% | 15 | `penetration` (274) |
| pudding_box | 480 | 75 | 15.6% | 15 | `penetration` (319) |
| rubiks_cube | 480 | 94 | 19.6% | 15 | `penetration` (243) |
| sponge | 480 | 143 | 29.8% | 15 | `slips_on_lift` (336) |
| strawberry | 480 | 49 | 10.2% | 15 | `slips_on_lift` (338) |
| sugar_box | 480 | 42 | 8.8% | 15 | `penetration` (391) |
| tomato_soup_can | 480 | 120 | 25.0% | 15 | `penetration` (228) |
| toy_airplane | 480 | 131 | 27.3% | 15 | `penetration` (245) |
| tuna_fish_can | 480 | 186 | 38.8% | 15 | `slips_on_lift` (279) |
| wood_block | 480 | 37 | 7.7% | 15 | `penetration` (385) |
<!-- STATS:END -->

---

## 4. 转换为 CMapDataset（阶段 3）

### 4.1 字段对照表

| CMap 字段 | shape / dtype | 本数据集来源 | 说明 |
|---|---|---|---|
| 顶层 `info` | dict | 由样本聚合 | 与 `cmap_dataset.pt` 同构 |
| `info[robot]["robot_name"]` | str | 常量 `"linkerhand"` | 与 `urdf_assets_meta.json` 的 key 一致 |
| `info[robot]["num_total"]` | int | `len(metadata)` | |
| `info[robot]["num_upper_object"]` | int | `max(num_per_object)` | 沿用原数据"单物体最大样本数"语义 |
| `info[robot]["num_per_object"]` | dict[str,int] | `ycb+<object>` → 计数 | |
| `metadata` | list[tuple] | 见下 | 长度 = `num_total` |
| `metadata[i][0]` `q` | float32[22] | **派生**：`[tx,ty,tz, roll,pitch,yaw, j1..j16]` | 前三+欧拉由 `root_pose`（xyzw 四元数）转出；rpy 约定见 §1.5，已数值验证 |
| `metadata[i][1]` | str | `"ycb+" + <object_dir>` | |
| `metadata[i][2]` | str | `"linkerhand"` | |

**无对应来源、需要填充/派生的字段**：

* `q[3:6]`（欧拉角）是**派生值**：CMap 只存欧拉角，GraspQP 只存四元数，转换不可逆地丢掉了
  万向节锁附近的信息（`|pitch|→π/2` 时 `roll` 置 0，实际抓取里未出现）。
* `robot_name="linkerhand"` 是**新取值**（原数据没有这个机器人）；`object_name` 前缀统一为
  `ycb+`，与原数据中 YCB 物体命名一致。
* **手模型**：CMap 的 `linkerhand` 是**右手** URDF，而 GraspQP 数据来自**左手**模型。
  `q` 的 16 个关节按**名字**一一对应（URDF 顺序一致），但左右手是镜像关系：
  同一 `q` 代回 CMap 的右手模型会得到镜像的姿态。**本数据集因此标注为 left-hand L20**，
  下游若用 CMap 的右手 URDF 复现，需要额外做一次镜像变换（未在本次范围内）。
* `q[0:3]` 的平移是**物体坐标系**下的 `root_link` 位置（物体在原点），与 CMap 的
  "物体在原点"约定一致；`root_pose` 的四元数/欧拉角描述 `root_link` 相对物体系的姿态。

### 4.2 转换前后 sanity check

本节内容由 `make_report.py` 自动生成于 §6.1（可 load 性、`q` 逐维分布、欧拉角回代校验）。

---

### 4.3 本阶段结论 + 遗留问题

**结论**：`cmap_dataset_from_graspqp.pt` 的顶层结构、`metadata` 元组、`q` 的 shape/dtype
与 `cmap_dataset.pt` 完全一致；`q` 的欧拉角已按 CMap 自己的 URDF 验证过的约定生成
（回代旋转误差 ~1e-15）。转换是**内容幂等**的（排序固定、只依赖 `index.csv` + 副本文件；
连续两次运行得到的 2115 条样本逐条相同）。注意 `torch.save` 的 zip 容器会写入时间戳，
因此两次输出的**文件字节**不同，但张量内容一致。

**遗留问题**：

1. 左右手镜像（见 §1.6）。
2. `num_upper_object` 在原数据里是"单物体最大样本数"，本数据集沿用该语义，但没有官方文档背书。
3. 本数据集没有 `heatmap` 变体（`cmap_dataset_heatmap.pt` 需要逐样本计算，超出本次范围）。

## 5. 已知问题与假设（**必读**）

### 5.1 主要发现：这批 GraspQP 抓取在仓库自带手模型下**普遍穿模**

这是本次验证最重要的结论，也是全部判定的背景：

* apple 32 个抓取的手表面采样点中，**中位数 15.4 mm、最大 29.6 mm** 的穿透深度
  （`geo_penetration_mm` 分布见 §3.3）；中位抓取有 29% 的手表面采样点在物体内部。
* 交叉验证：最低能量抓取（E=1.942）的 `contact_idx` 选中的 link
  （`middle_proximal / middle_middle / ring_proximal / ring_middle / pinky_proximal`）
  **正是几何上穿入物体最深的那些 link**（深度 10–25 mm），这不是随机现象。
* 截面可视化（`_scratch/slice_apple_15.png`）显示：手指从物体上方斜穿进苹果内部。
* 已排除的解释：
  * 四元数序/FK/root link 约定错误 —— 4 个指尖同时贴合 0.00 mm，且 `docs/linkerhand_l20_notes.md`
    记录的零位几何完全一致；
  * 全局平移偏置 —— 沿手部 z 轴扫描 0–60 mm，最优偏移随抓取变化（20/35/45/50/60 mm 都有），
    且最优处仍有大量顶点在物体内；
  * 全局缩放 —— 手整体缩放到 0.55–1.25 倍、物体缩放到 1.0–2.0 倍都**不能**把穿透降到 0
    （手放大只会让指尖更靠外、但仍然穿模；物体放大只会更深）。
* 未排除的假设（**无法在本机验证**）：
  1. 生成数据时的相机/手模型（URDF、网格尺度）与当前 `graspqp/assets/linkerhand_l20/` 不一致；
     该 URDF 在工作区里于数据生成之后被修改过（`git status` 显示 `M`，diff 只涉及碰撞文件路径，
     但无法排除生成时用的是另一个分支/版本）。
  2. 生成时的 `E_pen` 项失效（`docs/linkerhand_l20_notes.md` 记录了 "inside-out mesh" 与
     "pseudo-normal sign" 两个会导致穿透被奖励的 bug），导致优化器把手指推进物体。
* **因此 `max_penetration=15 mm` 只用作"是否值得做提离仿真"的界限，不作为物理容差**。
  真正决定 `feasible` 的是 SAPIEN 的提离测试（物理）。这在 §3.3 的统计里表现为：
  一部分穿透 5–15 mm 的抓取仍能真实地把物体提起来，而穿透 >15 mm 的抓取无法验证。

### 5.2 其他限制

* **提离测试是运动学抬升**，不是完整机械臂/驱动仿真：手是运动学刚体，手指关节不发力，
  因此它只检验"几何包围 + 摩擦能否承住物体"，不能检验力闭合/接触力是否可达。
  （任务允许"提离/扰动/力闭合"三选一，本次选提离。）
* **未实现力闭合检查**（接触法向锥）。原计划若时间允许再补，未完成。
* 手侧凸包近似：指尖 3–4 mm、掌部最大 18 mm 的"幽灵体积"，可能造成少量假接触/假稳定；
  `sapien_contact_*` 只作参考，判定主依据是几何接触 + 提离结果。
* SAPIEN mesh-convex 的 `separation` 有约 6 mm 偏置（§2.3），故未用于穿透判据。
* 体素 SDF 在物体表面附近有约 2 mm 量化误差，与 `trimesh.contains` 一致率 94.7%。
* **8/28 个物体的 `<object>.stl` 不是水密网格**（`gelatin_box, mini_soccer_ball,
  mustard_bottle, pear, power_drill, sponge, tomato_soup_can, tuna_fish_can`；`trimesh`
  的 `merge_vertices + fill_holes + fix_normals` 也无法修复）。对它们 SAPIEN 会打印
  `SDF creation: Mesh is not suitable for SDF ...` 的 critical 警告，动态三角网格碰撞的
  可靠性下降，这些物体的判定置信度**低于**其余 20 个物体。日志里保留了该警告。
* `.pt` 的 `full_grasp_velocities` 数值可达 ±9 rad（未裁剪），本次**未执行**任何闭合动作；
  验证的是文件里保存的位姿本身。
* 每个 `.pt` 的 15 个 step 文件都参与了验证；它们是同一优化过程的不同迭代快照，
  因此统计里同一物体的抓取并非互相独立。
* 运行时间：全量约 40–60 分钟（CPU，单进程）。`--limit` / `--object` / `--grasp` 可缩短。

### 5.3 假设清单

| # | 假设 | 验证方式 / 状态 |
|---|---|---|
| A1 | `.pt` 关节向量顺序 = `DRIVE_NAMES`（= 当前 URDF 链顺序） | 由 `linkerhand_l20.py` 强制；FK 结果与文档零位几何一致 ✅ |
| A2 | `root_pose` 四元数为 xyzw，作用在 URDF `root_link` 上 | 由 `visualize_result.py` 代码路径 + 4 指尖贴合 0.00 mm ✅ |
| A3 | 物体在物体坐标系原点、scale=1 | 由 `<object>.urdf` + `ObjectModel` 代码路径确认 ✅ |
| A4 | CMap `q[3:6]` 为 `Rx@Ry@Rz` 欧拉角 | 在 CMap extended URDF 上 FK 验证，旋转误差 0.0 ✅ |
| A5 | 数据集的手是左手 L20，CMap 目标格式是右手 | URDF 名 `linkerhand_l20_left` + 关节 origin 的 y 符号 ⚠️（下游需镜像） |
| A6 | 生成时的手模型/几何与当前仓库一致 | **未验证**（数据穿模，见 §5.1）❌ |

---

## 6. 自动生成的结果片段

<!-- AUTO:BEGIN -->
### 6.1 转换结果与 sanity check

* 顶层 keys：`['info', 'metadata']`（参考数据：`['info', 'metadata']`）
* 样本数：**2115**；物体覆盖：**28** 个；`num_upper_object`=186
* `metadata` 类型：`list`，元素类型：`tuple`，长度 3（参考数据同样为 3 元组）
* `q`：shape `(2115, 22)`，dtype `torch.float32`（参考：`(N,22)`, `torch.float32`）

**逐维数值范围（本数据集 vs 参考 CMap 的 linkerhand 无对照，改用同维度统计）**

| dim | 本数据 min | 本数据 max | 本数据 mean | 参考 allegro min | 参考 allegro max |
|---|---:|---:|---:|---:|---:|
| 0 | -0.1204 | +0.1258 | +0.0047 | -0.1486 | +0.1604 |
| 1 | -0.1931 | +0.2085 | +0.0045 | -0.1569 | +0.1555 |
| 2 | -0.1904 | +0.1689 | +0.0109 | -0.1886 | +0.1764 |
| 3 | -3.1017 | +3.1308 | +0.0310 | -3.1416 | +3.1413 |
| 4 | -1.5303 | +1.5223 | -0.0495 | -1.5608 | +1.5530 |
| 5 | -3.1258 | +3.1405 | -0.2169 | -3.1410 | +3.1395 |
| 6 | -0.2265 | +0.8155 | +0.2892 | -0.4710 | +0.4706 |
| 7 | -0.1316 | +0.4579 | +0.1497 | -0.2202 | +1.6814 |
| 8 | -0.1379 | +0.7723 | +0.2604 | -0.2473 | +1.7162 |
| 9 | -0.2060 | +0.5501 | +0.1312 | -0.2270 | +1.6180 |
| 10 | -0.2229 | +0.5125 | +0.1774 | -0.4759 | +0.4722 |
| 11 | -0.2485 | +1.0175 | +0.3576 | -0.2643 | +1.6245 |
| 12 | -0.5129 | +0.3441 | -0.0320 | -0.1991 | +1.7151 |
| 13 | -0.1687 | +0.6539 | +0.2283 | -0.2274 | +1.6183 |
| 14 | -0.1746 | +1.1671 | +0.3904 | -0.4873 | +0.4725 |
| 15 | -0.8288 | +0.2458 | -0.1853 | -0.2345 | +1.6235 |
| 16 | -0.2298 | +0.6118 | +0.1593 | -0.2002 | +1.7388 |
| 17 | -0.1406 | +1.1930 | +0.3610 | -0.2281 | +1.6185 |
| 18 | +0.0380 | +1.1083 | +0.6212 | +0.2367 | +1.3965 |
| 19 | -0.6756 | +0.9040 | +0.0771 | -0.1069 | +1.1721 |
| 20 | +0.0096 | +0.7569 | +0.4592 | -0.2506 | +1.6442 |
| 21 | -0.1087 | +0.8535 | +0.3626 | -0.1623 | +1.7203 |

**抽样（前 3 条）**

* `ycb+apple` / `linkerhand`: q = [-0.0979, -0.0064, -0.0128, -2.9528, +1.4135, -0.0299, …] (22 维)
* `ycb+apple` / `linkerhand`: q = [-0.0327, +0.0894, -0.0142, +0.0217, +0.4068, -1.2687, …] (22 维)
* `ycb+apple` / `linkerhand`: q = [-0.0118, +0.0410, -0.1085, +0.4392, -0.0251, -0.3710, …] (22 维)

**欧拉角回代校验（随机 200 条：`Rx@Ry@Rz` 重建旋转 vs 源四元数）**

* 最大旋转重建误差：**2.26e-15**（200 条抽样）
<!-- AUTO:END -->

---

## 7. 脚本与参数说明

### `run_filter.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--object` | 全部 28 个 | 只跑单个物体 |
| `--grasp` | 全部 15 个文件 | 只跑单个 `.pt`（需配 `--object`） |
| `--limit` | 无 | 每个 `.pt` 只验证前 N 个抓取（冒烟用） |
| `--max-files-per-object` | 无 | 每个物体最多取前 N 个文件 |
| `--output-dir` | `new_grasp_filter/` | 日志/拷贝/index 的根目录 |
| `--max-penetration-mm` | 15 | 见 §2.5（仅决定是否做提离） |
| `--lift-min-rise-mm` | 25 | 提离判定阈值 |
| `--include-uncertain` / `--no-include-uncertain` | 开 | 是否把 `uncertain` 的文件也拷到 `filtered_grasps/` |

### `convert_to_cmap.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | `new_grasp_filter/filtered_grasps` | 阶段 2 的输出目录（含 `index.csv`） |
| `--output` | `new_grasp_filter/cmap_dataset_from_graspqp.pt` | 输出文件 |
| `--include-uncertain` | 关 | 是否也导出 `uncertain` 抓取 |

### `make_report.py`

无参数；读取 `logs/*.jsonl`、`filtered_grasps/index.csv`、`cmap_dataset_from_graspqp.pt`，
重新生成 REPORT 中的统计与 sanity check 段落（幂等）。

### `sapien_grasp_checker.py` / `object_sdf.py`

作为库使用（`SapienGraspChecker(object_name, CheckerConfig())`），阈值集中在 `CheckerConfig` 数据类，
没有隐藏常量。
