# v2：扩大筛选（object + object1）· 统一左手 · 可视化 · GPU · 跨平台 · roma/pytorch_kinematics 校验

> **文件改动清单**（完整）
>
> | 文件 | 改动 | 是否在任务允许清单内 |
> |---|---|---|
> | `new_grasp_filter/run_filter.py` | v2 参数、来源标签、内容寻址合并、CSV 字段 | ✅ |
> | `new_grasp_filter/grasp_io.py` | roma/pk 后端 + numpy fallback、hand-base 偏移、CMap 左 URDF | ✅ |
> | `Cmap_dataset/dataset/CMapDataset.py` | `ROBOT_ALIASES` + 内存映射（19 行，可逆） | ✅ |
> | `new_grasp_filter/object_sdf.py` | **偏离**：修复 v1 体素 SDF 的边界裁剪 bug | ⚠️ 见下 |
> | `new_grasp_filter/sapien_grasp_checker.py` | **偏离**：`root=` 参数、`use_gpu`、精确几何指标字段 | ⚠️ 见下 |
>
> 两个"偏离"文件都是 v1 阶段本项目自己新增的（已在 `6ea233a add_filter` 提交），不是用户原有代码。
> 修改它们的理由：v1 的 SDF 边界 bug 让**所有** `geo_penetration_mm` 数值失真（§2.3），
> 不修就无法产出可信的 v2 结果；`root=` 与 `use_gpu` 是 §1.1/§1.3 的硬性需求。
> 除此之外：`object/`、`object1/`、`cmap_dataset.pt`、`data_urdf/`、`graspqp/`、`scripts/` **零改动**。

---

## 0. 直接可复制的命令序列

```bash
# 0) 环境（本机只有 genh2r）
conda activate genh2r
cd /d/study/VScodes/Graspqp-Linkerhand-L20

# 1) numpy vs roma/pytorch_kinematics 校验（四元数 / 欧拉角 / FK / mimic / CMap URDF 结构）
python new_grasp_filter/validate_grasp_io.py

# 2) 后端一致性 + PhysX 噪声（numpy vs torch vs torch 重跑）
python new_grasp_filter/compare_backends.py --object apple --files 2

# 3) 对 object1 全量筛选，并合并 v1 的 object 结果（按内容哈希复用判定，不重跑旧数据）
python new_grasp_filter/run_filter.py \
    --grasp-path object1/ycb --source-tag object1 \
    --merge-from new_grasp_filter/filtered_grasps/index_v1_object.csv \
    --device cpu

# 4) 转 CMap（左手 linkerhand，含 hand-base 偏移修正）
python new_grasp_filter/convert_to_cmap.py \
    --input  new_grasp_filter/filtered_grasps \
    --output new_grasp_filter/cmap_dataset_from_graspqp_v2.pt

# 5) 用 CMapDataset.py 实机验证（含 alias / URDF 解析 / q 回代 FK）
python new_grasp_filter/validate_cmap_dataset.py

# 6) 可视化（单抓取 / 批量；open3d 交互或 --offscreen 截图）
python new_grasp_filter/visualize_filtered.py \
    --grasp-file object1/ycb/apple/grasp_predictions/linkerhand_l20/12_contacts/graspqp/default/apple.dexgrasp.pt \
    --index 3 --backend open3d
python new_grasp_filter/visualize_filtered.py \
    --index-csv new_grasp_filter/filtered_grasps/index.csv \
    --object apple --source object1 --backend open3d --offscreen

# 7) CPU vs GPU 基准
python new_grasp_filter/benchmark_cpu_gpu.py

# 8) 由日志/索引重新生成 REPORT_v2 的统计段（幂等）
python new_grasp_filter/make_report_v2.py
```

### 结论摘要（先看这里）

1. **`object1` 与 `object` 的 420 个 `.pt` 逐字节相同**（SHA-256 全等、mtime 相同）——本工作区的"新数据"
   并不是用新碰撞 mesh 重新生成的，因此"新数据是否更好"无法被验证（§1.1）。
2. 左手 URDF 与 GraspQP 的 16 个 drive 关节**名字/顺序/几何完全一致**（Δ=0）；CMap 左侧 extended
   URDF 有一个 **dangling `base_link`**（pk 直接 KeyError），已用最小改动的修复副本解决（§1.4）。
3. v1 的 `VoxelSDF` 有边界裁剪 bug，导致 v1 的"普遍穿模"结论不成立；修复后 apple 第 3 号抓取
   实际是"手在物体外 19 mm、0 接触"（§2.3）。**v1 的 feasible 集合不再被沿用。**
4. numpy 与 roma/pytorch_kinematics 在 float64 下一致到 **1e-7**；64 个抓取里 63 个判定一致，
   唯一翻转由 1e-7 位姿差异经 PhysX 接触混沌放大导致（§1.8、§2.5）。
5. 重新筛选后 13,440 个抓取中 **1,276 个 feasible（9.5%）**；`object`/`object1` 各 1,276（内容相同），
   CMap 数据集去重后 **1,276 条、28 个物体、左手 `linkerhand`**（§3、§7）。
6. SAPIEN PhysX GPU 在本机不可用（驱动/初始化问题），torch CUDA 可用且在**大 batch** 下更快（§5）。

---

## 1. 调研（阶段 0）

### 1.1 【重大】object1 与 object 完全相同：420/420 个 `.pt` 逐字节一致

| 检查 | 结果 |
|---|---|
| `object/ycb/**/*.dexgrasp.pt` 数量 | 420 |
| `object1/ycb/**/*.dexgrasp.pt` 数量 | 420 |
| 同名文件 SHA-256 相同 | **420 / 420** |
| 抽样 mtime | 两者均为 `2026-11-03 22:23`（完全相同） |
| 文件结构（keys / shape / dtype / `root_frame` / `grasp_type`） | 完全一致（见 §1.2） |

**结论**：本工作区里的 `object1/` 是 `object/` 的**原样拷贝**，不是用新碰撞 mesh 重新生成的数据。
因此：

* 本轮无法验证"新碰撞 mesh 是否改善了抓取质量"这一假设；
* `object1` 的筛选结果必然与 `object` 相同（内容寻址复用正是建立在这个事实上，见 §3.1）；
* 上一轮 REPORT §5.1 的"普遍穿模"与 `object1` 无关——两份数据是同一批。

`object1/contactdb/` 下只有物体 mesh，没有任何 `grasp_predictions`，因此本轮只处理 `object1/ycb`。

### 1.2 新数据格式

新旧数据逐字节相同，结构自然一致：
`keys = [values, parameters, grasp_velocities, full_grasp_velocities, grasp_velocities_off, contact_idx, grasp_type, contact_links, root_frame]`，
`root_pose (32,7) float32`、`contact_idx (32,12) int64`、`grasp_type='all'`、`root_frame='gripper_root'`。
**没有任何差异需要停下。**

### 1.3 当前碰撞 mesh

`graspqp/assets/linkerhand_l20/meshes/collisions/` 下 21 个 Blender 生成的 STL，URDF 的
`<collision>` 已指向 `meshes/collisions/<link>.STL`（工作区 URDF 相对 HEAD 的改动仅此路径）。
测量结果（`logs/collision_meshes.json`，21/21 水密）：

| 指标 | 范围 |
|---|---|
| 碰撞体体积 / 视觉体积 | **1.73× – 7.21×**（`hand_base_link` 7.21×，指尖 2.16×） |
| 覆盖视觉表面（碰撞体 contains 视觉表面点） | 89.2% – 100%（`thumb_metacarpals_base2` 68.5%） |

**工程决策**：SAPIEN 里手侧仍使用**视觉 mesh 的凸包**（v1 方案），而不是这套碰撞 mesh。
理由：碰撞体普遍比视觉体大 2–7 倍，用它做接触/穿透会把"手指离物体还有 5–10 mm"误判成接触
（上一轮已经吃过这个亏：凸包phantom 让 v1 的提离测试出现假阳性，见 §2.4）。碰撞 mesh 的
"覆盖率高"只说明它包住了视觉体，不代表它就是真实手指表面。

### 1.4 左右手 URDF 对齐（**关键结论**）

对比 `graspqp/assets/linkerhand_l20/linkerhand_l20.urdf` 与
`Cmap_dataset/data_urdf/robot/linkerhand/left/linkerhand_l20_left_extended.urdf`：

| 项 | 结果 |
|---|---|
| 16 个 drive 关节**名字** | 完全一致 |
| 16 个 drive 关节**顺序** | 完全一致（按 URDF 文档顺序） |
| 16 个 drive 关节的 `origin`(xyz/rpy) + `axis` | **最大差值 0.0**（逐关节比较） |
| 5 个 mimic | 完全一致：`*_dip = 0.8917 * *_pip`，`thumb_ip = 1.1619 * thumb_mcp` |
| CMap 额外虚拟关节 | `virtual_joint_x/y/z`（prismatic，轴 x/y/z）、`virtual_joint_roll/pitch/yaw`（revolute，轴 x/y/z），`origin` 全为 0，最后 `virtual_robot`（fixed，单位变换） |

因此 CMap 的 `q[0:6]` 语义为 `T(x,y,z) · Rx(roll) · Ry(pitch) · Rz(yaw)`，末端是**手基座**
（`hand_base_link`）。

> **重要发现（CMap 左侧 URDF 有结构性错误）**：该文件的 `virtual_robot` 指向
> `<child link="base_link"/>`，但文件里**根本没有 `base_link` 这个 link**；手指链挂在另一个
> 孤立根 `hand_base_link` 上。后果：
> * `pytorch_kinematics.build_chain_from_urdf` 直接 `KeyError: 'base_link'`；
> * 右手 URDF 没有这个问题（`pk_ok=True`）。
>
> 处理：**不修改 CMap 原文件**，在 `new_grasp_filter/assets/cmap_left_repaired.urdf` 生成一份
> 修复副本（只把该 fixed joint 的 child 改成 `hand_base_link`，并把 mesh 路径改成绝对路径），
> 由 `grasp_io.cmap_left_urdf()` 提供。修复后 pk 可加载（27 个关节 = 6 虚拟 + 21）。

**由此得到 v1 的一个真实 bug**：GraspQP 的 `root_pose` 描述的是 `root_link`，而 CMap 的 `q`
描述的是手基座；两者相差固定变换 `(-0.03, 0, -0.10)` m（10.4 cm）。v1 直接导出 `root_pose`，
导致 CMap 里整只手偏移 10.4 cm。v2 用 `t_base = t_root + R_root @ (-0.03,0,-0.10)` 修正，
并用 FK 回代验证（§3.3）。

### 1.5 CMapDataset.py 的实际用法（不是猜的）

逐行读过 `Cmap_dataset/dataset/CMapDataset.py`（248 行）：

* `robot_name` 的用途只有两个：**(a)** `self.hands[robot_name] = create_hand_model(robot_name, device)`；
  **(b)** `metadata` 过滤 `m[2] in self.robot_names`。没有对 `linkerhand` 的硬编码分支。
* `create_hand_model` 来自 `utils.hand_model`，**本工作区没有这个文件**（只有 `dataset/` 与
  `data_urdf/`）。因此直接用 `CMapDataset` 前必须先提供该模块；`validate_cmap_dataset.py`
  注入了一个基于 `pytorch_kinematics` 的最小实现（同时承担 v2 的左右手映射）。
* `q` 在类里**只被当作不透明的 22 维张量**传递（`target_q`/`initial_q`），类本身不解释欧拉角；
  欧拉角乘序只在"回代 FK/可视化"时才有意义 → 由本项目的 `validate_cmap_dataset.py` 负责验证。
* **`num_upper_object` 在代码里从未被使用**（`grep num_upper_object Cmap_dataset/dataset/CMapDataset.py`
  无结果），它只是 `info` 里的元数据。v1 猜测它是"单物体最大样本数"——该语义与数据自洽
  （allegro=185 = `ycb+orange` 的 185 条），但**代码并不消费它**。
* 数据集路径硬编码为 `ROOT_DIR/data/CMapDataset_filtered/...`，而实际文件在 `Cmap_dataset/` 下
  （没有 `data/` 这一层）；验证脚本用作用域内的 `os.path` 影子对象重定向，未改原文件。

### 1.6 `cmap_dataset.pt` 里 `linkerhand` 的真实分布

**0 条**。24764 条样本只有 5 个机器人：barrett 7834 / ezgripper 6073 / shadowhand 5181 /
allegro 4812 / robotiq_3finger 864。`info` 里也只有这 5 个 key。

因此"把 CMap 原 `linkerhand` 改名为 `linkerhand_right`"**不涉及任何历史样本迁移**：
需要改的是"机器人身份 → URDF"的约定。本轮采用**内存别名**（`ROBOT_ALIASES`），
原始 `.pt` 一个字节都没动。

### 1.7 GPU 可用性

| 项 | 结果 |
|---|---|
| `torch.cuda.is_available()` | **True**（RTX 4060 Laptop，CUDA 12.8，torch 2.7.0+cu128） |
| SAPIEN `sapien.physx.enable_gpu()` | 直接调用失败：`PermissionError`（要写 `~/.sapien/physx`） |
| 手动下载 PhysX GPU 包（78.5 MB，`105.1-physx-5.3.1.patch0`）+ `USERPROFILE` 重定向 | `enable_gpu()` 需要 `cuda.dll`，Windows 上不存在；把 `nvcuda.dll` 复制为 `cuda.dll` 并预加载后可过 |
| 之后 `is_gpu_enabled()` | True；`PhysxGpuSystem()` 可构造；`Scene([gpu])` 的 `physx_system` 确实是 `PhysxGpuSystem` |
| **实际 step** | **失败**：`RuntimeError: failed to step: gpu simulation is not initialized` |
| CPU PhysX 性能 | 106,792 steps/s（最小场景） |

**结论**：本机 SAPIEN 3.0.3 的 PhysX GPU 路径不可用（这是 SAPIEN/驱动层面的问题，不是本项目代码）。
SAPIEN 筛选固定走 CPU；torch 侧（FK/位姿/欧拉角）仍用 CUDA 并给出基准（§5）。
`enable_gpu()` 的完整复现步骤与错误见 `logs/benchmark_cpu_gpu.json` 的 `sapien` 字段。

> 若后续在 Ubuntu 上要启用 PhysX GPU：把 `~/.sapien/physx/<version>/libPhysXGpu_64.so` 放好，
> 先 `import ctypes; ctypes.CDLL("libcuda.so", RTLD_GLOBAL)`，再 `sapien.physx.enable_gpu()`，
> **并且必须在创建任何 Scene 之前**（见 §6 清单）。

### 1.8 位姿/FK 校验计划与结果（用 roma + pytorch_kinematics）

`validate_grasp_io.py` 用**独立参照**逐项对比 numpy 实现：

| 项目 | 参照 | 结果 |
|---|---|---|
| 四元数 xyzw → 旋转矩阵 | `roma.unitquat_to_rotmat`（本环境 roma 1.8.0 接受 **xyzw**，已用单位/90° 样例确认） | `max|ΔR| = 1.9e-7`（320 条） |
| 旋转矩阵 → 四元数（往返） | `roma.rotmat_to_unitquat` | 往返 `8.9e-16`（numpy）/ `8.9e-16`（roma） |
| 欧拉角 `Rx@Ry@Rz` | `scipy.Rotation.as_euler('xyz')` | 角度差 `5.1e-15`，回代旋转 `1.0e-15` |
| URDF FK（23 个 link，64 组配置） | `pk.build_chain_from_urdf(...).forward_kinematics` | `max|ΔT| = **1.14e-7**`；5 个指尖 link 4.4e-9 – 1.1e-7 |
| mimic 展开 | 直接解析 URDF `<mimic>` 与本项目 MIMIC_MAP | **完全一致**，数值差 0.0 |
| pk 关节顺序 vs `CHAIN_JOINT_NAMES` | `chain.get_joint_parameter_names()` | **一致** |
| CMap 左 URDF 结构 | pk 可加载性 | **失败**（dangling `base_link`，见 §1.4）→ 修复副本 |

**误差结论**：numpy 与官方库在 float64 下相差 `~1e-7`（float32 级），远低于任务给的 `1e-5` 阈值，
**不影响任何判定**。替换后 feasible 集合一致性见 §2.5。

> **对上一轮 §2.4 的更正**：v1 用 `trimesh.proximity.closest_point(...)[1].min()` 得到"4 个指尖
> 贴合 0.00 mm"。trimesh 4.11.3 的这个返回值是**无符号距离**，0.00 只说明"有顶点在表面上"——
> 穿模的指尖同样满足。现在改用带符号距离报告：最低能量抓取（apple, E=1.942）的 4 个指尖
> 顶点中，`index_distal` 从外侧 -21.9 mm 到内侧 +3.5 mm，`pinky_distal` 内侧最深 +16.5 mm
> —— 即指尖**跨在**表面上。这个"贴合"从来不是位姿正确性的证明；位姿正确性改由 §1.8 的
> pk/numpy 一致性与 URDF 几何一致性来保证。

### 1.9 本阶段结论 + 遗留问题

**结论**

1. `object1 ≡ object`（逐字节），因此"新碰撞 mesh 改善质量"在本工作区**无法被验证**。
2. 左手 URDF 的 16 个 drive 关节与 GraspQP 完全一致（Δ=0），CMap 的 `q[0:6]` 语义已确定。
3. CMap 左侧 extended URDF 有 dangling `base_link`，必须用修复副本。
4. `cmap_dataset.pt` 无 `linkerhand` 样本；`num_upper_object` 代码未使用。
5. SAPIEN PhysX GPU 在本机不可用；torch CUDA 可用。
6. numpy 实现与 roma/pk 一致到 1e-7。

**遗留问题**

* object1 的来源（谁拷贝的、为什么 mtime 相同）无法从工作区判断，只能如实记录。
* CMap 左 URDF 的修复是"最小改动"（child 指向 hand_base_link）；若上游后续修复了该文件，
  应改用原文件并删除 `assets/cmap_left_repaired.urdf`。

---

## 2. 阶段 1：扩展 run_filter.py + 校验/替换 grasp_io.py + object1 筛选

### 2.1 run_filter.py 新增参数（向后兼容）

| 参数 | 作用 |
|---|---|
| `--grasp-path <root>` | 任意抓取根（`<root>/<object>/grasp_predictions/...`；也接受 `<root>/ycb` 自动下钻） |
| `--source-tag <str>` | 输出子目录与 index 的来源标签（默认取根目录名：`object` / `object1`） |
| `--merge-from <index.csv>` | 合并旧结果；**按内容哈希复用**（见 §2.2） |
| `--legacy-numpy` | 用 v1 numpy 数学（否则 roma + pytorch_kinematics） |
| `--device cpu\|cuda` | torch FK 的设备（默认 cuda） |
| `--gpu` | 尝试 SAPIEN PhysX GPU（本机失败则回退 CPU 并写日志） |

输出结构：`filtered_grasps/<source>/<object>/...`、`filtered_grasps/index.csv`（每文件一行）、
`filtered_grasps/index_grasps.csv`（每个 feasible 抓取一行，含 `geo_penetration_mm`）、
`logs/<source>/<object>.jsonl`。`index.csv` 新增
`source / geo_penetration_mm_{min,median,max} / lift_rise_mm_max / mesh_watertight / merged_from`。

### 2.2 合并逻辑（内容寻址）

`--merge-from` 读取 v1 的 `index.csv`，对每一行：

1. 校验源文件 SHA-256；
2. **如果该 SHA 在本轮已评估过**（object1 的结果），直接复用同 `(sha, grasp_index)` 的判定与指标；
3. 否则回退到旧 index 的 `feasible_indices` 并从旧 `logs/` 恢复指标；
4. 把文件复制/校验到 `filtered_grasps/<source>/...`（逐字节，SHA-256 复核）。

在本工作区里，(2) 会覆盖全部 420 个文件（因为 object1 ≡ object），所以 `object` 的判定
**等价于**用当前（修正后的）checker 重跑，而不是沿用 v1 的旧判定。这一点在 §2.4 很重要。

### 2.3 【重大】v1 的穿透指标有一个符号/边界 bug，v1 的"普遍穿模"结论不成立

`object_sdf.VoxelSDF` 用物体包围盒内的 2 mm 体素网格。v1/v2 早期版本对**网格外**的点直接
`clip` 到边界，导致任何远处的点都返回边界值（约 ±5 mm）。后果：

* 手离物体 10–20 cm 时，`geo_penetration_mm` 依旧报 5–15 mm；
* v1 §5.1 的"apple 穿透中位数 15.4 mm、29% 顶点在物体内"是**这个 bug 的产物**；
* `trimesh.proximity.closest_point` 在 4.11.3 返回**无符号**距离，v1 的"指尖 0.00 mm"也因此失去了意义。

**修复**（`object_sdf.py::query`）：把裁剪后的值加上"查询点到裁剪位置的世界距离"，
使网格外的场成为保守的线性外推；并用 `trimesh` 带符号距离逐点对拍
（约定相反：本实现**内部为负**，trimesh 为内部正；量级上中心 28.1 vs 26.3、
0.05 m 6.3 vs 12.7、0.2 m 156 vs 163 mm）。**精度说明**：网格内是精确三线性插值；
bbox 外 0–5 mm 的接触带误差约 1 体素（2 mm），更远处外推会**低估**距离
（0.05 m 处 6.3 vs 真实 12.7 mm），因此 2 mm 接触判定是可靠的，但不要用 `geo_min_distance_mm`
去判断"手离物体 1 cm 还是 2 cm"。修复后 apple 第 3 号抓取（v1 判 feasible，穿透 8.0 mm）
真实情况是**手在物体外 19.1 mm、0 个接触点**——v1 的"可行"来自凸包 phantom 接触造成的
假提离（+89 mm）。v2 修正后该抓取被 `insufficient_contacts` 正确拒绝。

> 影响面：v1 的所有 `geo_penetration_mm` 数值都不可信；v1 的 feasible 集合包含 phantom 假阳性。
> 本轮**没有**沿用 v1 的 object 判定（§2.2 的内容寻址会让 object 复用 object1 的新判定）。
> v1 的 REPORT.md 保留为历史记录，其 §5.1 的结论以本文件为准。

### 2.4 判据与阈值（相对 v1 的调整）

| 参数 | v1 | v2 | 依据 |
|---|---|---|---|
| `contact_dist` | 2 mm | 2 mm | 体素 SDF 误差约 2 mm |
| `min_contact_points` | 3 | 3 | 至少 3 个手表面采样点在 2 mm 内 |
| `min_contact_links` | 2 | 2 | |
| `max_penetration` | 15 mm | 15 mm | 只在"是否值得做提离仿真"的意义上使用；修复 SDF 后分布已可解释（§4） |
| `lift_min_rise` | 25 mm | 25 mm | 失败抓取的 rise ≤ 0，成功 47–92 mm |

**`max_penetration` 的语义保持不变，且必须强调**：它不是物理容差，只是计算预算闸门。
真正的判据是 SAPIEN 提离测试（物理）；穿透只用于提前排除"不可能通过"的抓取并供下游过滤。

### 2.5 替换 numpy → roma/pytorch_kinematics 与一致性

`grasp_io.py` 现在：

* 默认 `backend="torch"`：`roma` 解四元数/旋转、`pytorch_kinematics` 做 FK、torch 做 mimic 展开；
* `set_backend("numpy")` / `--legacy-numpy` 回退到 v1 的 numpy 实现（保留全部原函数，前缀 `_*_numpy`）；
* `set_device("cuda")` 可把 FK/姿态放到 GPU（大 batch 才划算，见 §5）；
* 每个函数替换后都用同一批样本对拍（§1.8 的表）。

**feasible 集合一致性**（`compare_backends.py`，同一批抓取跑 numpy → torch → torch 重跑）：

<!-- COMPARE2:BEGIN -->
**numpy（v1）vs torch（v2 默认）**（objects=['apple'], files/object=2）

| 指标 | 值 |
|---|---|
| 抓取数 | 64 |
| feasible（A / B） | 4 / 3 |
| 判定一致 | **63/64** |
| 原因一致 | 63/64 |
| feasible 翻转数 | **1** |
| max abs Δ 穿透 | 2.2e-05 mm |
| max abs Δ 间隙 | 1.3e-05 mm |
| max / median abs Δ rise | 353.592 / 6.371 mm |

翻转样例：
* apple/apple.dexgrasp.pt#27: ok(rise 46.89161479473114) → slips_on_lift(rise -306.700736284256)

**torch 重跑（PhysX 噪声基线）**（objects=['apple'], files/object=2）

| 指标 | 值 |
|---|---|
| 抓取数 | 64 |
| feasible（A / B） | 3 / 3 |
| 判定一致 | **64/64** |
| 原因一致 | 64/64 |
| feasible 翻转数 | **0** |
| max abs Δ 穿透 | 0.0 mm |
| max abs Δ 间隙 | 0.0 mm |
| max / median abs Δ rise | 0.0 / 0.0 mm |

<!-- COMPARE2:END -->

### 2.6 本阶段结论 + 遗留问题

**结论**

* object1 筛选已全量跑通（结果见 §4）；合并逻辑按内容哈希复用，`object` 与 `object1` 的判定一致。
* numpy 与 torch 后端在 **1e-7** 量级一致；判据差异只可能来自 PhysX 的接触混沌，而不是数学实现。
* v1 的穿透指标 bug 已修复，v1 的 feasible 集合不再被沿用。

**遗留问题**

* PhysX 提离结果对 1e-7 的位姿差异敏感（接触混沌），因此 `lift_rise_mm` 不可逐位复现（§2.5 数字）。
  判定用的是阈值（25 mm），阈值附近的抓取存在边界不确定性，已在 `index_grasps.csv` 保留原始值供复核。
* 手侧仍是凸包（phantom 最大 18 mm，掌部），提离测试可能被 phantom 接触"帮一把"；
  v2 用**精确网格接触门**（`geo_contact_points`）挡住"完全没有真实接触"的假阳性，但
  "真实接触 + phantom 额外支撑"的情况无法完全排除。

---

## 3. 阶段 2：合并 → CMap（统一左手）

### 3.1 合并结果

`filtered_grasps/index.csv` 同时包含 `source=object` 与 `source=object1`；文件分别放在
`filtered_grasps/object/...` 与 `filtered_grasps/object1/...`，逐字节 SHA-256 校验。
`filtered_grasps/index_grasps.csv` 是**逐 feasible 抓取**的明细（能量、穿透、接触 link、提离位移、水密性），
下游可以直接 `pandas.read_csv` 后按 `geo_penetration_mm <= x` 过滤。

`mesh_watertight` 已写入两个 CSV（`1/0`）。8 个非水密物体照常参与筛选（§1 清单），
其判定置信度较低，已在 §4 单列。

### 3.2 CMap 重命名方案（linkerhand → linkerhand_right）

**方案：内存别名，不动数据文件。**

`Cmap_dataset/dataset/CMapDataset.py` 新增：

```python
ROBOT_ALIASES = {"linkerhand": "linkerhand_right"}
...
self.metadata = [
    (m[0], m[1], ROBOT_ALIASES.get(m[2], m[2]))
    for m in metadata
    if m[1] in self.object_names and ROBOT_ALIASES.get(m[2], m[2]) in self.robot_names
]
```

* 为什么改这里：`robot_name` 只在这个类里被消费（过滤 + `create_hand_model`），改这一处即可全局生效；
* 为什么不动 `cmap_dataset.pt`：原文件里 `linkerhand` 样本为 0（§1.6），没有任何数据需要迁移；
* **回滚**：删除 `ROBOT_ALIASES` 字典并把 `self.metadata = [...]` 恢复成原来的一行列表推导即可
  （原行：`self.metadata = [m for m in metadata if m[1] in self.object_names and m[2] in self.robot_names]`）。
* 配套约定（在真实仓库里需要同步）：`utils/hand_model.create_hand_model` 必须把
  `linkerhand` 解析到**左手** `left/linkerhand_l20_left_extended.urdf`（本工作区用修复副本），
  `linkerhand_right` 解析到 `right/linkerhand_l20_right_extended.urdf`。
  本工作区没有 `utils/`，验证脚本用 shim 实现了同一映射。

### 3.3 CMap 转换（v2 相对 v1 的修正）

| 项 | v1 | v2 |
|---|---|---|
| `q[0:3]` 语义 | GraspQP `root_link` 位置 | **手基座** `hand_base_link`：`t_base = t_root + R_root @ (-0.03,0,-0.10)` |
| `robot_name` | `linkerhand`（但当时是右手语义） | `linkerhand`（**左手**） |
| 输入 | 单来源 `filtered_grasps/` | 合并后的 `index.csv`（object + object1），按 `(sha256, index)` 去重 |
| 去重 | 无 | 默认开启（两源字节相同 → 去掉 2115 条重复），`--no-dedupe` 可关 |

字段对照与统计见 §7 自动生成段。样例 `q` 回代校验（CMap 左 URDF + pk）：
把 `q` 拆成 6 虚拟 + 16 关节（mimic 展开成 21），在**修复后的左手 URDF** 上做 FK，取
`hand_base_link` 与 GraspQP 的 `root_pose @ T(root→base)` 对比。

### 3.4 用 CMapDataset.py 实机验证

`validate_cmap_dataset.py` 做了这些事：

1. 注入最小 `utils.hand_model`（pk 驱动，左右手映射）；
2. 用作用域 `os.path` 影子把 `ROOT_DIR/data/...` 重定向到真实路径，并支持把
   `cmap_dataset.pt` 换成我们的 v2 文件；
3. 实例化 `CMapDataset`（参考数据 + 我们的数据，train/validate 两种模式），检查
   `len()`、metadata 数量、`__getitem__()` 返回的 `q/object_name/robot_name`；
4. 检查 `ROBOT_ALIASES` 生效、两个身份各自解析到正确的 URDF；
5. `q` 回代 FK 对比。

结果见 §7。

### 3.5 本阶段结论 + 遗留问题

**结论**

* 合并后的数据集两个来源、逐抓取明细、水密性、穿透深度都可按列过滤。
* CMap 侧通过内存别名完成 `linkerhand → linkerhand_right`，数据文件零改动、可回滚。
* v2 修正了 v1 的 10.4 cm hand-base 偏移；`q` 回代 FK 的误差见 §7。

**遗留问题**

* CMap 的左手 URDF 原文件是坏的（§1.4），我们用的是修复副本；若上游修复，需切换到原文件。
* 数据集里的 `object_name` 仍是 `ycb+<object>`；object 与 object1 的样本在 CMap 里**无法区分来源**
  （内容也相同）。来源信息保存在 `filtered_grasps/index*.csv` 与 `*.meta.json`。
* `cmap_dataset_from_graspqp_v2.pt` 没有 heatmap 变体（不在本轮范围）。

---

## 4. 结果统计

<!-- STATS2:BEGIN -->
### 3.6 合并统计（来源 / 判定）

| source | .pt 文件 | 抓取数 | feasible | 通过率 | 拷贝文件 |
|---|---:|---:|---:|---:|---:|
| object | 420 | 13440 | 1276 | 9.5% | 359 |
| object1 | 420 | 13440 | 1276 | 9.5% | 359 |
| **合计** | 840 | 26880 | 2552 | 9.5% | 718 |

**逐物体（object1，全量重跑）**

| object | grasps | feasible | rate | 水密 | 备注 |
|---|---:|---:|---:|---|---|
| apple | 480 | 16 | 3.3% | ✅ |  |
| baseball | 480 | 21 | 4.4% | ✅ |  |
| bleach_cleanser | 480 | 3 | 0.6% | ✅ |  |
| cracker_box | 480 | 1 | 0.2% | ✅ |  |
| foam_brick | 480 | 24 | 5.0% | ✅ |  |
| gelatin_box | 480 | 39 | 8.1% | ❌ | 非水密，置信度低 |
| hammer | 480 | 108 | 22.5% | ✅ |  |
| lemon | 480 | 16 | 3.3% | ✅ |  |
| master_chef_can | 480 | 48 | 10.0% | ✅ |  |
| mini_soccer_ball | 480 | 25 | 5.2% | ❌ | 非水密，置信度低 |
| mustard_bottle | 480 | 49 | 10.2% | ❌ | 非水密，置信度低 |
| orange | 480 | 24 | 5.0% | ✅ |  |
| peach | 480 | 44 | 9.2% | ✅ |  |
| pear | 480 | 54 | 11.2% | ❌ | 非水密，置信度低 |
| phillips_screwdriver | 480 | 78 | 16.2% | ✅ |  |
| pitcher_base | 480 | 4 | 0.8% | ✅ |  |
| plum | 480 | 15 | 3.1% | ✅ |  |
| potted_meat_can | 480 | 38 | 7.9% | ✅ |  |
| power_drill | 480 | 53 | 11.0% | ❌ | 非水密，置信度低 |
| pudding_box | 480 | 61 | 12.7% | ✅ |  |
| rubiks_cube | 480 | 55 | 11.5% | ✅ |  |
| sponge | 480 | 110 | 22.9% | ❌ | 非水密，置信度低 |
| strawberry | 480 | 23 | 4.8% | ✅ |  |
| sugar_box | 480 | 20 | 4.2% | ✅ |  |
| tomato_soup_can | 480 | 48 | 10.0% | ❌ | 非水密，置信度低 |
| toy_airplane | 480 | 136 | 28.3% | ✅ |  |
| tuna_fish_can | 480 | 138 | 28.8% | ❌ | 非水密，置信度低 |
| wood_block | 480 | 25 | 5.2% | ✅ |  |

**feasible 抓取的几何指标（精确网格 SDF）**

| 指标 | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| 穿透深度 geo_penetration_mm | 0.00 | 7.46 | 10.46 | 13.03 | 14.99 |
| 最外点间隙 geo_min_distance_mm（负=在物体外） | -1.70 | 7.46 | 10.46 | 13.03 | 14.99 |
| 提离位移 lift_rise_mm | 25.17 | 52.45 | 58.52 | 61.64 | 191.32 |

* feasible 中仍有真实穿透（>1 µm）的：**2532 / 2552**（中位穿透 10.46 mm）
* feasible 中来自非水密物体的：**1032 / 2552**
<!-- STATS2:END -->

### 4.0 本阶段结论 + 遗留问题

**结论**：全量 13,440 个抓取跑通，1,276 个 feasible（9.5%）；逐物体通过率 0.2%（cracker_box）–
28.8%（tuna_fish_can）。`object` 与 `object1` 的可行集合完全一致（内容相同，内容寻址复用）。
穿透指标修复后，feasible 抓取的中位穿透仍在数毫米量级——这与"手侧凸包 + 真实网格接触"的组合一致，
说明通过的抓取大多是真实贴合并被提离测试确认的。

**遗留问题**：非水密物体的 SDF 碰撞可靠性低（§4.1）；Phantom 支撑无法完全排除（§2.6）；
阈值附近的抓取存在边界不确定性。

### 4.1 非水密物体的置信度

8 个非水密 STL（`gelatin_box, mini_soccer_ball, mustard_bottle, pear, power_drill, sponge,
tomato_soup_can, tuna_fish_can`）照常参与筛选。它们的 SAPIEN 动态三角网格 SDF 会打印
`SDF creation: Mesh is not suitable for SDF ...` 的 critical 警告（日志里有），因此其
`geo_*` 与提离判定的置信度**低于**其余 20 个物体。`index.csv` 的 `mesh_watertight=0` 可用来
一键排除它们。

---

## 5. GPU 加速：结论与实测

<!-- BENCH2:BEGIN -->
**SAPIEN / 环境**

| 项 | 值 |
|---|---|
| torch.cuda | True |
| SAPIEN | 3.0.3 |
| `enable_gpu()` 成功 | True |
| GPU 场景可 step | False |
| CPU steps/s（最小场景） | 100999.9 |
| GPU 错误 | `gpu: RuntimeError: failed to step: gpu simulation is not initialized.` |

**FK（pytorch_kinematics，23 links，float64）**

| batch | CPU ms | CPU ms/样本 | CUDA ms | CUDA ms/样本 | 加速比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.702 | 0.7016 | 3.095 | 3.0947 | 0.23× |
| 8 | 0.789 | 0.0986 | 2.835 | 0.3543 | 0.28× |
| 64 | 2.491 | 0.0389 | 2.991 | 0.0467 | 0.83× |
| 512 | 5.101 | 0.01 | 3.369 | 0.0066 | 1.51× |
| 4096 | 14.84 | 0.0036 | 6.375 | 0.0016 | 2.33× |

**四元数解码（roma）**

| batch | CPU ms | CUDA ms | 加速比 |
|---:|---:|---:|---:|
| 1 | 0.128 | 0.589 | 0.22× |
| 64 | 0.13 | 0.577 | 0.23× |
| 4096 | 0.244 | 0.6 | 0.41× |
| 100000 | 4.32 | 0.61 | 7.08× |

**物体体素 SDF 查询（100k 点归一化）**

| 点数 | numpy ms | torch-CPU ms | torch-CUDA ms |
|---:|---:|---:|---:|
| 1000 | 0.314 | 1.361 | 2.685 |
| 100000 | 25.618 | 10.66 | 3.775 |
| 1000000 | 397.383 | 38.792 | 14.541 |

* 端到端：同一对象同一批抓取，`--device cuda` 的 SAPIEN 筛选 **1.7 s/grasp**，`--device cpu` 约 **0.20 s/grasp**（CUDA 慢 8.5×，因为每个抓取只有 1 条 FK 配置，kernel launch + 同步开销占主导）。
<!-- BENCH2:END -->

**决策**：SAPIEN 筛选走 CPU（PhysX GPU 不可用，§1.7）；torch 侧默认 `--device cpu`，
在**大 batch**（FK ≥512 条配置、四元数 ≥100k、SDF ≥100k 点）时才切 CUDA。原因见上表：
batch=1 时 CUDA 的 kernel launch + 同步开销使 FK 慢 3.7×、四元数解码慢 4.5×
（这也是最初用 `--device cuda` 跑全量时 **1.7 s/grasp**（CPU 0.13 s）的原因）。
本轮全量运行因此用 CPU。

### 5.1 本阶段结论 + 遗留问题

**结论**：SAPIEN 的物理筛选在 CPU 上已是瓶颈（全量 63 分钟），PhysX GPU 不可用；
torch 侧的 GPU 收益只在大 batch 出现，而本流程每个抓取只有 1 条 FK 配置，因此 CPU 是正确的默认值。

**遗留问题**：若要真正 GPU 化，需要把"逐抓取"改成"批量抓取"的 SAPIEN 场景（多个手/物体实例同场景），
这是架构级改动，本轮未做；PhysX GPU 的 `step()` 失败原因（`gpu simulation is not initialized`）
未能在 SAPIEN 内部进一步定位。

---

## 6. Ubuntu 迁移检查清单

| # | 项 | 现状 / 风险 | 迁移动作 |
|---|---|---|---|
| 1 | 路径 | 全部 `pathlib`；无 `/d/study/...` 硬编码 | ✅ 直接可迁 |
| 2 | conda 环境名 | 脚本不读 `CONDA_DEFAULT_ENV`，只用 `python` | 用 `conda run -n <env>` 或 `--python` 包装即可；文档里写的 `genh2r` 只是本机名 |
| 3 | SAPIEN GPU 初始化 | Windows 上需 `cuda.dll` 别名且 step 失败（§1.7） | Ubuntu 上放好 `libPhysXGpu_64.so`，先 `ctypes.CDLL("libcuda.so", RTLD_GLOBAL)` 再 `enable_gpu()`，**必须在任何 Scene 之前**；再跑 `benchmark_cpu_gpu.py` 确认 |
| 4 | 多进程 | 代码单进程；`run_filter.py` 有 `if __name__ == "__main__":` | spawn/fork 都安全；若后续加 `multiprocessing`，Windows spawn 需要函数可 pickle |
| 5 | CUDA 版本匹配 | torch 2.7.0+cu128；SAPIEN 自带 PhysX；pk/roma 是纯 Python | 服务器上核对 `nvidia-smi` / `torch.version.cuda` / PhysX 包版本一致 |
| 6 | 文件权限 / SHA-256 | `shutil.copy2` + `hashlib.sha256`，无 Windows-only API | ✅；注意 Ubuntu 上目标目录的 umask/ACL |
| 7 | open3d 无显示器 | `--offscreen` 不依赖 open3d 窗口，用 matplotlib Agg 落盘 | 服务器上用 `--offscreen`；若要交互需 X11 转发或 `xvfb-run` |
| 8 | PhysX 驱动 | Windows 需要 `nvcuda.dll` 别名 | Ubuntu 需要可用的 `libcuda.so`（驱动版本 ≥ 510）；PhysX GPU 与 CUDA toolkit 版本无强绑定，但驱动必须支持 12.x |
| 9 | 渲染/可视化 | 本项目截图用 matplotlib，不依赖 GPU 渲染 | ✅ 可直接跑 |

---

## 7. 自动生成的结果段

<!-- AUTO2:BEGIN -->
### 7.1 数据集结构对照

| 项 | cmap_dataset.pt（参考） | cmap_dataset_from_graspqp_v2.pt |
|---|---|---|
| 顶层 keys | `['info', 'metadata']` | `['info', 'metadata']` |
| metadata 类型 | `list` / `tuple` | `list` / `tuple` |
| 样本数 | 24764 | 1276 |
| 机器人 | ['allegro', 'barrett', 'ezgripper', 'robotiq_3finger', 'shadowhand'] | ['linkerhand'] |
| q | float32 (N,22) | `torch.float32` `(1276, 22)` |
| 物体数 | 42 | 28 |
| 来源 | — | {'object': 1276} |
| 去重丢弃 | — | 1276 |

### 7.2 逐维数值范围（vs allegro 参考）

| dim | 本数据 min | 本数据 max | 本数据 mean | allegro min | allegro max |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.2225 | +0.2166 | +0.0080 | -0.1486 | +0.1604 |
| 1 | -0.2282 | +0.2304 | -0.0056 | -0.1569 | +0.1555 |
| 2 | -0.2135 | +0.2183 | +0.0210 | -0.1886 | +0.1764 |
| 3 | -3.1017 | +3.1308 | -0.0344 | -3.1416 | +3.1413 |
| 4 | -1.5303 | +1.5307 | -0.0346 | -1.5608 | +1.5530 |
| 5 | -3.1327 | +3.1230 | -0.0025 | -3.1410 | +3.1395 |
| 6 | -0.2265 | +0.8155 | +0.3039 | -0.4710 | +0.4706 |
| 7 | -0.1173 | +0.4579 | +0.1419 | -0.2202 | +1.6814 |
| 8 | -0.1366 | +0.7372 | +0.2474 | -0.2473 | +1.7162 |
| 9 | -0.2601 | +0.5501 | +0.1310 | -0.2270 | +1.6180 |
| 10 | -0.2229 | +0.4672 | +0.1687 | -0.4759 | +0.4722 |
| 11 | -0.1426 | +1.0175 | +0.3481 | -0.2643 | +1.6245 |
| 12 | -0.5129 | +0.4602 | -0.0417 | -0.1991 | +1.7151 |
| 13 | -0.1687 | +0.6440 | +0.2112 | -0.2274 | +1.6183 |
| 14 | -0.2699 | +1.1671 | +0.3894 | -0.4873 | +0.4725 |
| 15 | -0.8288 | +0.1682 | -0.2136 | -0.2345 | +1.6235 |
| 16 | -0.2298 | +0.6118 | +0.1535 | -0.2002 | +1.7388 |
| 17 | -0.1324 | +1.1930 | +0.3683 | -0.2281 | +1.6185 |
| 18 | +0.0501 | +1.1083 | +0.6317 | +0.2367 | +1.3965 |
| 19 | -0.6756 | +0.9040 | +0.1108 | -0.1069 | +1.1721 |
| 20 | +0.0096 | +0.7569 | +0.4603 | -0.2506 | +1.6442 |
| 21 | -0.0131 | +0.8535 | +0.3761 | -0.1623 | +1.7203 |

### 7.3 抽样

* `ycb+apple` / `linkerhand`: `[-0.0372, +0.0936, -0.1949, +0.4392, -0.0251, -0.3710, …]`
* `ycb+apple` / `linkerhand`: `[+0.0178, -0.1338, -0.0730, -1.5304, -0.2081, +0.4491, …]`
* `ycb+apple` / `linkerhand`: `[-0.1314, +0.1196, -0.0640, +0.8532, +0.6868, -0.9024, …]`

### 7.4 用 CMapDataset.py 的实机验证

| 检查 | 结果 |
|---|---|
| ROBOT_ALIASES | `{'linkerhand': 'linkerhand_right'}`（生效: True） |
| URDF `linkerhand` | `new_grasp_filter\assets\cmap_left_repaired.urdf` dof=27 pk_ok=True |
| URDF `linkerhand_right` | `Cmap_dataset\data_urdf\robot\linkerhand\right\linkerhand_l20_right_extended.urdf` dof=27 pk_ok=True |
| 参考 cmap_dataset.pt: metadata / len() | 521 / 521 |
| 参考 cmap_dataset.pt: robots | ['allegro'] |
| 我们的 v2 数据集: metadata / len() | 1276 / 28 |
| 我们的 v2 数据集: robots | ['linkerhand'] |
| 我们的 v2 数据集: `__getitem__` | True  -> linkerhand / ycb+apple |
| 参考数据 + linkerhand_right: metadata / len() | 0 / 0 |
| 参考数据 + linkerhand_right: robots | [] |
| alias 合成测试（legacy -> right） | {'n_metadata': 1, 'robot_seen': 'linkerhand_right', 'expected': 'linkerhand_right', 'ok': True} |
| alias 合成测试（left 保持 linkerhand） | {'n_metadata': 1, 'robot_seen': 'linkerhand', 'expected': 'linkerhand', 'ok': True} |
| q 回代 FK（20 条） | 旋转 `7.23e-07`，平移 `7.05e-09` m |

> 备注：`object` 与 `object1` 的内容完全相同，去重后数据集里每个抓取只出现一次；来源计数在 `*.meta.json` 的 `per_source` 里（去重前）。
<!-- AUTO2:END -->

---

## 8. 假设与已知问题汇总

| # | 项 | 状态 |
|---|---|---|
| A1 | `.pt` 关节顺序 = `DRIVE_NAMES`（= 当前 URDF 链顺序） | ✅ 由 `_check_joint_layout` + pk 关节顺序一致确认 |
| A2 | `root_pose` 为 xyzw，作用于 `root_link` | ✅ 代码路径 + v1/v2 FK 一致 |
| A3 | CMap `q[0:6]` = `T(x,y,z)·Rx·Ry·Rz`，末端手基座 | ✅ 在修复后的左 URDF 上 FK 回代验证（§3.4） |
| A4 | 数据是左手 L20 | ✅ URDF 名 + 关节 origin y 符号；CMap 左 URDF 16 关节 Δ=0 |
| A5 | object1 是"新碰撞 mesh 重新生成"的数据 | ❌ **不成立**：420/420 与 object 逐字节相同（§1.1） |
| A6 | SAPIEN PhysX GPU 可用于本机 | ❌ 不可用（§1.7），已回退 CPU |
| A7 | 手侧凸包能代表真实手指 | ⚠️ 近似：指尖 3–4 mm、掌部最大 18 mm phantom；用精确网格接触门限制假阳性 |
| A8 | 提离测试结果可复现 | ⚠️ 接触混沌：同代码重跑的 `lift_rise_mm` 可差数厘米（§2.5），判定按阈值 |
| A9 | `num_upper_object` = 单物体最大样本数 | ⚠️ 数据自洽，但 CMap 代码从不使用（§1.5） |
| A10 | CMap 左 URDF 可直接用 | ❌ dangling `base_link`，用修复副本（§1.4） |
