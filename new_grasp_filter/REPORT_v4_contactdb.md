# v4：ContactDB 可行抓取 → CMap 格式，并与 YCB 数据集拼接

> **范围**：本轮只做 `HANDOFF_ubuntu.md` §8.1 的 P0。
> **(A)** 跑完 §11 的交接检查清单；**(B)** 扩展 `convert_to_cmap.py`（多输入 + 按来源前缀），
> 把 ContactDB 的可行抓取转成 CMap 格式，与已有 YCB 数据集拼成 **一个新文件**；
> **(C)** 实机校验（结构 / 前缀 / 计数 / FK 回代 / `CMapDataset.py` 加载）。
>
> **没有**重跑任何筛选（ycb 3429 条沿用 v2），**没有**改物体网格 / 碰撞体 / 阈值，**没有**写 `object1/`。
> §8.2 的暂缓项（D1–D6）一项未做。
>
> 环境：Ubuntu 24.04.3，`conda activate graspqp`（版本见 §2.1）。

---

## 0. 可直接复制运行的命令序列

```bash
conda activate graspqp
cd ~/codes/graspqp

# 1) 交接检查清单（HANDOFF §11；见 §1 的结果表）
python new_grasp_filter/validate_grasp_io.py                 # numpy vs roma/pk
python new_grasp_filter/validate_cmap_dataset.py \
    --dataset new_grasp_filter/cmap_dataset_from_graspqp_v2.pt \
    --index  new_grasp_filter/filtered_grasps/index_ycb.csv --prefixes ycb \
    --out    new_grasp_filter/logs/validate_cmap_dataset_v2_ubuntu.json

# 2) 【P0】ContactDB + YCB → 一个 CMap 数据集
python new_grasp_filter/convert_to_cmap.py \
    --inputs  new_grasp_filter/filtered_grasps/index_ycb.csv \
              new_grasp_filter/filtered_grasps/index.csv \
    --prefixes ycb contactdb \
    --output  new_grasp_filter/cmap_dataset_from_graspqp_v3.pt

# 3) 结构 / 前缀 / 计数 / 与 v2 逐样本一致 / FK 回代（20 条 contactdb + 3 条 ycb）
python new_grasp_filter/verify_cmap_v3.py

# 4) 用 CMapDataset.py 实机加载新文件
python new_grasp_filter/validate_cmap_dataset.py \
    --dataset new_grasp_filter/cmap_dataset_from_graspqp_v3.pt \
    --index  new_grasp_filter/filtered_grasps/index_ycb.csv \
             new_grasp_filter/filtered_grasps/index.csv \
    --prefixes ycb contactdb \
    --out    new_grasp_filter/logs/validate_cmap_dataset_v3.json
```

---

## 1. 交接检查清单（HANDOFF §11）结果

| # | 检查 | 期望 | 实测 | 结论 |
|---|---|---|---|---|
| 1 | 依赖版本 | 记录 | 见 §2.1 | ✅ |
| 2a | `index_ycb.csv` | 840 行（object 420 + object1 420）、副本 0 处不符 | 840 行、`Counter(object=420, object1=420)`、**786 个副本、hash 不符 0** | ✅ |
| 2b | `index.csv`（contactdb） | 只含 `object_contactdb`、0 处不符 | 600 行、`Counter(object_contactdb=600)`、**573 个副本、hash 不符 0** | ✅ |
| 2c | `sum(n_feasible)` | — | ycb **3429**（= `index_ycb_grasps.csv` 行数）；contactdb **2911**（= `index_grasps.csv` 行数） | ✅ |
| 2d | 9 个无抓取物体 | 在 `object/contactdb_nograsp/`、不在 `object/contactdb/` | 9 个目录齐全（cube/cylinder/pyramid × s/m/l）；`object/contactdb/` 下 **40** 个物体，无这 9 个 | ✅ |
| 3 | 手模型可加载 | 23 link / 21 主动关节 | `hand links 23 joints 21`；`meshes/collisions` **22** 个（含 v3 的 `thumb_metacarpals.STL`） | ✅ |
| 4 | SAPIEN PhysX GPU（可选） | 能用则用，否则 CPU | `sapien.physx.enable_gpu()` **挂起**（>100 s 被 kill，无输出）；`libcuda.so` 预加载无效 → **保持 CPU** | ⚠️ 环境限制（不影响本轮） |
| 5 | 数据集 + `CMapDataset.py` 验证（v2 基线） | `n_metadata=3429`、`len=28`、`getitem_ok` | 完全一致；`q_roundtrip` 平移 **7.40e-9 m**（与 Windows 的记录逐位相同），旋转 2.02e-2（见 §2.2 pk 偏置） | ✅（旋转偏差有明确原因） |

**额外发现（清单里未列，但影响后续步骤）**：仓库根的 `Cmap_dataset/` 在本机**不存在**
（`.gitignore` 忽略且未随迁移带过来），§11-5 与 §8.1-7/8 的实机校验因此一开始无法运行。
处理方式见 §2.3。

---

## 2. Ubuntu 环境与 Windows 的差异（本轮新发现，**下游必读**）

### 2.1 版本快照（记录进报告，便于复现）

| 项 | 值 |
|---|---|
| OS / Python | Ubuntu 24.04.3 LTS x86_64 / 3.11.18（conda `graspqp`） |
| torch | 2.8.0+cu129，`cuda.is_available()=True`（本轮未用） |
| sapien | 3.0.3（CPU） |
| trimesh / numpy / scipy / roma | 4.8.0 / 1.26.4 / 1.15.3 / 1.5.4 |
| **pytorch_kinematics** | **0.3.0，来自 fork `renezurbruegg/pytorch_kinematics` @ `92c3b97`**（见 §2.2） |
| Blender | `/home/chenyc/Blender/blender-5.1.2-linux-x64/blender`（5.1.2）— **注意不是 `/usr/bin/blender`** |
| CPU / RAM | 32 核 / 125 GB |

### 2.2 【重要】本机 `pytorch_kinematics` 给每个 revolute 关节加 ±0.01 rad

`site-packages/pytorch_kinematics/frame.py::Frame.get_transform` 里有一行刻意加上的偏移：

```python
if self.joint.joint_type == 'revolute':
    # ugly fix
    theta = theta + 2 * ((theta > 0).float() - 0.5) * 1e-2      # = ±0.01 rad
```

上游 `UM-ARM-Lab/pytorch_kinematics` **没有**这行；本机的安装来源是
`renezurbruegg` 的 fork（`direct_url.json` 记录 commit `92c3b97`）。实测后果：

| 检查 | Windows（历史） | 本机 |
|---|---|---|
| `validate_grasp_io.py`：`max|T_numpy − T_pk|` | ~1e-7（REPORT_v2 §1.8/§2.5） | **3.97e-2**（最大在 `thumb_distal`，指尖 3.0–4.0 cm） |
| CMap `q` 回代旋转误差（transl. 不变） | 2.45e-7（REPORT_v3 §2.1） | **2.02e-2**（平移仍是 7.40e-9） |

复现最小例子：单关节 `roll=0.5 rad` 经该 pk 得到的夹角是 **0.51 rad**（逐角度均匀 +0.01）。

**影响面（本轮只记录、不修，按用户指示）**：

* **数据本身不受影响**：`q` 的 16 个关节角是 `.pt` 文件里的原值，`q` 的位姿来自 `roma` + numpy 欧拉分解，
  与 pk 无关；用 `grasp_io` 的 **numpy** FK 回代，误差回到 **2.0e-7 / 1.5e-8**（§4.3）。
* **判据可能受影响**：`run_filter.py` 默认 `--legacy-numpy=False` → `gio.set_backend("torch")`，
  `sapien_grasp_checker` 的手部 link 位姿来自 pk FK。本次 ContactDB 筛选（本机 2026-11-09）用的是默认后端，
  因此它的接触/提离判定是在**带 ±0.01 rad/关节偏置**的手型上算的（ycb 那一批在 Windows 上算，当时的 pk 无此偏置）。
* **要不要重跑**：不在本轮范围（D1）。若下游对两批判据的**严格一致性**有要求，建议在
  `--legacy-numpy`（numpy FK）下重跑 ContactDB（600 个文件、19,200 个抓取，按 §10 的 0.2 s/抓取约 1 小时），
  或把 pk 换回上游版本后重跑。**本轮未做。**

### 2.3 `Cmap_dataset/` 缺失 → 从 `~/codes/TRO-Grasp/data` 重建（42 MB）

`.gitignore` 第 5 行忽略 `Cmap_dataset/`，迁移时未带过来，但它被 `grasp_io.CMAP_ROOT` 引用
（CMap 左手 URDF、参考数据集、`CMapDataset.py`、`PointCloud/`）。本机 `~/codes/TRO-Grasp/` 有完整的上游布局，
因此按"最小可复现集合"重建（**只在新仓库根写入，已在 `.gitignore` 内，不进版本库**）：

| 复制/链接 | 来源 | 体积 |
|---|---|---|
| `CMapDataset_filtered/{cmap_dataset.pt, split_train_validate_objects.json}` | `TRO-Grasp/data/CMapDataset_filtered/` | 4.0 MB |
| `PointCloud/{object,robot}` | `TRO-Grasp/data/PointCloud/` | 2.4 MB |
| `data_urdf/robot/{linkerhand, allegro}` + `urdf_assets_meta.json` | `TRO-Grasp/data/data_urdf/robot/` | 36 MB |
| `data_urdf/object` → **符号链接** 到 `TRO-Grasp/data/data_urdf/object`（107 MB，只读采样用） | — | 0 |
| `dataset/{CMapDataset.py, train.json, val.json}` | `TRO-Grasp/dataset/` | 20 KB |

**对 `CMapDataset.py` 的唯一改动**（等价于 REPORT_v2 §3.2 的 v2 改动，但按 HANDOFF §7-5 的**条件化**规则实现，
否则我们的左手数据会被误改名——见 §5.2）：

```python
ROBOT_ALIASES = {"linkerhand": "linkerhand_right"}

def _apply_robot_aliases(metadata, declared_robots, aliases=None):
    aliases = ROBOT_ALIASES if aliases is None else aliases
    return [(m[0], m[1], m[2] if m[2] in declared_robots else aliases.get(m[2], m[2])) for m in metadata]
...
payload = torch.load(dataset_path)
declared_robots = set((payload.get('info') or {}).keys())     # 我们声明了 linkerhand → 不改名
metadata = _apply_robot_aliases(payload['metadata'], declared_robots)
self.metadata = [m for m in metadata if m[1] in self.object_names and m[2] in self.robot_names]
```

**另外重生成了一处派生文件**：`new_grasp_filter/assets/cmap_left_repaired.urdf` 原来写死了 Windows 的
`D:/…/Cmap_dataset/…/meshes/*.STL`，本机不存在 → 删掉后由 `grasp_io.cmap_left_urdf()` 自动重生成
（内容差异**只有** mesh 的绝对路径前缀；`gripper` 结构、`virtual_robot` → `hand_base_link` 的修复不变）。
重生成后 22 个 link 的 visual mesh 全部存在（0 缺失）。

### 2.4 其它

* **Blender 路径**：`repair_stls.py:41` 仍是 Windows 的 `D:\Blender_Foundation\…\blender.exe`。
  本机实际是 `/home/chenyc/Blender/blender-5.1.2-linux-x64/blender`。本轮不需要修网格，**未改**；
  将来要用时改这一处常量即可（HANDOFF §3）。
* **`logs/` 只带了 contactdb 的部分**：本机 `logs/` 下只有 `object_contactdb/`（41 个 JSONL）与本轮新生成的 JSON，
  ycb 的 `logs/object*/*.jsonl`（Windows 侧的可追溯证据）**不在本机**。逐抓取明细仍可由
  `filtered_grasps/index_ycb_grasps.csv`（3429 行）完整复原，故不影响本轮结论。
* **PhysX GPU**：`enable_gpu()` 在本机**挂起**（不是报错），连同 `libcuda.so` 预加载也一样。保持 CPU。

---

## 3. 代码改动（全部向后兼容）

| 文件 | 改动 |
|---|---|
| `new_grasp_filter/convert_to_cmap.py` | 新增 `--inputs <csv…>` + `--prefixes <p…>`；`object_name = f"{prefix}+{obj}"`（ycb→`ycb+apple`，contactdb→`contactdb+alarm_clock`）。不传 `--inputs` 时行为与 v2 完全相同（`--input` 目录 + 前缀 `ycb`）。去重键 `(sha256, grasp_index)` 跨输入生效；`meta.json` 增写 `per_prefix`、`prefixes`、`inputs`、`n_missing_grasp_files`。新增可复用的 `iter_pairs()`（数据集顺序的唯一来源，校验脚本共用）。 |
| `new_grasp_filter/validate_cmap_dataset.py` | 新增 `--index <csv…>` / `--prefixes <p…>`（修 HANDOFF §8.1 已知陷阱：默认只读 `filtered_grasps/index.csv`，合并后会把 contactdb 的 q 和 ycb 的文件配错）；`check_q_roundtrip` 改用 `convert_to_cmap.iter_pairs()` 复现顺序；**`load_cmap_module()` 改为每次加载独立模块对象**（原来 `importlib.reload` 会共用同一个 module，导致后一次 `dataset_override` 悄悄改掉"参考数据集"检查的目标，见 §5.2）。 |
| `new_grasp_filter/verify_cmap_v3.py` | **新增**。结构 / 前缀 / 计数 / info 聚合 / 9 个无抓取物体缺席 / 与 v2 逐样本一致 / FK 回代（numpy 与 pk 两个后端）一次跑完，写 `logs/verify_cmap_v3.json`，失败非零退出。 |
| `Cmap_dataset/dataset/CMapDataset.py` | 条件化 `ROBOT_ALIASES`（§2.3 代码块）。**CMap 原始数据文件零改动**。 |

`q` 的生成**没有变**：`grasp_io.root_pose_to_cmap_q(root_pose, joints)`（手基座语义 `t_base = t_root + R_root @ (-0.03,0,-0.10)`，
欧拉 `R = Rx(roll) @ Ry(pitch) @ Rz(yaw)`）——没有回退到 v1 的 `root_link` 直转。

---

## 4. 结果

### 4.1 输入（`new_grasp_filter/filtered_grasps/`）

| 索引 | 行数 | source 分布 | 抓取数 | feasible | 有 feasible 的文件 / 物体 | 副本（hash 全等） |
|---|---|---|---|---|---|---|
| `index_ycb.csv` | 840 | object 420 + object1 420 | 26,880 | **3,429** | 786 / 28 | 786 ✅ |
| `index.csv`（contactdb） | 600 | object_contactdb 600 | 19,200 | **2,911** | 573 / 40（27 个文件 0 可行） | 573 ✅ |

`index_ycb_v1_object.csv` / `index_v1_object.csv`（v1 旧数据）**未参与**本轮合并。

ContactDB 的筛选理由分布（`index.csv` 的 `reason_counts` 汇总）：`ok 2911`、`insufficient_contacts 7745`、
`penetration 4278`、`slips_on_lift 4265`、`too_few_contact_links 1`。
feasible 抓取的穿透中位数 6.96 mm（p95 14.18、max 14.99 mm，闸门 15 mm），提离 25.1–193.4 mm（中位 59.0 mm）。

### 4.2 输出：`cmap_dataset_from_graspqp_v3.pt`

| 项 | 值 |
|---|---|
| 样本数 | **6,340** = 3,429（ycb）+ 2,911（contactdb）✅ |
| 物体数 | **68** = 28（ycb）+ 40（contactdb 有可行抓取者）✅ |
| 前缀 | `ycb` 3,429 / `contactdb` 2,911 ✅ |
| 来源 | object 2,115 + object1 1,314 + object_contactdb 2,911 |
| `robot_name` | `linkerhand`（左手，`info` 中已声明） |
| `q` | `float32 (22,)`：`[x,y,z, roll,pitch,yaw, 16 joints]` |
| `info` | `num_total=6340`、`num_upper_object=285`、`num_per_object` 68 项 |
| 去重 | 键 `(sha256, grasp_index)`，跨来源丢弃 **0** 条（两批数据字节不同） |
| 缺失副本 | 0 |
| 体积 | 2.4 MB（v2 是 1.25 MB） |

9 个面太少的 contactdb 物体（`cube_*`/`cylinder_*`/`pyramid_*`）在数据集中**不出现**——预期行为，不是丢数据
（网格在 `object/contactdb_nograsp/`）。逐物体计数见 `cmap_dataset_from_graspqp_v3.meta.json`；
contactdb 里样本最少的 5 个物体：`sphere_large 2`、`apple 11`、`sphere_medium 16`、`torus_small 17`、`piggy_bank 18`，
最多：`mug 179`、`wine_glass 170`、`cup 158`、`ps_controller 146`、`utah_teapot 130`。

> **可复现性**：连跑两次 `convert_to_cmap.py`，输出**内容逐位相同**（`max|Δq| = 0.0`、样本顺序相同、
> `info` 相同），但 `.pt` 的**字节**不同（md5 `f136a319…` vs `f3ecafad…`）—— `torch.save` 的 zip 容器
> 本身不保证字节级可复现。**校验请比较张量内容（`torch.equal`），不要比较 md5。**

**字段定义沿用 REPORT_v2 §4.1 / §7.1**（`info` / `metadata` 三元组 `(q, object_name, robot_name)`），本轮的差别只有
`object_name` 现在有两种前缀。

### 4.3 校验（全部实际运行，输出在 `logs/`）

`python new_grasp_filter/verify_cmap_v3.py` → `logs/verify_cmap_v3.json`（`all_ok: true`，`failures: []`）：

| 检查 | 结果 |
|---|---|
| `torch.load` + 顶层 keys | ✅ `['info','metadata']` |
| `metadata` | ✅ `list[tuple(3)]`，`q` 全为 `torch.float32 (22,)` |
| `robot_name` | ✅ `info` 与全部样本都是 `linkerhand` |
| `N == 3429 + 2911` | ✅ 6340 |
| 物体数 `== 28 + 40` | ✅ 68 |
| `info` 聚合自洽 | ✅ `num_total` / `num_upper_object=285` / `num_per_object` 与 metadata 完全一致 |
| 前缀 | ✅ ycb 全部 `ycb+*`、contactdb 全部 `contactdb+*`，无第三种前缀、无缺 `+` |
| 9 个无抓取物体 | ✅ 一个都没出现 |
| 逐物体计数 == 索引 `n_feasible` 之和 | ✅ 无差异（0 处） |
| **YCB 部分与 v2 逐样本一致** | ✅ 前 3429 条 `torch.equal` 全等，`max|Δq| = 0.0`，`object_name` 全部相同 |
| `q` 回代 FK（**20 条 contactdb**，numpy FK） | ✅ 旋转 **2.0e-7**、平移 **1.5e-8 m** |
| `q` 回代 FK（3 条 ycb 抽样，numpy FK） | ✅ 旋转 1.5e-7、平移 1.4e-8 m |
| 同上但用 pk（作对照） | ⚠️ 旋转 2.0e-2、平移 6.9e-9 —— §2.2 的 pk 偏置，非数据问题 |

`python new_grasp_filter/validate_cmap_dataset.py --dataset …v3.pt --index … --prefixes …`
→ `logs/validate_cmap_dataset_v3.json`（用 `CMapDataset.py` 实机加载，参考数据也一并检查）：

| 项 | 结果 |
|---|---|
| 参考数据（allegro） | ✅ `metadata=521`、`len=521`、3 个 debug 物体（与 REPORT_v3 §2.1 一致，未受我们的改动影响） |
| **我们的 v3 数据** | ✅ `metadata=6340`、`len=68`、`robots=['linkerhand']`、`q=(22,) float32` |
| `__getitem__(0)` | ✅ `getitem_ok`，`robot=linkerhand`、`object=contactdb+airplane`、`initial_q=(1,27)`、`object_pc=(1,512,3)` |
| PointCloud 覆盖 | ✅ 68 个物体在 `Cmap_dataset/PointCloud/object/` 里都有 `.pt`（无需 shim） |
| `ROBOT_ALIASES` | ✅ 字典存在；我们的数据在 `info` 里声明了 `linkerhand` → **不被改名**；evidence：`legacy` 合成 payload → `linkerhand_right` ✅、`left` 合成 payload → `linkerhand` ✅ |
| 左右手 URDF 解析 | ✅ `linkerhand` → `assets/cmap_left_repaired.urdf`（27 dof，pk 可加载）；`linkerhand_right` → `Cmap_dataset/…/right/…extended.urdf`（27 dof） |
| `q` 回代（同一脚本，pk 后端） | ⚠️ 旋转 1.96e-2 / 平移 6.4e-9 —— 见 §2.2；numpy 后端见上表 |

**v2 基线回归**（`logs/validate_cmap_dataset_v2_ubuntu.json`）：`n_metadata=3429`、`len=28`、
`getitem` 返回 `ycb+apple`，平移误差 7.40e-9 —— 与 Windows 侧记录逐位相同，说明宿主机迁移没有改变数值行为。

---

## 5. 已知问题与遗留

| # | 项 | 状态 / 建议 |
|---|---|---|
| E1 | 本机 `pytorch_kinematics` fork 给每个 revolute 关节 +±0.01 rad（§2.2） | ⚠️ **未修**。它不影响导出的 `q`（用 numpy FK 回代 2e-7），但 ContactDB 那批 SAPIEN 判定是用偏置后的手型算的；若要两批判据严格一致，建议 `--legacy-numpy` 重跑 ContactDB 或换回上游 pk（不在本轮范围） |
| E2 | `Cmap_dataset/` 未随迁移带过来 | ✅ 已从 `TRO-Grasp/data` 重建最小集合（§2.3，42 MB，`data_urdf/object` 为符号链接，指向 `~/codes/TRO-Grasp`，移动该目录会导致参考数据校验失效） |
| E3 | `CMapDataset.py` 的别名规则必须**条件化** | ✅ 已按 HANDOFF §7-5 实现（无条件版本会把我们的左手 6340 条全部过滤成 0 条 —— 本轮第一次跑就踩到了） |
| E4 | `validate_cmap_dataset.py` 的模块复用污染 | ✅ 已修为每次加载独立模块对象；否则 `reference_linkerhand_right` 会误报 91 条 |
| E5 | `assets/cmap_left_repaired.urdf` 含 Windows 绝对路径 | ✅ 已重生成（只换 mesh 绝对路径前缀） |
| E6 | `repair_stls.py:41` 的 `BLENDER` 仍是 Windows 路径 | ⚠️ 未改（本轮不动网格）；将来用 Blender 时改成 `/home/chenyc/Blender/blender-5.1.2-linux-x64/blender` |
| E7 | SAPIEN PhysX GPU 在本机 `enable_gpu()` 挂起 | ⚠️ 保持 CPU |
| E8 | ContactDB 的 `utah_teapot` 网格仍非水密（v3 修复失败，原文件保留） | ⚠️ 它贡献 130 条 feasible，按 v2 §4.1 的"非水密置信度"口径属低置信样本；数据集中正常保留（`mesh_watertight=0` 可在 `index.csv` 里过滤） |
| E9 | ycb 8 个物体非水密（`object1/` 的原始网格） | ℹ️ 与 v2 相同（240 个文件、`mesh_watertight=0`），来源是 `object1/` 未同步 v3 修复（D2） |
| E10 | 本机 `logs/` 只有 contactdb 的 JSONL | ℹ️ ycb 的逐抓取证据不在本机；结论仍可由 `index_ycb_grasps.csv` 复原 |
| E11 | `q` 的 16 关节是 `.pt` 原值；mimic 展开（5 个）在 `CMapDataset` 侧由 URDF 处理 | ℹ️ 与 v2 相同，未变 |

---

## 6. 产物清单

| 路径 | 说明 |
|---|---|
| `new_grasp_filter/cmap_dataset_from_graspqp_v3.pt` | **本轮主产物**：6340 条 / 68 物体 / ycb+contactdb / `linkerhand` |
| `new_grasp_filter/cmap_dataset_from_graspqp_v3.meta.json` | `per_object`、`per_source`、`per_prefix`、`prefixes`、`inputs`、`dropped_duplicates` |
| `new_grasp_filter/logs/verify_cmap_v3.json` | 结构/前缀/计数/FK 回代全部检查结果（含 20 条逐样本误差） |
| `new_grasp_filter/logs/validate_cmap_dataset_v3.json` | `CMapDataset.py` 实机加载 + 别名 + URDF 解析 + 回代 |
| `new_grasp_filter/logs/validate_cmap_dataset_v2_ubuntu.json` | v2 基线在 Ubuntu 上的回归 |
| `new_grasp_filter/logs/validate_grasp_io.json` | numpy vs roma/pk（含 §2.2 的 pk 偏置证据） |
| `new_grasp_filter/logs/report_v4_summary.json` | 本报告的机器可读摘要（输入/输出计数 + 校验结果 + 环境） |
| `new_grasp_filter/verify_cmap_v3.py` | 新增校验脚本 |
| `Cmap_dataset/`（`.gitignore` 第 5 行） | 从 `TRO-Grasp/data` 重建的参考数据（42 MB + `data_urdf/object` 符号链接） |

**未改动**：`cmap_dataset_from_graspqp_v2.pt`（基线，仍在）、`filtered_grasps/*.csv`、`object/`、`object1/`、
`graspqp/assets/linkerhand_l20/`、CMap 原始数据文件、所有阈值与 checker 代码。
