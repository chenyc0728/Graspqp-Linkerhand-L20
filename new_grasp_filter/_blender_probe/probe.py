"""Blender gate probe (v3.1): workspace read/write/delete in headless mode.

Differences from the first attempt: no ``wm.read_factory_settings`` (it can
restart the interpreter/deadlock under ``--background`` on 4.5); instead the
default objects are simply deleted.

Run:
    "<blender.exe>" --background --python probe.py -- <workspace_probe_dir>
"""

import os
import sys

import bpy

argv = sys.argv
args = argv[argv.index("--") + 1:] if "--" in argv else []
if not args:
    raise SystemExit("usage: -- <probe_dir>")
probe_dir = os.path.abspath(args[0])
os.makedirs(probe_dir, exist_ok=True)
print(f"[probe] blender={bpy.app.version_string} probe_dir={probe_dir}", flush=True)

# start from an empty scene without read_factory_settings
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
print(f"[probe] cube created, objects={len(bpy.data.objects)}", flush=True)

stl_path = os.path.join(probe_dir, "gate_cube.stl")
exported = False
for op in ("wm.stl_export", "export_mesh.stl"):
    try:
        mod, name = op.split(".")
        getattr(getattr(bpy.ops, mod), name)(filepath=stl_path, export_selected_objects=False)
        exported = True
        print(f"[probe] exported via bpy.ops.{op}", flush=True)
        break
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] {op} failed: {type(exc).__name__}: {exc}", flush=True)
if not exported:
    raise SystemExit("[probe] FAIL: could not export STL")

size = os.path.getsize(stl_path)
with open(stl_path, "rb") as f:
    head = f.read(16)
print(f"[probe] stl size={size} head={head[:8]!r}", flush=True)
if size < 12:
    raise SystemExit("[probe] FAIL: exported STL is empty")

reimport_ok = False
for op in ("wm.stl_import", "import_mesh.stl"):
    try:
        mod, name = op.split(".")
        getattr(getattr(bpy.ops, mod), name)(filepath=stl_path)
        reimport_ok = True
        print(f"[probe] re-imported via bpy.ops.{op}; objects={len(bpy.data.objects)}", flush=True)
        break
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] {op} failed: {type(exc).__name__}: {exc}", flush=True)

txt_path = os.path.join(probe_dir, "gate_write_test.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"blender {bpy.app.version_string} wrote this\n")
print(f"[probe] txt size={os.path.getsize(txt_path)}", flush=True)

os.remove(txt_path)
os.remove(stl_path)
print(f"[probe] deleted temp files; remaining={os.listdir(probe_dir)}", flush=True)
print("[probe] OK", flush=True)
