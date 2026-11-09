"""Blender-side mesh repair (headless).

Steps: merge-by-distance -> fill holes -> recalc normals outside -> (if still not
closed) voxel remesh + smooth + decimate -> export STL.

All thresholds are relative to the bounding-box diagonal, never absolute.

Run:
    blender.exe --background --factory-startup --python repair_mesh.py -- \
        --input <in.stl> --output <out.stl> [--voxel-div 256] [--decimate-ratio 0.5] \
        [--json <report.json>]
"""

import argparse
import json
import math
import os
import sys

import bmesh
import bpy


def parse_args():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--voxel-div", type=int, default=256)
    ap.add_argument("--decimate-ratio", type=float, default=0.5)
    ap.add_argument("--max-faces", type=int, default=0, help="if >0, decimate until faces <= this")
    ap.add_argument("--json", default=None)
    ap.add_argument("--force-remesh", action="store_true")
    return ap.parse_args(args)


def clean_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)


def nonmanifold_edges(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    n = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    bm.free()
    return n


def main():
    a = parse_args()
    report = {"input": a.input, "output": a.output, "steps": [], "blender": bpy.app.version_string}
    clean_scene()

    bpy.ops.wm.stl_import(filepath=a.input)
    ob = bpy.context.selected_objects[0] if bpy.context.selected_objects else bpy.context.active_object
    me = ob.data
    report["faces_in"] = len(me.polygons)
    report["verts_in"] = len(me.vertices)
    report["nonmanifold_in"] = nonmanifold_edges(me)

    # bbox diagonal -> adaptive thresholds
    xs = [v.co.x for v in me.vertices]
    ys = [v.co.y for v in me.vertices]
    zs = [v.co.z for v in me.vertices]
    diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2)
    merge_thr = max(1e-7, diag * 1e-5)
    report["bbox_diag"] = diag
    report["merge_threshold"] = merge_thr

    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)

    # 1) merge by distance + 2) fill holes + 3) normals outside
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=merge_thr)
    bpy.ops.mesh.fill_holes(sides=0)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    report["steps"].append("merge_by_distance+fill_holes+normals")
    report["faces_after_cleanup"] = len(ob.data.polygons)
    report["nonmanifold_after_cleanup"] = nonmanifold_edges(ob.data)

    # 4) escalate to voxel remesh when still open
    if report["nonmanifold_after_cleanup"] > 0 or a.force_remesh:
        bpy.ops.object.mode_set(mode="OBJECT")
        voxel = max(1e-4, diag / a.voxel_div)
        ob.data.remesh_voxel_size = voxel
        bpy.ops.object.voxel_remesh()
        report["voxel_size"] = voxel
        report["steps"].append(f"voxel_remesh({voxel:.6f})")
        bpy.ops.object.modifier_add(type="SMOOTH")
        bpy.context.object.modifiers["Smooth"].factor = 0.5
        bpy.context.object.modifiers["Smooth"].iterations = 5
        bpy.ops.object.modifier_apply(modifier="Smooth")
        report["steps"].append("smooth(0.5,5)")

        faces = len(ob.data.polygons)
        target = a.max_faces if a.max_faces > 0 else max(2000, int(report["faces_in"] * 2))
        for _pass in range(3):
            if faces <= target:
                break
            ratio = max(0.02, min(0.95, target / faces))
            bpy.ops.object.modifier_add(type="DECIMATE")
            bpy.context.object.modifiers["Decimate"].decimate_type = "COLLAPSE"
            bpy.context.object.modifiers["Decimate"].ratio = ratio
            bpy.ops.object.modifier_apply(modifier="Decimate")
            report["steps"].append(f"decimate({ratio:.3f})")
            faces = len(ob.data.polygons)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")

    report["faces_out"] = len(ob.data.polygons)
    report["verts_out"] = len(ob.data.vertices)
    report["nonmanifold_out"] = nonmanifold_edges(ob.data)

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.wm.stl_export(filepath=a.output, export_selected_objects=True)
    report["out_bytes"] = os.path.getsize(a.output)
    print("[repair] " + json.dumps(report), flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
