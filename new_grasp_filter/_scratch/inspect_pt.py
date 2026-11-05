"""Scratch: dump structure of a GraspQP .pt file and CMapDataset .pt."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def describe(obj, prefix="", depth=0, maxdepth=4):
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{prefix}dict[{len(obj)}] keys={list(obj.keys())[:40]}")
        if depth < maxdepth:
            for k, v in obj.items():
                describe(v, f"{k}: ", depth + 1, maxdepth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{prefix}{type(obj).__name__}[{len(obj)}]")
        if depth < maxdepth and len(obj):
            describe(obj[0], "[0]: ", depth + 1, maxdepth)
    elif isinstance(obj, torch.Tensor):
        finite = ""
        if obj.is_floating_point() and obj.numel():
            f = obj.float()
            finite = f" min={f.min().item():.5g} max={f.max().item():.5g} mean={f.mean().item():.5g}"
        print(f"{pad}{prefix}Tensor shape={tuple(obj.shape)} dtype={obj.dtype}{finite}")
    elif hasattr(obj, "__dataclass_fields__"):
        print(f"{pad}{prefix}{type(obj).__name__} dataclass fields={list(obj.__dataclass_fields__)}")
        if depth < maxdepth:
            for k in obj.__dataclass_fields__:
                describe(getattr(obj, k), f"{k}: ", depth + 1, maxdepth)
    else:
        s = repr(obj)
        print(f"{pad}{prefix}{type(obj).__name__} = {s[:200]}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "object/ycb/apple/grasp_predictions/linkerhand_l20/12_contacts/graspqp/default/apple.dexgrasp.pt"
    )
    print("=" * 100)
    print("FILE:", target, f"({target.stat().st_size} bytes)")
    d = torch.load(target, map_location="cpu", weights_only=False)
    describe(d)
