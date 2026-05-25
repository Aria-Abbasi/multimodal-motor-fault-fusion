"""Deep-inspect one Paderborn MAT file."""

from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io as sio


def describe(obj, name="root", depth=0, max_depth=6):
    indent = "  " * depth
    typ = type(obj).__name__
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)

    print(f"{indent}{name}: type={typ}, shape={shape}, dtype={dtype}")

    if depth >= max_depth:
        return

    if hasattr(obj, "_fieldnames"):
        for field in obj._fieldnames:
            try:
                value = getattr(obj, field)
                describe(value, field, depth + 1, max_depth)
            except Exception as e:
                print(f"{indent}  {field}: ERROR {type(e).__name__}: {e}")

    elif hasattr(obj, "flat") and getattr(obj, "dtype", None) == object:
        for i, item in enumerate(obj.flat):
            if i >= 3:
                print(f"{indent}  ...")
                break
            describe(item, f"{name}[{i}]", depth + 1, max_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path)
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    print(f"File: {path}")
    print("Top-level keys:")
    for key in mat:
        if not key.startswith("__"):
            value = mat[key]
            print(" ", key, type(value).__name__, getattr(value, "shape", None))

    print("\nDeep structure:")
    for key, value in mat.items():
        if not key.startswith("__"):
            describe(value, key)


if __name__ == "__main__":
    main()
