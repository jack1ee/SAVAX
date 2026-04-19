#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write all features in a directory into <feature_dir>/features.lmdb.

key = plain video id (without extension), value = float32 bytes
Also save a shape table to recover each array as (T, D).
"""
import os, glob, json, argparse, lmdb, tqdm, numpy as np, struct, torch

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="",
                    help="Directory should contain only *.npy / *.pth.tar files.")
    ap.add_argument("--pattern",     default="*.npy",
                    help="Glob pattern, defaults to *.npy.")
    ap.add_argument("--dry_run",     action="store_true")
    return ap.parse_args()

def main():
    args  = parse_args()
    files = glob.glob(os.path.join(args.feature_dir, args.pattern))
    assert files, f"No file matched in {args.feature_dir}"

    # Estimate LMDB map_size in bytes from the total input size.
    total = sum(os.path.getsize(f) for f in files)
    map_size = int(total * 1.2)

    lmdb_path   = os.path.join(args.feature_dir, "features.lmdb")
    shape_path  = os.path.join(args.feature_dir, "shapes.json")

    if args.dry_run:
        print(f"[Dry-run] Expected to write {len(files)} files, total {total/1e9:.2f} GB, map_size={map_size/1e9:.2f} GB")
        return

    env = lmdb.open(lmdb_path,
                    map_size   = map_size,
                    subdir     = False,      # Use a single LMDB file.
                    meminit    = False,
                    map_async  = True)       # Sync explicitly on close.

    shapes = {}
    with env.begin(write=True) as txn, tqdm.tqdm(files) as bar:
        for f in bar:
            vid = os.path.splitext(os.path.basename(f))[0]   # e.g. v_1234
            if f.endswith(".npy"):
                arr = np.load(f, mmap_mode="r")              # Keep the load zero-copy.
            elif f.endswith(".pth.tar"):
                arr = torch.load(f).numpy()
            else:
                raise NotImplementedError(f"Unknown ext: {f}")

            assert arr.dtype == np.float32, "Please convert features to float32 offline first."
            txn.put(vid.encode(), arr.tobytes(order="C"))    # Store the raw array bytes.
            shapes[vid] = arr.shape                          # Save the original (T, D) shape.

    env.sync(); env.close()
    json.dump(shapes, open(shape_path, "w"))
    print(f"Done! LMDB={lmdb_path}  shapes.json saved with {len(shapes)} entries")

if __name__ == "__main__":
    main()
