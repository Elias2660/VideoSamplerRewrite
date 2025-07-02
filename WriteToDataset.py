#! /usr/bin/python3
"""
WriteToDataset.py

Writes valid samples into a WebDataset .tar, but preserves any truncated
or corrupted samples on disk for later debugging.
"""

import os
import io
import time
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor

import webdataset as wds
from PIL import Image

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)

def process_sample(key, png_root, txt_root, frames_per_sample, out_channels):
    """
    Reads one sample "<key>/" + "<key>.txt" and returns a WebDataset sample dict,
    or None if the folder doesn't contain exactly frames_per_sample PNGs or any
    of the PNGs are corrupt/truncated.
    """
    frame_dir = os.path.join(png_root, key)
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))

    # Check count
    if len(files) != frames_per_sample:
        logging.warning(
            f"{key}: expected {frames_per_sample} frames but found {len(files)}; leaving on disk"
        )
        return None

    # Verify each image is readable
    for fname in files:
        path = os.path.join(frame_dir, fname)
        try:
            with open(path, "rb") as f:
                data = f.read()
            # verify PNG integrity
            img = Image.open(io.BytesIO(data))
            img.verify()
        except Exception as e:
            logging.error(f"Truncated/corrupt image detected: {path} ({e}); dropping sample {key}")
            return None

    # build the sample dict
    sample = {}
    safe_key = key.replace(".", "_")
    sample["__key__"] = safe_key
    parts = key.split("_")
    if len(parts) < 2:
        logging.error(f"Malformed sample key: {key}")
        return None
    cls = parts[1]
    sample["cls"] = cls.encode("utf-8")

    # read metadata
    txt_path = os.path.join(txt_root, f"{key}.txt")
    try:
        with open(txt_path, "rb") as f:
            sample["metadata.txt"] = f.read()
    except Exception as e:
        logging.error(f"Could not read metadata for {key}: {e}")
        return None

    # read frames into memory
    for i, fname in enumerate(files):
        path = os.path.join(frame_dir, fname)
        try:
            with open(path, "rb") as img:
                sample[f"{i}.png"] = img.read()
        except Exception as e:
            logging.error(f"Could not read frame {path}: {e}")
            return None

    return sample


def write_to_dataset(
    png_root: str,
    tar_file: str,
    dataset_path: str,
    frames_per_sample: int = 1,
    out_channels: int = 3,
    batch_size: int = 60,
    equalize_samples: bool = False,
    max_workers: int = 4,
):
    """
    Walks each subfolder under png_root, writes *only* fully complete, uncorrupted samples
    into tar_file. Successful samples are deleted from disk; truncated/corrupt ones stay.
    """
    start = time.time()
    logging.info(f"Writing {tar_file} from samples in {png_root}")
    tar = wds.TarWriter(tar_file, encoder=False)
    txt_root = png_root.rstrip(os.sep) + "txt"
    keys = [d for d in os.listdir(png_root)
            if os.path.isdir(os.path.join(png_root, d))]
    logging.info(f"Found {len(keys)} sample folders")

    # optional equalization
    if equalize_samples:
        bycls = {}
        for k in keys:
            cls = k.split("_")[1]
            bycls.setdefault(cls, []).append(k)
        minc = min(len(v) for v in bycls.values())
        logging.info(f"Equalizing to {minc} samples per class")
        drop = []
        for v in bycls.values():
            v.sort()
            drop.extend(v[minc:])
        for d in drop:
            try:
                shutil.rmtree(os.path.join(png_root, d))
                os.remove(os.path.join(txt_root, f"{d}.txt"))
            except:
                pass
        keys = [k for k in keys if k not in drop]
        logging.info(f"Dropped {len(drop)} (equalize), {len(keys)} remain")

    count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            for key, sample in zip(
                batch,
                ex.map(
                    process_sample,
                    batch,
                    [png_root] * len(batch),
                    [txt_root] * len(batch),
                    [frames_per_sample] * len(batch),
                    [out_channels] * len(batch),
                ),
            ):
                if sample is None:
                    continue  # truncated/corrupt or error—left on disk

                # write valid sample to TAR
                tar.write(sample)
                count += 1

                # delete only successful samples
                try:
                    shutil.rmtree(os.path.join(png_root, key))
                    os.remove(os.path.join(txt_root, f"{key}.txt"))
                except Exception as e:
                    logging.warning(f"Cleanup failed for {key}: {e}")

                if count % 1000 == 0:
                    logging.info(f"  wrote {count} samples…")

    tar.close()
    logging.info(f"Finished writing {count}/{len(keys)} samples in {time.time()-start:.1f}s")

    with open(os.path.join(dataset_path, "RUN_DESCRIPTION.log"), "a+") as rd:
        rd.write(f"{count} samples → {tar_file}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("png_root")
    p.add_argument("tar_file")
    p.add_argument("dataset_path")
    p.add_argument("--frames_per_sample", type=int, default=1)
    p.add_argument("--out_channels", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=60)
    p.add_argument("--equalize", action="store_true")
    p.add_argument("--max_workers", type=int, default=4)
    args = p.parse_args()
    write_to_dataset(
        args.png_root,
        args.tar_file,
        args.dataset_path,
        frames_per_sample=args.frames_per_sample,
        out_channels=args.out_channels,
        batch_size=args.batch_size,
        equalize_samples=args.equalize,
        max_workers=args.max_workers,
    )
