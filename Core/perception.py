import hashlib
from pathlib import Path
import numpy as np
from .env import ROOT

BASE_DIM = 64
OBJECT_DIM = 128

def hash_to_vec(text: str, dim=BASE_DIM):
    h = hashlib.sha256(text.encode()).digest()
    nums = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    vec = np.tile(nums, dim // len(nums) + 1)[:dim]
    return vec / 255.0

def project(vec, out_dim=OBJECT_DIM):
    if len(vec) > out_dim:
        return vec[:out_dim]
    elif len(vec) < out_dim:
        return np.concatenate([vec, np.zeros(out_dim - len(vec))])
    return vec

def encode_file(path: Path):
    content = path.read_text()
    meta = f"FILE|{path.suffix}|{path.parent.name}|{len(content)}"
    return project(np.concatenate([
        hash_to_vec(content),
        hash_to_vec(meta)
    ]))

def encode_dir(path: Path):
    meta = f"DIR|{path.name}|{len(list(path.iterdir()))}"
    return project(hash_to_vec(meta))

def encode_world(root: Path):
    objects = {}
    parts = []

    for item in sorted(root.rglob("*")):
        rel = item.relative_to(root).as_posix()
        if item.is_dir():
            vec = encode_dir(item)
            key = f"dir:{rel}"
        else:
            vec = encode_file(item)
            key = f"file:{rel}"

        objects[key] = vec
        parts.append(vec)

    global_vec = np.mean(np.stack(parts), axis=0)

    return objects, global_vec

if __name__ == "__main__":
    objects, global_vec = encode_world(ROOT)
    print(len(objects), global_vec.shape)
