from pathlib import Path
import shutil
from dataclasses import dataclass
import subprocess
import sys

ROOT = Path("creature_world")

DIRS = ["docs", "scripts", "data", "logs", "tmp"]

FILES = {
    "docs/notes.txt": "meeting at 3pm\n",
    "docs/todo.txt": "- buy milk\n- finish project\n",
    "scripts/sum.py": "nums = [1, 2, 3, 4]\nprint(sum(nums))\n",
    "data/numbers.txt": "1,2,3,4\n"
}

@dataclass(frozen=True)
class Action:
    type: str
    args: tuple

def create_world():
    """
    Ensures a clean, deterministic world state.
    WARNING: Destructive.
    """
    if ROOT.exists():
        try:
            shutil.rmtree(ROOT)
        except OSError:
            # Fallback for Windows locking issues
            import time
            time.sleep(0.5)
            shutil.rmtree(ROOT, ignore_errors=True)

    ROOT.mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        (ROOT / d).mkdir(exist_ok=True)

    for path, content in FILES.items():
        p = ROOT / path
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    print("Creature world created (clean slate).")

# ---- File System Helpers ----

def read_file(path: Path):
    if not path.exists():
        return None  # Changed from empty string to None to indicate failure
    return path.read_text(errors='replace')

def write_file(path: Path, content: str):
    # Append
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a") as f:
            f.write(content)
        return True
    except OSError:
        return False

def replace_file(path: Path, content: str):
    # Overwrite
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content)
        return True
    except OSError:
        return False

def create_file(root_dir: Path, filename: str, content: str):
    # args: (dir_name, filename, content)
    # The 'root_dir' arg passed here is actually the directory path
    target = root_dir / filename
    if target.exists():
         # Strictness: Don't overwrite existing files with CREATE
         return False
         
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(content)
        return True
    except OSError:
        return False

def move_file(src: Path, dest_dir: Path):
    if not src.exists():
        return False
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dest_dir / src.name))
        return True
    except shutil.Error:
        return False # Destination might exist

def run_file(path: Path):
    if not path.exists():
        return False
    
    # We only run python scripts safely for now
    if path.suffix == ".py":
        try:
            result = subprocess.run(
                [sys.executable, str(path)], 
                capture_output=True, 
                text=True, 
                cwd=path.parent.parent 
            )
            # Log output to logs/{script_name}.log
            log_dir = ROOT / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / (path.stem + ".log")
            
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n"
            log_file.write_text(output)
            return True
        except Exception as e:
            print(f"Failed to run script: {e}")
            return False
    return False

def apply_action(action: Action):
    """
    Executes an action on the file system.
    Returns success boolean.
    """
    t, a = action.type, action.args

    if t == "READ":
        if len(a) > 0:
            return read_file(ROOT / a[0]) is not None

    elif t == "WRITE":
        if len(a) > 1:
            return write_file(ROOT / a[0], a[1])

    elif t == "REPLACE":
        if len(a) > 1:
            return replace_file(ROOT / a[0], a[1])

    elif t == "CREATE":
        if len(a) > 2:
            return create_file(ROOT / a[0], a[1], a[2])

    elif t == "MOVE":
        if len(a) > 1:
            return move_file(ROOT / a[0], ROOT / a[1])

    elif t == "RUN":
        if len(a) > 0:
            return run_file(ROOT / a[0])
            
    return False


if __name__ == "__main__":
    create_world()
