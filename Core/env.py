from pathlib import Path
import shutil
from dataclasses import dataclass
import subprocess
import sys

ROOT = Path("creature_world")


DIRS_STANDARD = ["docs", "scripts", "data", "logs", "tmp"]

FILES_STANDARD = {
    "docs/notes.txt": "meeting at 3pm\n",
    "docs/todo.txt": "- buy milk\n- finish project\n",
    "scripts/sum.py": "nums = [1, 2, 3, 4]\nprint(sum(nums))\n",
    "data/numbers.txt": "1,2,3,4\n"
}


# Dense Environment Configuration
DIRS_DENSE = list(set(DIRS_STANDARD + [
    "src", "include", "bin", "config", "tests", "web", "assets"
]))

FILES_DENSE = FILES_STANDARD.copy()
FILES_DENSE.update({
    # --- Dense Logic Files ---
    # src: Mixed Python and JS
    "src/main.py": "import utils\nprint('main')\n",
    "src/utils.py": "def helper(): pass\n",
    "src/script.js": "console.log('hello');\n",
    "src/index.html": "<html><body>Hello</body></html>\n",
    
    # include: C-like headers
    "include/header.h": "#ifndef HEADER_H\n#define HEADER_H\n#endif\n",
    "include/defs.h": "#define MAX 100\n",
    
    # bin: Scripts/Executables
    "bin/app.exe": "BINARY_CONTENT\n",
    "bin/script.sh": "#!/bin/bash\necho hello\n",
    
    # data: CSV, JSON, Binary
    "data/dataset.csv": "id,value\n1,100\n2,200\n",
    "data/params.json": "{\"timeout\": 100}\n",
    "data/image.png": "BINARY_IMAGE_DATA\n",
    
    # logs: Log files
    "logs/server.log": "Server started at 10:00\n",
    "logs/error.log": "Error on line 55\n",
    "logs/access.log": "127.0.0.1 - - [10/Jan/2026:10:00:00]\n",
    
    # docs: Markdown and Text
    "docs/readme.md": "# Project Title\nDescription here.\n",
    "docs/setup.txt": "1. Install\n2. Run\n",
    
    # config: Config files
    "config/settings.yaml": "version: 1.0\n",
    "config/env.prod": "DEBUG=False\n",
    
    # tests: Python tests
    "tests/test_main.py": "assert True\n",
    "tests/test_utils.py": "assert 1+1==2\n",
    
    # web: HTML/CSS
    "web/about.html": "<html>About Us</html>\n",
    "web/contact.html": "<html>Contact</html>\n",
    "web/style.css": "body { color: red; }\n",
    
    # assets: Misc
    "assets/logo.png": "PNG_DATA\n",
    "assets/icon.ico": "ICON_DATA\n"
})

@dataclass(frozen=True)
class Action:
    type: str
    args: tuple

def create_world(config="standard"):
    """
    Ensures a clean, deterministic world state.
    config: "standard" or "dense"
    WARNING: Destructive.
    """
    import time
    if ROOT.exists():
        try:
            shutil.rmtree(ROOT)
            time.sleep(0.1) # Give OS time to catch up
        except OSError:
            # Fallback for Windows locking issues
            time.sleep(0.5)
            shutil.rmtree(ROOT, ignore_errors=True)
            time.sleep(0.1)

    ROOT.mkdir(parents=True, exist_ok=True)
    
    # Select configuration
    if config == "dense":
        dirs_to_create = DIRS_DENSE
        files_to_create = FILES_DENSE
    else:
        dirs_to_create = DIRS_STANDARD
        files_to_create = FILES_STANDARD
        
        # EXPLICIT CLEANUP: Ensure dense directories are removed if they exist
        for d in ["src", "bin", "config", "tests", "web", "assets", "include"]:
            p = ROOT / d
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    for d in dirs_to_create:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    for path, content in files_to_create.items():
        p = ROOT / path
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    print(f"Creature world created (clean slate, config={config}).")

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
    if dest_dir.is_file():
        # Pathology: Cannot move a file into another file as a directory
        return False
    if not dest_dir.exists():
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
    try:
        shutil.move(str(src), str(dest_dir / src.name))
        return True
    except Exception:
        return False # Destination might exist or other OS-level mismatch

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
