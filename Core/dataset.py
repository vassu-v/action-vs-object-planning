import random
import string
from .env import ROOT, create_world, Action, apply_action
from .perception import encode_world

def random_text(n=20):
    return ''.join(random.choice(string.ascii_lowercase + " ") for _ in range(n))

# Canonical filenames for stability and diversity
CANONICAL_FILES = [
    "docs/notes.txt",
    "docs/todo.txt",
    "docs/report_A.txt",
    "docs/report_B.txt",
    "docs/summary_day1.txt",
    "docs/summary_day2.txt",
    "logs/sum.log",
    "logs/system.log",
    "data/dataset_01.csv",
    "data/dataset_02.csv",
    "tmp/scratch_1.txt",
    "tmp/scratch_2.txt"
]

def random_text(n=20):
    return ''.join(random.choice(string.ascii_lowercase + " ") for _ in range(n))

def traj_append_random():
    file_path = random.choice([
        "docs/notes.txt",
        "docs/todo.txt"
    ])
    return [
        Action("WRITE", (file_path, "\n" + random_text()))
    ]


def traj_run_and_store():
    # Use stable report names to ensure stable indexing
    report_name = random.choice(["report_A.txt", "report_B.txt"])
    return [
        Action("RUN", ("scripts/sum.py",)),
        Action("CREATE", ("docs", report_name, "Result:\n")),
        Action("WRITE", (f"docs/{report_name}", "See logs/sum.log\n"))
    ]

# NOTE: These trajectories are heuristic and formulaic.
# They do NOT represent optimal planning or intelligent reasoning.
# Training data consists of valid but non-optimal trajectories, 
# sufficient to seed predictive models but not encode planning competence.

def traj_make_summary():
    notes = (ROOT / "docs/notes.txt").read_text(errors='replace')
    todo = (ROOT / "docs/todo.txt").read_text(errors='replace')

    # Use stable summary names
    summary_name = random.choice(["summary_day1.txt", "summary_day2.txt"])

    return [
        Action("CREATE", ("docs", summary_name, "SUMMARY\n")),
        Action("WRITE", (f"docs/{summary_name}", notes)),
        Action("WRITE", (f"docs/{summary_name}", "\n")),
        Action("WRITE", (f"docs/{summary_name}", todo)),
        Action("WRITE", (f"docs/{summary_name}", todo)),
    ]

def traj_fail_write():
    # Failing trajectory: Writing to non-existent dir
    return [
        Action("WRITE", ("nonexistent/test.txt", "content"))
    ]

def traj_move_random():
    # Diversity: Move files between folders
    target = random.choice(["docs/notes.txt", "docs/todo.txt"])
    dest = random.choice(["tmp", "data"])
    return [
        Action("MOVE", (target, dest))
    ]

def traj_write_data():
    # Diversity: Write to data folder
    target = random.choice(["data/dataset_01.csv", "data/dataset_02.csv"])
    return [
        Action("CREATE", ("data", target.split("/")[-1], "id,val\n1,0.5\n")),
        Action("WRITE", (target, "2,0.8\n"))
    ]

def traj_create_tmp():
    # Diversity: Create temp scratch files
    target = random.choice(["tmp/scratch_1.txt", "tmp/scratch_2.txt"])
    return [
        Action("CREATE", ("tmp", target.split("/")[-1], "scratchpad\n")),
        Action("REPLACE", (target, "wiped\n"))
    ]



def snapshot_world():
    """
    Capture the creature's full perception of the world.
    """
    objects, global_vec = encode_world(ROOT)
    return {
        "objects": objects,
        "global": global_vec
    }


def rollout_trajectory(action_sequence):
    data = []

    for action in action_sequence:
        before = snapshot_world()
        apply_action(action)
        after = snapshot_world()

        data.append({
            "state_before": before,
            "action": action,
            "state_after": after
        })

    return data


def generate_dataset(n_trajectories=100):
    """
    Generates a synthetic dataset of agent trajectories.
    
    NOTE: These trajectories are heuristic and formulaic.
    They do NOT represent optimal planning or intelligent reasoning.
    They serve only to seed the world model with valid state transitions.
    """
    dataset = []

    generators = [
        traj_append_random,
        traj_run_and_store,
        traj_run_and_store, # Upweight success cases
        traj_make_summary,
        traj_fail_write,
        traj_move_random,
        traj_write_data,
        traj_create_tmp
    ]

    for _ in range(n_trajectories):
        create_world()
        traj_fn = random.choice(generators)
        actions = traj_fn()
        dataset.extend(rollout_trajectory(actions))

    return dataset

if __name__ == "__main__":
    dataset = generate_dataset(200)
    len(dataset)

