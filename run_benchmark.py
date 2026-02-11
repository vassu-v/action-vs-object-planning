import subprocess
import sys
import argparse

TASKS = ["log", "summary", "refactor", "deploy", "config", "test", "archive"]
VARIANTS = ["A", "B"]

def run_eval(variant, task, env_cfg="mixed", runs=10):
    cmd = [
        sys.executable, "run_eval.py", 
        "--variant", variant, 
        "--task", task, 
        "--runs", str(runs), 
        "--env", env_cfg
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error running {variant}/{task}: {res.stderr}")
            return "ERR"
        # Parse output for "Success Rate: X%"
        for line in res.stdout.splitlines():
            if "Success Rate:" in line:
                return line.split(":")[1].strip()
        return "N/A"
    except Exception as e:
        return f"ERR({e})"

def main():
    parser = argparse.ArgumentParser(description="Experiment C Benchmarking")
    parser.add_argument("--runs", type=int, default=10, help="Runs per task")
    parser.add_argument("--env", choices=["standard", "dense", "mixed"], default="mixed")
    args = parser.parse_args()

    with open("results.txt", "w") as f:
        # Header / Settings
        header = f"EXPERIMENT C BENCHMARK\nSettings: Runs={args.runs}, Env={args.env}\n"
        header += "-" * 40 + "\n"
        print(header, end="")
        f.write(header)

        table_head = f"{'Task':<15} | {'Variant A':<10} | {'Variant B':<10}\n"
        table_separator = "-" * 40 + "\n"
        print(table_head + table_separator, end="")
        f.write(table_head + table_separator)

        for t_key in TASKS:
            res_a = run_eval("A", t_key, env_cfg=args.env, runs=args.runs)
            res_b = run_eval("B", t_key, env_cfg=args.env, runs=args.runs)
            
            row = f"{t_key:<15} | {res_a:<10} | {res_b:<10}\n"
            print(row, end="")
            f.write(row)

if __name__ == "__main__":
    main()
