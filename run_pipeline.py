import subprocess
import argparse
import os
import datetime
import sys

def main():
    parser = argparse.ArgumentParser(description="Automated Training and Evaluation Pipeline")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--runs", type=int, default=10, help="Evaluation runs (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Create exp directory if it doesn't exist
    exp_dir = "exp"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Generate timestamp and filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp-e{args.epochs}-r{args.runs}_{timestamp}.txt"
    filepath = os.path.join(exp_dir, filename)

    print(f"Starting pipeline...")
    print(f"Results will be saved to: {filepath}")

    with open(filepath, "w") as f:
        # Write Parameters
        f.write("="*50 + "\n")
        f.write("EXPERIMENT PARAMETERS\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Runs: {args.runs}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write("="*50 + "\n\n")
        f.flush()

        # 1. Run Training
        print(f"Running Training ({args.epochs} epochs)...")
        train_cmd = [sys.executable, "run_train.py", "--epochs", str(args.epochs), "--seed", str(args.seed)]
        subprocess.run(train_cmd, check=True)
        f.write("--- TRAINING COMPLETE ---\n")
        f.write("\n" + "-"*30 + "\n\n")
        f.flush()

        # 2. Run Evaluations
        variants = ["A", "B"]
        tasks = ["log", "summary"]

        for variant in variants:
            for task_name in tasks:
                print(f"Running Evaluation: Variant {variant}, Task {task_name}...")
                f.write(f"--- EVALUATION OUTPUT: Variant {variant}, Task {task_name} ---\n")
                eval_cmd = [
                    sys.executable, "run_eval.py", 
                    "--variant", variant, 
                    "--task", task_name, 
                    "--runs", str(args.runs), 
                    "--seed", str(args.seed)
                ]
                process = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    sys.stdout.write(line)
                    f.write(line)
                process.wait()
                f.write("\n" + "-"*30 + "\n\n")
                f.flush()

    print(f"\nPipeline complete. Results saved to {filepath}")

if __name__ == "__main__":
    main()
