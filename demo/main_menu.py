###
## cluster_maker: Master Menu
##
## A unified interface to run all exam demos.
###

import sys
import os
import subprocess

def print_header():
    print("\n" + "=" * 60)
    print(" MA52109: PRACTICAL EXAM SUBMISSION")
    print("=" * 60)
    print(" Available Demos:")
    print("   1. Task 2: Debugged Cluster Plot (Points in 2D)")
    print("   2. Task 4: Simulated Data Analysis (K-Means)")
    print("   3. Task 5: Agglomerative Clustering (Difficult Data)")
    print("   q. Quit")
    print("-" * 60)

def run_script(script_path, args=None):
    """Runs a python script in a subprocess."""
    # Ensure we use the same python executable (python3)
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    try:
        print(f"\n>>> Launching {script_path}...\n")
        subprocess.run(cmd, check=True)
        print(f"\n>>> {script_path} finished.")
    except subprocess.CalledProcessError:
        print(f"\n!!! Error occurred while running {script_path}")
    except KeyboardInterrupt:
        print("\n>>> Execution interrupted by user.")

def main():
    # Get the absolute path to the demo folder to avoid path errors
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        print_header()
        choice = input(" Select an option (1-3, q): ").strip().lower()
        
        if choice == '1':
            # Task 2 requires an input CSV argument
            script = os.path.join(base_dir, "cluster_plot.py")
            data_file = os.path.join(base_dir, "..", "data", "demo_data.csv")
            run_script(script, [data_file])
            
        elif choice == '2':
            # Task 4 is standalone
            script = os.path.join(base_dir, "simulated_clustering.py")
            run_script(script)
            
        elif choice == '3':
            # Task 5 is standalone
            script = os.path.join(base_dir, "demo_agglomerative.py")
            run_script(script)
            
        elif choice == 'q':
            print("\nExiting. Goodbye!")
            break
        else:
            print("\nInvalid selection. Please try again.")
            
        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    main()