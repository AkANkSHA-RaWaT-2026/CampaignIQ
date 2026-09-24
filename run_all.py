"""
run_all.py - run the whole CampaignIQ pipeline in order with one command.
Usage:  python run_all.py
"""
import subprocess                                   # run other Python scripts
import sys                                          # sys.executable = the Python of your active venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["src/generate_data.py", "src/01_clean_data.py", "src/02_features.py", "src/03_eda.py",
         "src/04_model.py", "src/05_predict.py", "src/06_report.py"]

for step in STEPS:
    print(f"\n{'=' * 8} {step} {'=' * 8}")
    # sys.executable guarantees we use the SAME Python (and packages) as this script
    result = subprocess.run([sys.executable, str(ROOT / step)], cwd=ROOT)
    if result.returncode != 0:                      # stop at the first failure so errors don't cascade
        sys.exit(f"\nStopped: {step} failed. Fix it, then re-run.")
print("\nAll done. Start the dashboard with:  streamlit run dashboard/app.py")