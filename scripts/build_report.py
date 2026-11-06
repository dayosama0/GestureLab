"""Build the current Assignment 2 completed-work report."""
from pathlib import Path
import runpy
import subprocess

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_word_report.py")), run_name="__main__")
    subprocess.run([
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(Path(__file__).with_name("export_word_report.ps1")),
    ], check=True)
