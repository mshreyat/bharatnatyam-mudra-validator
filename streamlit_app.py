import os
import subprocess
import sys

# 1. Force the Linux kernel to install Mesa drivers dynamically on boot
try:
    subprocess.run(["apt-get", "update"], check=False)
    subprocess.run(["apt-get", "install", "-y", "libgl1"], check=False)
except Exception:
    pass

# 2. Hand off control cleanly to your actual application file
sys.path.insert(0, os.path.dirname(__file__))
import app

if __name__ == "__main__":
    # This acts as an automated proxy wrapper
    pass