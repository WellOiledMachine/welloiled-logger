import subprocess
from collections import namedtuple

# Define a named tuple datatype to make the code more readable
ProcessInfo = namedtuple(
    "ProcessInfo",
    ["cpu_percent", "ram_used", "threads", "disk_io"],
)


def check_for_nvidia_gpu():
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except Exception:
        return False
