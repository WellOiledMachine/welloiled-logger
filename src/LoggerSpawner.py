# project: AI Benchmarking
# Date: 2025-01-10
# Updated: 2025-01-20

import os
import shlex
import subprocess
import sys
from pathlib import Path


class LoggerSpawner:
    def __init__(self, output_dir, interval=None, pid=None):
        """
        Creates a LoggerSpawner object to start a logger process, which will
        monitor the system utilization of the python process that creates this object.

        In other words, use this class in your code to log the system utilization of
        that code.

        Parameters
        ----------
        output_dir : str
            The path to the directory where the logging results will be saved.
        interval : int, optional
            The interval (in seconds) between logged samples. Default is 1 second.
        """

        self.interval = max(1, interval)  # Ensure the interval is at least 1 second
        self.output_dir = str(Path(output_dir).expanduser().resolve())
        print(f"Logging results will be saved to {self.output_dir}")

        self.parent_pid = os.getpid()
        print(f"Parent PID: {self.parent_pid}")

    def start_logger(self):
        logger_script = os.path.join(os.path.dirname(__file__), "logger.py")
        command = f"{shlex.quote(sys.executable)} {shlex.quote(logger_script)} -p {self.parent_pid} -i {self.interval} -o {shlex.quote(self.output_dir)}"

        logger_script = subprocess.Popen(
            shlex.split(command),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"Logging started with PID: {logger_script.pid}")
