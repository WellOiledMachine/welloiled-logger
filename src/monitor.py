import argparse
import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil
import pynvml

from utils import ProcessInfo, check_for_nvidia_gpu

MONITOR_PID = os.getpid()  # The PID of the monitor process


def sync_process_children(process, children_map=None):
    """
    Update the dictionary of child processes of a given process.
    This function will take care not to redefine any of the existing process
    objects in the dictionary. Doing so would cause the process objects to
    reset their CPU and memory usage statistics every time they we wanted to
    update them.

    Parameters
    ----------
    process : psutil.Process
        The parent process whose children will be updated.
    children_map : dict, optional
        The dictionary of child processes to update.
        Omit this parameter to create a new dictionary, with new process objects.

    Returns
    -------
    children_map : dict
        The updated (or newly created) dictionary of child processes.
    """
    global MONITOR_PID

    if children_map is None:
        children_map = {}

    current_children = set(process.children(recursive=True))

    # Remove children that are no longer running
    for pid in list(children_map.keys()):
        # Make sure that the monitor process is never monitored
        # Also, remove the child if it is no longer a child of the parent process
        if pid == MONITOR_PID or not any(
            child.pid == pid for child in current_children
        ):
            del children_map[pid]

    # Add new children
    for child in current_children:
        if child.pid != MONITOR_PID and child.pid not in children_map:
            children_map[child.pid] = child

    return children_map


def get_ps_info(process):
    """
    Get the CPU, RAM, and Disk utilization of a process.

    Parameters
    ----------
    process : psutil.Process
        The process to query.

    Returns
    -------
    ProcessInfo: namedtuple
        A namedtuple containing the CPU, RAM, and Disk utilization of the process.
    """
    try:
        cpu_percent = process.cpu_percent(interval=None)
        ram_used = process.memory_info().rss  # in bytes
        threads = process.num_threads()

        disk_io = process.io_counters()
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        psutil.ZombieProcess,
        AttributeError,
    ):
        return None

    return ProcessInfo(cpu_percent, ram_used, threads, disk_io)


def get_ps_gpu_mem(pid):
    """
    Get the GPU memory usage of a process.

    Parameters
    ----------
    pid : int
        The process id to query.

    Returns
    -------
    """


def start_logging(args):
    """
    Start logging process and GPU information to CSV files.

    This function creates two log files:
    1. Process Information: Records CPU, RAM, and Disk utilization.
    2. GPU Information: Logs GPU metrics IFF an NVIDIA GPU is available and
       automatically detected.

    Process Metrics:
    - The logged information captures the total resource usage of the main process
      and all its subprocesses (child processes) combined.
    - CPU usage: Sum of CPU utilization across all related processes.
    - RAM usage: Total memory consumed by the main process and all its subprocesses.
    - Disk I/O: Cumulative disk read/write operations from all processes.
    - Threads: Total number of threads across all processes.
    - (TODO: Not yet implemented) GPU Memory: Memory usage of the GPU by the process.

    GPU Metrics (if available):
    - GPU utilization
    - Memory usage
    - Temperature
    - Power consumption

    Note on Disk I/O:
    Since we are interested in the total disk I/O of the main process and all its
    children, we need to ensure that any children who exit early still have their
    disk I/O stats accounted for. Therefore, we disk I/O stats are cached
    and summed up at the end of each monitoring interval.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.
    """
    # Initialize common variables
    total_ram_avail = psutil.virtual_memory().total  # in bytes
    has_nvidia_gpu = check_for_nvidia_gpu()  # Check if NVIDIA GPU is available
    monitor_timestamp = datetime.now().strftime(
        "%Y_%m_%d_%H-%M-%S"
    )  # Timestamp for the monitoring session

    # Create output directory
    args.output_dir = args.output_dir / f"{monitor_timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define output files. One for process info and one for GPU info
    psinfo_file = args.output_dir / "psinfo.csv"
    gpuinfo_file = args.output_dir / "gpuinfo.csv"

    # Open output files and monitor the process
    with open(psinfo_file, mode="w", newline="") as psinfo:
        with open(gpuinfo_file, mode="w", newline="") as gpuinfo:
            ps_writer = csv.writer(psinfo)
            ps_writer.writerow(
                [
                    "Timestamp (Unix)",
                    "CPU Utilization (%)",
                    "Thread Count",
                    "RAM Utilization (Bytes)",
                    "RAM Utilization (%)",
                    "Disk Read (Bytes)",
                    "Disk Write (Bytes)",
                    "Disk Read Operations (Count)",
                    "Disk Write Operations (Count)",
                ]
            )

            gpu_writer = csv.writer(gpuinfo)
            if has_nvidia_gpu:
                gpu_writer.writerow(
                    [
                        "Timestamp (Unix)",
                        "GPU ID",
                        "Utilization (%)",
                        "Memory (Bytes)",
                        "Memory (%)",
                        "Temperature (°C)",
                        "Power Usage (W)",
                    ]
                )

                # Initialize NVML to monitor GPU stats
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
            else:
                gpu_writer.writerow(
                    ["# `nvidia-smi` not found. No GPU monitoring available."]
                )

            # If a command is provided, run the command and monitor the process
            if args.command:
                # Convert list to single string. Required for using shell=True in Popen
                command = " ".join(args.command)

                # Execute command and redirect stdout and stderr to log files
                stdout = args.output_dir / "stdout.log"
                stderr = args.output_dir / "stderr.log"
                with open(stdout, "w") as out, open(stderr, "w") as err:
                    out.write(f"Running command: {command}\n")
                    print(f"Running command: {command}")
                    proc = subprocess.Popen(
                        command,
                        shell=True,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=err,
                    )
                parent_proc = psutil.Process(proc.pid)
                args.pid = proc.pid
            else:
                parent_proc = psutil.Process(args.pid)

            child_proc_map = sync_process_children(parent_proc)

            # Initialize disk I/O cache to keep track of TOTAL disk I/O stats
            disk_io_cache = {}
            while (
                parent_proc.is_running()
                and parent_proc.status() != psutil.STATUS_ZOMBIE
            ):
                start_timestamp = time.time()

                # Get Process Info from parent
                parent_info = get_ps_info(parent_proc)
                if parent_info is None:
                    break

                # Extract info from dictionary to make code more readable
                cpu_percent = parent_info.cpu_percent
                total_ram_used = parent_info.ram_used
                threads = parent_info.threads
                disk_io_cache[parent_proc.pid] = parent_info.disk_io

                # Aggregate CPU and RAM usage for child processes
                for child in child_proc_map:
                    child_info = get_ps_info(child_proc_map[child])
                    if child_info is not None:
                        cpu_percent += child_info.cpu_percent
                        total_ram_used += child_info.ram_used
                        threads += child_info.threads
                        disk_io_cache[child] = child_info.disk_io

                total_ram_used_percent = (total_ram_used / total_ram_avail) * 100

                if has_nvidia_gpu:
                    for i in range(gpu_count):
                        gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(gpu)
                        temperature = pynvml.nvmlDeviceGetTemperature(gpu, 0)
                        power = pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000
                        mem_percent = memory.used / memory.total * 100
                        gpu_writer.writerow(
                            [
                                start_timestamp,
                                i,
                                utilization.gpu,
                                memory.used,
                                mem_percent,
                                temperature,
                                power,
                            ]
                        )

                # Sum up all disk stats from the disk_io_cache
                disk_read_bytes = sum(io.read_bytes for io in disk_io_cache.values())
                disk_write_bytes = sum(io.write_bytes for io in disk_io_cache.values())
                disk_read_ops = sum(io.read_count for io in disk_io_cache.values())
                disk_write_ops = sum(io.write_count for io in disk_io_cache.values())
                ps_writer.writerow(
                    [
                        start_timestamp,
                        cpu_percent,
                        threads,
                        total_ram_used,
                        total_ram_used_percent,
                        disk_read_bytes,
                        disk_write_bytes,
                        disk_read_ops,
                        disk_write_ops,
                    ]
                )

                # Update the child process map
                child_proc_map = sync_process_children(parent_proc, child_proc_map)

                # Sleep for the remaining time in the interval
                end_timestamp = time.time()
                elapsed_time = end_timestamp - start_timestamp
                sleep_time = max(0, args.interval - elapsed_time)
                time.sleep(sleep_time)

    if has_nvidia_gpu:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor the system utilization of a process."
    )
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-p",
        "--pid",
        type=int,
        help="The PID of the process to monitor.",
    )
    group.add_argument(
        "-c",
        "--command",
        nargs=argparse.REMAINDER,
        help="The command to run the process to monitor.",
    )

    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=1,
        help="The interval (in seconds) between monitoring samples.",
    )

    default_output_dir = Path(__file__).parent.parent / "monitoring_results"
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=default_output_dir,
        help="The directory to save the monitoring results. \
        Default: ../monitoring_results",
    )
    args = parser.parse_args()
    # Expand user and resolve any symlinks in the path
    args.output_dir = args.output_dir.expanduser().resolve()

    # Start logging the process information
    start_logging(args)
