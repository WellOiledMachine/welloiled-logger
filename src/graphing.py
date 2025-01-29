import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_disk_throughput(ps_df, save_path, graph_title):
    """
    Plot Disk Throughput

    This function reads disk throughput data from a pandas DataFrame and plots it as a
    scatter plot against time.

    Parameters
    ----------
    ps_df : pd.Dataframe
        The DataFrame containing the disk throughput data.
    save_dir : str
        The directory where the generated plot image will be saved.
    """
    plt.plot(
        ps_df["datetime"],
        ps_df["read_throughput"],
        marker="o",
        markersize=5,
        color="blue",
        linestyle="-",
        label="Read Throughput",
    )
    plt.plot(
        ps_df["datetime"],
        ps_df["write_throughput"],
        marker="x",
        markersize=5,
        color="orange",
        linestyle="-",
        label="Write Throughput",
    )
    plt.xlabel("Time", fontsize=14)
    plt.xticks(rotation=315, ha="left")
    plt.ylabel("Disk Throughput (Bytes/s)", fontsize=14)
    plt.title(graph_title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_disk_read_throughput(ps_df, save_path, graph_title):
    """
    Plot Disk Read Throughput

    This function reads disk read throughput data from a pandas DataFrame and plots it as a
    scatter plot against time.

    Parameters
    ----------
    ps_df : pd.Dataframe
        The DataFrame containing the disk read throughput data.
    save_dir : str
        The directory where the generated plot image will be saved.
    """
    plt.plot(
        ps_df["datetime"],
        ps_df["read_throughput"],
        marker="o",
        markersize=5,
        color="blue",
        linestyle="-",
    )
    plt.xlabel("Time", fontsize=14)
    plt.xticks(rotation=315, ha="left")
    plt.ylabel("Disk Read Throughput (Bytes/s)", fontsize=14)
    plt.title(graph_title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_disk_write_throughput(ps_df, save_path, graph_title):
    """
    Plot Disk Write Throughput

    This function reads disk write throughput data from a pandas DataFrame and plots it as a
    scatter plot against time.

    Parameters
    ----------
    ps_df : pd.Dataframe
        The DataFrame containing the disk write throughput data.
    save_dir : str
        The directory where the generated plot image will be saved.
    """
    plt.plot(
        ps_df["datetime"],
        ps_df["write_throughput"],
        marker="o",
        markersize=5,
        color="blue",
        linestyle="-",
    )
    plt.xlabel("Time", fontsize=14)
    plt.xticks(rotation=315, ha="left")
    plt.ylabel("Disk Write Throughput (Bytes/s)", fontsize=14)
    plt.title(graph_title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ram_utilization(ps_df, save_path, graph_title):
    """
    Plot RAM Utilization

    This function reads RAM utilization data from a pandas DataFrame and plots it as a
    scatter plot against time.

    Parameters
    ----------
    ps_df : pd.Dataframe
        The DataFrame containing the RAM utilization data.
    save_dir : str
        The directory where the generated plot image will be saved.
    """
    plt.plot(
        ps_df["datetime"],
        ps_df["ram_util_mib"],
        marker="o",
        markersize=5,
        color="blue",
        linestyle="-",
    )
    plt.xlabel("Time", fontsize=14)
    plt.xticks(rotation=315, ha="left")
    plt.ylabel("RAM Utilization (MiB)", fontsize=14)
    plt.title(graph_title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cpu_utilization(ps_df, save_path, graph_title):
    """
    Plot CPU Utilization

    This function reads CPU utilization data from a pandas DataFrame and plots it as a
    scatter plot against time.

    Parameters
    ----------
    ps_df : pd.Dataframe
        The DataFrame containing the CPU utilization data.
    save_dir : str
        The directory where the generated plot image will be saved.
    """
    plt.plot(
        ps_df["datetime"],
        ps_df["cpu_util"],
        marker="o",
        markersize=5,
        color="blue",
        linestyle="-",
    )
    plt.xlabel("Time", fontsize=14)
    plt.xticks(rotation=315, ha="left")
    plt.ylabel("CPU Utilization (%)", fontsize=14)
    plt.title(graph_title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gpu_memory_usage(gpu_dfs, save_path, graph_title):
    # TODO: Refactor this function to correctly handle situations where each GPU has
    # a different RAM size. The current implementation assumes that all GPUs have the
    # same RAM size.
    """
    Plot GPU Memory Usage

    This function reads GPU memory usage data from pandas dataframes. The monitored data
    may contain multiple GPUs, so this function expects a list of dataframes, and plots
    the GPU memory usage data for each GPU as a scatter plot against time.

    Extra code is added to ensure that redundant data is not shown if there is only one
    GPU.

    Parameters
    ----------
    gpu_dfs : list of pd.Dataframe
        The list of dataframes containing the GPU memory usage data.
    save_dir : str
        The directory where the generated plot image will be saved.
    """
    # Define colors for different GPUs to be plotted
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    fig, ax1 = plt.subplots()

    for i, gpu_df in enumerate(gpu_dfs):
        ax1.plot(
            gpu_df["datetime"],
            gpu_df["mem_perc"],
            marker="o",
            markersize=5,
            color=colors[i % len(colors)],
            linestyle="-",
            label=f"GPU {i} (%)",
        )

    ax1.set_xlabel("Time", fontsize=14)
    ax1.set_ylabel("GPU Memory Usage (%)", fontsize=14)
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    for i, gpu_df in enumerate(gpu_dfs):
        ax2.plot(
            gpu_df["datetime"],
            gpu_df["mem_mib"],
            marker="x",
            markersize=5,
            color=colors[i % len(colors)],
            linestyle="--",
            label=f"GPU {i} (Size)",
        )

    ax2.set_ylabel("GPU Memory Usage (MB)", fontsize=14)
    ax2.tick_params(axis="y")

    if len(gpu_dfs) > 1:
        fig.legend()

    plt.title(graph_title, fontsize=16)
    plt.grid(True)

    # This seems to be the only way to rotate the x-axis labels in this case.
    for label in ax1.get_xticklabels():
        label.set_rotation(-45)
        label.set_horizontalalignment("left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_gpu_utilization(gpu_dfs, save_path, graph_title):
    """
    Plot GPU Utilization

    This function reads GPU utilization data from pandas dataframes. The monitored data
    may contain multiple GPUs, so this function expects a list of dataframes, and plots
    the GPU utilization data for each GPU as a scatter plot against time.

    Extra code is added to ensure that redundant data is not shown if there is only one
    GPU.

    Parameters
    ----------
    gpu_dfs : list of pd.Dataframe
        The list of dataframes containing the GPU utilization data.
    save_dir : str
        The directory where the generated plot image will be saved.
    """
    # Define colors for different GPUs to be plotted
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    for i, gpu_df in enumerate(gpu_dfs):
        plt.plot(
            gpu_df["datetime"],
            gpu_df["util"],
            marker="o",
            markersize=5,
            color=colors[i % len(colors)],
            linestyle="-",
            label=f"GPU {i}",
        )

    if len(gpu_dfs) > 1:
        plt.legend()

    plt.xlabel("Time", fontsize=14)
    plt.xticks(rotation=315, ha="left")
    plt.ylabel("GPU Utilization (%)", fontsize=14)
    plt.title(graph_title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_iops(ps_df, save_path, graph_title):
    """
    Plot Disk IOPS

    This function reads disk IOPS (Input/Output Operations Per Second) data
    from a pandas DataFrame and plots it as a scatter plot against time.

    Parameters
    ----------
    ps_df : pd.Dataframe
        The DataFrame containing the disk IOPS.
    save_dir : str
        The directory where the generated plot image will be saved.
    """
    plt.plot(
        ps_df["datetime"],
        ps_df["read_iops"],
        marker="o",
        markersize=5,
        color="blue",
        linestyle="-",
        label="Read IOPS",
    )
    plt.plot(
        ps_df["datetime"],
        ps_df["write_iops"],
        marker="x",
        markersize=5,
        color="orange",
        linestyle="-",
        label="Write IOPS",
    )
    plt.xlabel("Time", fontsize=14)
    plt.xticks(rotation=315, ha="left")
    plt.ylabel("Disk IOPS", fontsize=14)
    plt.title(graph_title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_disk_read_iops(ps_df, save_path, graph_title):
    """
    Plot Disk read IOPS

    This function reads disk read IOPS (Input/Output Operations Per Second)
    data from a pandas DataFrame and plots it as a scatter plot against time.

    Parameters
    ----------
    ps_df : pd.Dataframe
        The DataFrame containing the disk write IOPS.
    save_dir : str
        The directory where the generated plot image will be saved.
    """
    plt.plot(
        ps_df["datetime"],
        ps_df["read_iops"],
        marker="o",
        markersize=5,
        color="blue",
        linestyle="-",
    )
    plt.xlabel("Time", fontsize=14)
    plt.xticks(rotation=315, ha="left")
    plt.ylabel("Disk Read IOPS", fontsize=14)
    plt.title(graph_title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_disk_write_iops(ps_df, save_path, graph_title):
    """
    Plot Disk Write IOPS

    This function reads disk write IOPS (Input/Output Operations Per Second)
    data from a pandas DataFrame and plots it as a scatter plot against time.

    Parameters
    ----------
    ps_df : pd.Dataframe
        The DataFrame containing the disk write IOPS.
    save_dir : str
        The directory where the generated plot image will be saved.
    graph_title : str
        The title that will be shown at the top of the graph.
    """
    plt.plot(
        ps_df["datetime"],
        ps_df["write_iops"],
        marker="o",
        markersize=5,
        color="blue",
        linestyle="-",
    )
    plt.xlabel("Time", fontsize=14)
    plt.xticks(rotation=315, ha="left")
    plt.ylabel("Disk Write IOPS", fontsize=14)
    plt.title(graph_title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def process_gpu_log(logfile):
    """
    Process GPU Log Data

    This function parses a log file containing GPU metrics and organizes the data into
    structured dataframes for each GPU. It extracts metrics such as temperature, GPU
    utilization, power usage, and memory usage, and timestamp info.

    Parameters
    ----------
    logfile : str
        The path to the log file to be processed.

    Returns
    -------
    gpu_dfs : list of pandas.DataFrame
        A list of dataframes, where each dataframe contains the processed log data for
        one GPU.
    """

    df = pd.read_csv(logfile)
    df = df.rename(
        columns={
            "Timestamp (Unix)": "timestamp",
            "GPU ID": "id",
            "Utilization (%)": "util",
            "Memory (Bytes)": "mem_b",
            "Memory (%)": "mem_perc",
            "Temperature (°C)": "temp",
            "Power Usage (W)": "power",
        }
    )

    gpu_dfs = []
    max_gpu_id = df["id"].nunique()
    for gpu_id in range(max_gpu_id):
        gpu_df = df[df["id"] == gpu_id].reset_index(drop=True)
        gpu_df["datetime"] = pd.to_datetime(gpu_df["timestamp"], unit="s")
        gpu_df["mem_mib"] = gpu_df["mem_b"] / 1024**2
        gpu_dfs.append(gpu_df)

    return gpu_dfs


def process_psinfo_log(logfile):
    """
    Process Process Info Log Data

    This function parses a log file containing process information metrics and organizes
    the data into a structured dataframe. It extracts metrics such as CPU percent, RAM
    used, number of threads, disk read/write bytes, and disk read/write operations.

    Parameters
    ----------
    logfile : str
        The path to the log file to be processed.

    Returns
    -------
    psinfo_df : pandas.DataFrame
        A dataframe containing the processed log data for process information metrics.
    """

    df = pd.read_csv(logfile)

    df = df.rename(
        columns={
            "Timestamp (Unix)": "timestamp",
            "CPU Utilization (%)": "cpu_util",
            "Thread Count": "num_threads",
            "RAM Utilization (Bytes)": "ram_util_b",
            "RAM Utilization (%)": "ram_util_perc",
            "Disk Read (Bytes)": "disk_read_b",
            "Disk Write (Bytes)": "disk_write_b",
            "Disk Read Operations (Count)": "disk_read_ops",
            "Disk Write Operations (Count)": "disk_write_ops",
        }
    )

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["ram_util_mib"] = df["ram_util_b"] / 1024**2
    df["time_diff"] = df["datetime"].diff().dt.total_seconds()
    df["read_diff_b"] = df["disk_read_b"].diff()
    df["write_diff_b"] = df["disk_write_b"].diff()
    df["read_throughput"] = df["read_diff_b"] / df["time_diff"]
    df["write_throughput"] = df["write_diff_b"] / df["time_diff"]
    df["read_ops_diff"] = df["disk_read_ops"].diff()
    df["write_ops_diff"] = df["disk_write_ops"].diff()
    df["read_iops"] = df["read_ops_diff"] / df["time_diff"]
    df["write_iops"] = df["write_ops_diff"] / df["time_diff"]

    return df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Process log files for GPU and process info."
    )
    argparser.add_argument(
        "dir", type=str, help="The directory containing the log files to be processed."
    )
    argparser.add_argument(
        "save_dir",
        type=str,
        help="The directory where the generated plot images will be saved.",
    )
    args = argparser.parse_args()

    args.dir = Path(args.dir).expanduser().resolve()
    gpuinfo = args.dir / "gpuinfo.csv"
    psinfo = args.dir / "psinfo.csv"
    graph_title = os.path.join(args.dir.parent.name, args.dir.name)  # noqa: F821

    args.save_dir = Path(args.save_dir).expanduser().resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    gpu_dfs = process_gpu_log(gpuinfo)
    psinfo_df = process_psinfo_log(psinfo)

    plot_disk_read_iops(psinfo_df, args.save_dir / "disk_write_iops.png", graph_title)
    plot_disk_write_iops(psinfo_df, args.save_dir / "disk_read_iops.png", graph_title)
    plot_iops(psinfo_df, args.save_dir / "disk_iops.png", graph_title)
    plot_gpu_utilization(gpu_dfs, args.save_dir / "gpu_utilization.png", graph_title)
    plot_gpu_memory_usage(gpu_dfs, args.save_dir / "gpu_memory_usage.png", graph_title)
    plot_cpu_utilization(psinfo_df, args.save_dir / "cpu_utilization.png", graph_title)
    plot_ram_utilization(psinfo_df, args.save_dir / "ram_utilization.png", graph_title)
    plot_disk_write_throughput(
        psinfo_df, args.save_dir / "disk_write_throughput.png", graph_title
    )
    plot_disk_read_throughput(
        psinfo_df, args.save_dir / "disk_read_throughput.png", graph_title
    )
    plot_disk_throughput(psinfo_df, args.save_dir / "disk_throughput.png", graph_title)

    # psinfo_df.to_csv(args.save_dir / "psinfo_processed.csv", index=False)
    # print(psinfo_df["datetime"])
