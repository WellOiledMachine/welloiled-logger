import argparse
import math
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Graphing:
    """
    Graphing class for visualizing custom psinfo.csv and gpuinfo.csv logs.
    The class is initialized with the source directory containing the logs
    and the save directory for the generated graphs.
    The class automatically searches for the psinfo.csv and gpuinfo.csv logs
    in the source directory and processes them for graphing. Then, it generates
    a series of plots based on the processed data and saves them in the save
    directory. Each plot shows a different aspect of system performance
    measured over time.
    """

    def __init__(self, source_dir, save_dir):
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.graph_title = self.source_dir.parent.name + "\\" + self.source_dir.name

        self.marker = "o"
        self.markersize = 4
        self.linestyle = "-"
        self.line_color = "blue"
        self.second_line_color = "orange"
        self.xtick_rotation = 310

        psinfo_log = self.source_dir / "psinfo.csv"
        gpuinfo_log = self.source_dir / "gpuinfo.csv"
        self._process_psinfo(psinfo_log)
        self._process_gpuinfo(gpuinfo_log)
        self._create_xtick_positions()

    def _process_psinfo(self, psinfo_log):
        """
        Part of class initialization:
        Process the psinfo.csv log file to prepare the data for graphing.

        Parameters:
            psinfo_log (Path): Path to the psinfo.csv log file.
        """
        df = pd.read_csv(psinfo_log)

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
        # Make RAM utilization human-readable - Should be able to always use MiB
        df["ram_util_mib"] = df["ram_util_b"] / 1024**2
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        # Calculate time difference between each row in seconds
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

        # Set format for x-axis labels
        if df["timestamp"].dt.date.nunique() > 1:
            self.time_format = mdates.DateFormatter("%d-%H:%M:%S")
            self.time_label_format = "D-H:M:S UTC"
        else:
            self.time_format = mdates.DateFormatter("%H:%M:%S")
            self.time_label_format = "H:M:S UTC"

        ## DATA UNIT CONVERSION ##
        # Initialize potential volume units for disk read and write amounts
        volume_units = ["B", "KiB", "MiB", "GiB", "TiB"]
        # For both read and write values:
        # Get the power of 1024 that the max value of the column is closest to
        read_scale_factor = min(
            math.floor(math.log(df["disk_read_b"].max(), 1024)), len(volume_units) - 1
        )
        write_scale_factor = min(
            math.floor(math.log(df["disk_write_b"].max(), 1024)), len(volume_units) - 1
        )
        # Use calculated scale factors to convert bytes to human-readable units
        df["disk_read_h"] = df["disk_read_b"].apply(
            lambda x: x / 1024**read_scale_factor
        )
        df["disk_write_h"] = df["disk_write_b"].apply(
            lambda x: x / 1024**write_scale_factor
        )

        # Use human-readable data volume columns to calculate throughput
        df["read_throughput"] = df["disk_read_h"].diff().fillna(0) / df["time_diff"]
        df["write_throughput"] = df["disk_write_h"].diff().fillna(0) / df["time_diff"]

        # Calculate IOPS
        df["read_iops"] = df["disk_read_ops"].diff().fillna(0) / df["time_diff"]
        df["write_iops"] = df["disk_write_ops"].diff().fillna(0) / df["time_diff"]

        # Save the converted dataset and other relevant info for graphing purposes
        self.disk_read_unit = volume_units[read_scale_factor]
        self.disk_write_unit = volume_units[write_scale_factor]
        self.ps_df = df

    def _process_gpuinfo(self, gpuinfo_log):
        """
        Part of class initialization:
        Process the gpuinfo.csv log file to prepare the data for graphing.

        Parameters:
            gpuinfo_log (Path): Path to the gpuinfo.csv log file.
        """
        df = pd.read_csv(gpuinfo_log)
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

        # We want a list of dataframes, one for each GPU. This will allow us to plot
        # each GPU's data separately. If only one GPU is present, the list will contain
        # only one dataframe.
        gpu_dfs = []
        max_gpu_id = df["id"].nunique()  # Get the number of unique GPU IDs
        for gpu_id in range(max_gpu_id):
            # Filter the dataset to only include the current GPU ID
            gpu_df = df[df["id"] == gpu_id].reset_index(drop=True)
            # Convert timestamp to datetime
            gpu_df["timestamp"] = pd.to_datetime(gpu_df["timestamp"], unit="s")
            # Convert memory usage from bytes to MiB
            gpu_df["mem_mib"] = gpu_df["mem_b"] / 1024**2
            # Add the converted dataset to the list of GPU datasets
            gpu_dfs.append(gpu_df)

        self.gpu_dfs = gpu_dfs

    def _create_xtick_positions(self):
        """
        Part of class initialization:
        Create evenly spaced x-ticks for the plots based on the dataset size.
        Automatically sets number of ticks, tick positions, and padding, as well
        as ensuring all data is lined up across all plots.
        """
        # Adjust number of ticks based on dataset size
        df_length = len(self.ps_df)
        num_ticks = min(9, df_length)

        # Handle small datasets
        if df_length <= 2:
            # For very small datasets, use all points as ticks
            ticks = self.ps_df["timestamp"].copy()
        else:
            # Create evenly spaced ticks
            step = max(1, (df_length - 1) // (num_ticks - 1))
            ticks = self.ps_df["timestamp"].iloc[::step].reset_index(drop=True)

        # Calculate average time difference between ticks
        tick_avg_diff = np.diff(ticks).mean()

        # Calculate left padding
        padding_factor = 0.1
        left_pad = ticks.iloc[0] - (tick_avg_diff * padding_factor)

        # Grab last timestamp and last tick
        last_timestamp = self.ps_df["timestamp"].iloc[-1]
        last_tick = ticks.iloc[-1]

        # Check if extra tick should be added
        if last_timestamp > last_tick:
            extra_tick = last_tick + tick_avg_diff
            ticks = pd.concat([ticks, pd.Series([extra_tick])], ignore_index=True)

        right_pad = ticks.iloc[-1] + tick_avg_diff * padding_factor

        self.padding = (left_pad, right_pad)
        self.ticks = ticks

    def _plot_y_against_time(self, df, y, title, y_label, save_name):
        """
        Generic plotting method for a single y-axis against time.
        Customized to work with the processed data from the psinfo.csv and gpuinfo.csv.
        """
        fig, ax = plt.subplots()
        ax.plot(
            df["timestamp"],
            df[y],
            marker=self.marker,
            markersize=self.markersize,
            color=self.line_color,
            linestyle=self.linestyle,
        )
        # Title info
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(f"Time ({self.time_label_format})", fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)

        # Tick manipulation
        ax.set_xticks(self.ticks)
        ax.set_xlim(self.padding)
        ax.tick_params(axis="both", labelsize=10)
        ax.tick_params(axis="x", rotation=self.xtick_rotation)
        ax.xaxis.set_major_formatter(self.time_format)

        # Grid and layout
        ax.grid(True)
        fig.tight_layout()

        # Save the plot
        save_path = self.save_dir / save_name
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    def _plot_two_y_against_time(
        self, df, y1, y2, title, y1_label, y2_label, save_name
    ):
        """
        Generic plotting method for two y-datasets against time.
        Customized to work with the processed data from the psinfo.csv and gpuinfo.csv.
        """
        fig, ax = plt.subplots()
        ax.plot(
            df["timestamp"],
            df[y1],
            marker=self.marker,
            markersize=self.markersize,
            color=self.line_color,
            linestyle=self.linestyle,
            label=y1_label,
        )
        ax.plot(
            df["timestamp"],
            df[y2],
            marker=self.marker,
            markersize=self.markersize,
            color=self.second_line_color,
            linestyle=self.linestyle,
            label=y2_label,
        )
        # Title info
        ax.set_xlabel(f"Time ({self.time_label_format})", fontsize=14)
        ax.set_ylabel(y1_label, fontsize=14)
        ax.set_title(title, fontsize=16)

        # Tick manipulation
        for label in ax.get_xticklabels():
            label.set_rotation(self.xtick_rotation)
            label.set_horizontalalignment("left")
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xticks(self.ticks)
        ax.set_xlim(self.padding)
        ax.xaxis.set_major_formatter(self.time_format)

        # Grid, layout, and legend
        ax.grid(True)
        fig.tight_layout()
        ax.legend()

        # Save the plot
        save_path = self.save_dir / save_name
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    def _plot_dual_yaxis_against_time(
        self, df, y1, y2, title, y1_label, y2_label, save_name
    ):
        """
        Function that plots dual y-axis data against time.
        'dual y-axis data' refers to two datasets that are identical to each other,
        relative to their maximum values. Basically, plotting the two datasets should
        have their datapoints line up exactly when plotted separately.
        For example, memory utilization (RAM and GPU memory) is plotted in MiB and %,
        which are identical in terms of the data they represent.
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(
            df["timestamp"],
            df[y1],
            marker=self.marker,
            markersize=self.markersize,
            color=self.line_color,
            linestyle=self.linestyle,
            label=y1_label,
        )
        ax2.plot(
            df["timestamp"],
            df[y2],
            marker=self.marker,
            markersize=self.markersize,
            color=self.line_color,
            linestyle=self.linestyle,
            label=y2_label,
        )
        # Title info
        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel(f"Time ({self.time_label_format})", fontsize=14)
        ax1.set_ylabel(y1_label, fontsize=14)
        ax2.set_ylabel(y2_label, fontsize=14)

        # Tick manipulation
        for label in ax1.get_xticklabels():
            label.set_rotation(self.xtick_rotation)
            label.set_horizontalalignment("left")
        ax1.tick_params(axis="both", labelsize=10)
        ax1.set_xticks(self.ticks)
        ax1.set_xlim(self.padding)
        ax1.xaxis.set_major_formatter(self.time_format)

        # Grid and layout
        ax1.grid(True)
        fig.tight_layout()

        # Save the plot
        save_path = self.save_dir / save_name
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    def plot_cpu_utilization(self):
        """Plot CPU utilization against time."""
        self._plot_y_against_time(
            self.ps_df,
            "cpu_util",
            self.graph_title,
            "CPU Utilization (%)",
            "cpu_utilization.png",
        )

    def plot_ram_utilization(self):
        """
        Plot RAM utilization against time.
        Uses dual y-axis plot to show MiB and %.
        """
        self._plot_dual_yaxis_against_time(
            self.ps_df,
            "ram_util_mib",
            "ram_util_perc",
            self.graph_title,
            "RAM Utilization (MiB)",
            "RAM Utilization (%)",
            "ram_utilization.png",
        )

    def plot_disk_read_throughput(self):
        """Plot disk read throughput against time."""
        self._plot_y_against_time(
            self.ps_df,
            "read_throughput",
            self.graph_title,
            f"Disk Read Throughput ({self.disk_read_unit}/s)",
            "disk_read_throughput.png",
        )

    def plot_disk_write_throughput(self):
        """Plot disk write throughput against time."""
        self._plot_y_against_time(
            self.ps_df,
            "write_throughput",
            self.graph_title,
            f"Disk Write Throughput ({self.disk_write_unit}/s)",
            "disk_write_throughput.png",
        )

    def plot_disk_iops(self):
        """Plot disk read and write IOPS against time."""
        self._plot_two_y_against_time(
            self.ps_df,
            "read_iops",
            "write_iops",
            self.graph_title,
            "Read IOPS",
            "Write IOPS",
            "disk_iops.png",
        )

    def plot_disk_read_iops(self):
        """Plot disk read IOPS against time."""
        self._plot_y_against_time(
            self.ps_df,
            "read_iops",
            self.graph_title,
            "Disk Read IOPS",
            "disk_read_iops.png",
        )

    def plot_disk_write_iops(self):
        """Plot disk write IOPS against time."""
        self._plot_y_against_time(
            self.ps_df,
            "write_iops",
            self.graph_title,
            "Disk Write IOPS",
            "disk_write_iops.png",
        )

    def plot_total_data_read(self):
        """Plot total data read against time."""
        self._plot_y_against_time(
            self.ps_df,
            "disk_read_h",
            self.graph_title,
            f"Total Data Read ({self.disk_read_unit})",
            "total_data_read.png",
        )

    def plot_total_data_written(self):
        """Plot total data written against time."""
        self._plot_y_against_time(
            self.ps_df,
            "disk_write_h",
            self.graph_title,
            f"Total Data Written ({self.disk_write_unit})",
            "total_data_written.png",
        )

    def plot_read_count(self):
        """Plot read operation count against time."""
        self._plot_y_against_time(
            self.ps_df,
            "disk_read_ops",
            self.graph_title,
            "Read Operation Count",
            "read_operation_count.png",
        )

    def plot_write_count(self):
        """Plot write operation count against time."""
        self._plot_y_against_time(
            self.ps_df,
            "disk_write_ops",
            self.graph_title,
            "Write Operation Count",
            "write_operation_count.png",
        )

    def plot_gpu_utilization(self):
        """Plot each GPU's individual Utilization against time."""
        for gpu_df in self.gpu_dfs:
            self._plot_y_against_time(
                gpu_df,
                "util",
                self.graph_title,
                "GPU Utilization (%)",
                f"gpu{gpu_df['id'].iloc[0]}_utilization.png",
            )

    def plot_gpu_memory(self):
        """
        Plot each GPU's individual Memory Usage against time.
        Uses dual y-axis plot to show MiB and % for each GPU.
        """
        for gpu_df in self.gpu_dfs:
            self._plot_dual_yaxis_against_time(
                gpu_df,
                "mem_mib",
                "mem_perc",
                self.graph_title,
                "Memory Usage (MiB)",
                "Memory Usage (%)",
                f"gpu{gpu_df['id'].iloc[0]}_memory.png",
            )

    def plot_psinfo(self):
        """Plot all PS info graphs."""
        self.plot_cpu_utilization()
        self.plot_ram_utilization()
        self.plot_disk_read_throughput()
        self.plot_disk_write_throughput()
        self.plot_disk_iops()
        self.plot_disk_read_iops()
        self.plot_disk_write_iops()
        self.plot_total_data_read()
        self.plot_total_data_written()
        self.plot_read_count()
        self.plot_write_count()

    def plot_gpuinfo(self):
        """Plot all GPU info graphs."""
        self.plot_gpu_utilization()
        self.plot_gpu_memory()

    def plot_all(self):
        """Plot all graphs."""
        self.plot_psinfo()
        self.plot_gpuinfo()


if __name__ == "__main__":
    """
    Command-line interface for the Graphing class.
    Allows the user to specify the source directory containing the logs,
    and the save directory for the generated graphs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_dir",
        type=str,
        help="The directory containing the GPU and PS logs.",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="The directory to save the graphs to.",
    )
    args = parser.parse_args()
    graphing = Graphing(args.source_dir, args.save_dir)
    graphing.plot_all()
