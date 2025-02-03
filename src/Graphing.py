import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


# TODO: Add docstrings to the class and methods
#       Fix date formatting on x-axis
class Graphing:
    def __init__(self, source_dir, save_dir):
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gpuinfo_log = self.source_dir / "gpuinfo.csv"
        self.psinfo_log = self.source_dir / "psinfo.csv"
        self.graph_title = self.source_dir.parent.name + "\\" + self.source_dir.name

        self._process_psinfo()
        self._process_gpuinfo()

    def _process_psinfo(self):
        df = pd.read_csv(self.psinfo_log)

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

        self.ps_df = df

    def _process_gpuinfo(self):
        df = pd.read_csv(self.gpuinfo_log)
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

        self.gpu_dfs = gpu_dfs

    def plot_cpu_utilization(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["cpu_util"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("CPU Utilization (%)", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        save_path = self.save_dir / "cpu_utilization.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_ram_utilization(self):
        fig, ax1 = plt.subplots()

        ax1.plot(
            self.ps_df["datetime"],
            self.ps_df["ram_util_mib"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
            label="RAM Utilization (MiB)",
        )
        plt.xlabel("Time", fontsize=14)
        ax1.set_ylabel("RAM Utilization (MiB)", fontsize=14)
        ax1.tick_params(axis="y")

        ax2 = ax1.twinx()
        ax2.plot(
            self.ps_df["datetime"],
            self.ps_df["ram_util_perc"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
            label="RAM Utilization (%)",
        )
        ax2.set_ylabel("RAM Utilization (%)", fontsize=14)
        ax2.tick_params(axis="y")

        plt.title(self.graph_title, fontsize=16)
        ax1.grid(True)

        for label in ax1.get_xticklabels():
            label.set_rotation(315)
            label.set_horizontalalignment("left")

        plt.tight_layout()

        save_path = self.save_dir / "ram_utilization.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_disk_throughput(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["read_throughput"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
            label="Read Throughput",
        )
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["write_throughput"],
            marker="x",
            markersize=4,
            color="orange",
            linestyle="-",
            label="Write Throughput",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Disk Throughput (Bytes/s)", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()

        save_path = self.save_dir / "disk_throughput.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_disk_read_throughput(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["read_throughput"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Disk Read Throughput (Bytes/s)", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        save_path = self.save_dir / "disk_read_throughput.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_disk_write_throughput(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["write_throughput"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Disk Write Throughput (Bytes/s)", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        save_path = self.save_dir / "disk_write_throughput.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_disk_iops(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["read_iops"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
            label="Read IOPS",
        )
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["write_iops"],
            marker="x",
            markersize=4,
            color="orange",
            linestyle="-",
            label="Write IOPS",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Disk IOPS", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()

        save_path = self.save_dir / "disk_iops.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_disk_read_iops(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["read_iops"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Disk Read IOPS", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        save_path = self.save_dir / "disk_read_iops.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_disk_write_iops(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["write_iops"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Disk Write IOPS", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        save_path = self.save_dir / "disk_write_iops.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_total_data_read(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["disk_read_b"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Total Data Read (Bytes)", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        save_path = self.save_dir / "total_data_read.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_total_data_written(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["disk_write_b"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Total Data Written (Bytes)", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        save_path = self.save_dir / "total_data_written.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_read_count(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["disk_read_ops"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Read Operation Count", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        save_path = self.save_dir / "read_operation_count.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_write_count(self):
        plt.plot(
            self.ps_df["datetime"],
            self.ps_df["disk_write_ops"],
            marker="o",
            markersize=4,
            color="blue",
            linestyle="-",
        )
        plt.xlabel("Time", fontsize=14)
        plt.xticks(rotation=315, ha="left")
        plt.ylabel("Write Operation Count", fontsize=14)
        plt.title(self.graph_title, fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        save_path = self.save_dir / "write_operation_count.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_gpu_utilization(self):
        for gpu_df in self.gpu_dfs:
            plt.plot(
                gpu_df["datetime"],
                gpu_df["util"],
                marker="o",
                markersize=4,
                color="blue",
                linestyle="-",
            )

            plt.xlabel("Time", fontsize=14)
            plt.xticks(rotation=315, ha="left")
            plt.ylabel("GPU Utilization (%)", fontsize=14)
            plt.title(self.graph_title, fontsize=16)
            plt.grid(True)
            plt.tight_layout()

            save_path = self.save_dir / f"gpu{gpu_df['id'].iloc[0]}_utilization.png"
            plt.savefig(save_path, dpi=300)
            plt.close()

    def plot_gpu_memory(self):
        for gpu_df in self.gpu_dfs:
            fig, ax1 = plt.subplots()

            ax1.plot(
                gpu_df["datetime"],
                gpu_df["mem_mib"],
                marker="o",
                markersize=4,
                color="blue",
                linestyle="-",
                label="Memory Usage (MiB)",
            )
            plt.xlabel("Time", fontsize=14)
            ax1.set_ylabel("Memory Usage (MiB)", fontsize=14)
            ax1.tick_params(axis="y")

            ax2 = ax1.twinx()
            ax2.plot(
                gpu_df["datetime"],
                gpu_df["mem_perc"],
                marker="o",
                markersize=4,
                color="blue",
                linestyle="-",
                label="Memory Usage (%)",
            )
            ax2.set_ylabel("Memory Usage (%)", fontsize=14)
            ax2.tick_params(axis="y")

            plt.title(self.graph_title, fontsize=16)
            ax1.grid(True)

            for label in ax1.get_xticklabels():
                label.set_rotation(315)
                label.set_horizontalalignment("left")

            plt.tight_layout()
            save_path = self.save_dir / f"gpu{gpu_df['id'].iloc[0]}_memory.png"
            plt.savefig(save_path, dpi=300)
            plt.close()

    def plot_psinfo(self):
        self.plot_cpu_utilization()
        self.plot_ram_utilization()
        self.plot_disk_throughput()
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
        self.plot_gpu_utilization()
        self.plot_gpu_memory()

    def plot_all(self):
        self.plot_psinfo()
        self.plot_gpuinfo()


if __name__ == "__main__":
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
