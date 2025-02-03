# py-process-monitor
- [📝 Overview](#-overview)
- [🛠️ Installation](#%EF%B8%8F-installation)
  - [Linux](#linux)
- [📚 Usage](#-usage)
# 📝 Overview
A Python-based process monitoring tool that allows you to monitor processes by command or PID.
This monitor provides the functionality to monitor the CPU, RAM, and DISK utilization of a specific process and all of that process's children. It also monitors the system-wide GPU utilization, given that it is an NVIDIA GPU and its drivers are properly installed. If multiple NVIDIA GPUs are detected,
they will all be monitored and have their utilization metrics recorded separately.  
All monitoring and command output will be placed in a single, configurable location.

This repository includes integrated visualization tools for analyzing resource utilization data. The built-in graphing functionality automatically generates a suite of informative charts, including:
* CPU Utilization
* RAM Utilization
* Read/Write IO Metrics:
  * Throughput
  * IOPS
  * Total Data
  * Operation Count
* Individual GPU performance, showcasing:
  * Utilization rates
  * Memory consumption


> [!CAUTION]
> Only Linux Systems with NVIDIA GPU's are supported at the moment.

# 🛠️ Installation
## Linux

**1. Clone the Repository**
```
git clone https://github.com/WellOiledMachine/py-process-monitor.git
cd py-process-monitor
```
**2. Install requirements**
> [!IMPORTANT]  
> It is recommended to create a virtual environment to manage dependencies and avoid potential conflicts with other Python projects. For a comprehensive guide on creating and using virtual environments, check out the [Python Packaging User Guide on Virtual Environments](https://realpython.com/python-virtual-environments-a-primer/). Otherwise, a virtual environment can easily be created and activated with these commands:
> ```
> python3 -m venv monitor
> source ./monitor/bin/activate
> ```
The required python modules can be installed using this command:
```
pip install -r ./requirements.txt
```
**3. Done!**

# 📚 Usage
## Monitor
Here is the command line usage of the monitor script:
```
usage: python monitor.py [-h] (-p PID | -c ...) [-i INTERVAL] [-o OUTPUT_DIR]

Monitor the system utilization of a process.

options:
  -h, --help            show this help message and exit
  -p PID, --pid PID     The PID of the process to monitor.
  -c ..., --command ...
                        The command to run the process to monitor.
  -i INTERVAL, --interval INTERVAL
                        The interval (in seconds) between monitoring samples. Default: 1
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        The directory to save the monitoring results. Default: ../monitoring_results
```

These options allow for the monitor to be used either by passing it the PID of an already running process, or by passing it a shell command that it will execute and monitor for you.

> [!NOTE]
>When using this script, it's important to note that the output directory specified by the user serves as a parent directory for the actual results.  
>Here's how the output structure works:
>1. User-specified directory: This is the top-level directory you provide as an argument.
>2. Timestamp subdirectory: Within the user-specified directory, the script automatically creates a new subdirectory named with a timestamp (e.g., "2025_02_03_16-30-45").
>3. Output files: All generated output files are placed inside this timestamp subdirectory.  
>
>This structure offers several benefits:
>* It prevents overwriting previous results
>* It organizes outputs by run time
>* It allows for easy identification of specific execution instances

> [!NOTE]
> The `--interval` and `--output_dir` options are not required, as they have defaults!

> [!WARNING]
> If you wish to use the `-c / --command` option, you MUST add it as LAST OPTION SPECIFIED in the full command.
> Otherwise, unexpected behavior will occur!

**Examples**
```
# CORRECT way to pass a command to the monitor script.
# Note how -c is the last flag passed to the monitor.
python src/monitor.py -o ../monitor_results -i 3 -c ~/example_script.sh arg1 arg2

# INCORRECT way to pass a command to the monitor script. 
# The -c option is not the last option!
python src/monitor.py -c ~/example_script.sh arg1 arg2 -o ../monitor_results -i 3

# Passing a PID to the monitor is easy. The flag order doesn't matter.
python src/monitor.py -p 12345 -o ../monitor_results
```

## Graphing
Here is the command line usage of our built-in graphing script:
```
usage: python Graphing.py [-h] source_dir save_dir

positional arguments:
  source_dir  The directory containing the GPU and PS logs.
  save_dir    The directory to save the graphs to.

options:
  -h, --help  show this help message and exit
```

> [!NOTE]
> As explained above, the directory chosen for monitor output is only a parent directory in which a timestamp-named subdirectory will be automatically created, and that is where output files are placed.
> The graphing script doesn't automatically search this subdirectory, only the directory that you specify. Because of this, you MUST include the subdirectory in the source_dir path. 