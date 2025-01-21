# py-process-monitor
- [📝 Overview](#-overview)
- [🛠️ Installation](#%EF%B8%8F-installation)
  - [Linux](#linux)
- [📚 Usage](#-usage)
# 📝 Overview
A Python-based process monitoring tool that allows you to monitor processes by command or PID. Includes a monitor spawner for easy python code integration and a test script for demonstration.
This monitor provides the functionality to monitor the CPU, RAM, and DISK utilization of a specific process and all of that process's children. It also monitors the system-wide GPU utilization.
This information is recorded while a specific process runs, and will be placed in a configurable location once the process has finished.

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
> python -m venv monitor
> source ./monitor/bin/activate
> ```
The required python modules can be installed using this command:
```
pip install -r ./requirements.txt
```
**3. Done!**

# 📚 Usage
Here is the command line usage of this script:
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
> The `--interval` and `--output_dir` options are not required, as they have defaults!

> [!WARNING]
> If you wish to use the `-c / --command` option, you MUST add it as LAST OPTION SPECIFIED in the full command.
> Otherwise, unexpected behavior will occur!

**Examples**
```
# CORRECT way to pass a command to the monitor script.
# Note how -c is the last flag passed to the monitor.
python3 src/monitor.py -o ../monitor_results -i 3 -c ~/example_script.sh arg1 arg2

# INCORRECT way to pass a command to the monitor script. 
# The -c option is not the last option!
python3 src/monitor.py -c ~/example_script.sh arg1 arg2 -o ../monitor_results -i 3

# Passing a PID to the monitor is easy. The flag order doesn't matter.
python3 src/monitor.py -p 12345 -o ../monitor_results
```
