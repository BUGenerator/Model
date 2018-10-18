# Working with ENG Grid

## First Thing to Know

- Installation of our enviroument in Anaconda: ~3.1 GB
- Home folder quota on ENGNAS: 10GB
- fastest file access on ENG Grid: `/tmp` (but FILES WILL BE LOST AFTER REBOOT! Remember to copy the files out of there)
- Everything related to Python in our environment should be Python 3(.6)!

To check the exact quota of file storage:

```shell
quota -s
```
To check which folder is large in the current working directory:

```shell
du --max-depth=1 -h
```

## Hardware-related Resources

ENG Grid includes computers in PHO 305/307 and some other computers. If possible try to work on PHO 305 (VLSI, better hardware) or remotely with ssh://engineering-grid.bu.edu (computers in other labs is more powerful!).

- Help docs on Grid: http://collaborate.bu.edu/engit/Grid/
- Grid resource monitor: http://eng-grid-monitor.bu.edu
- There is some GPU resource here (enrollment required), for more: http://collaborate.bu.edu/engit/Grid/GridGPU

## Development Environment

Development environment has everything like Jupyter, etc.

We are not using Docker because it needs root access, while Anaconda does not.

- Init Anaconda environment to your HOME folder:
   - firstly install Anaconda on the machine
      - which already exists on ENG Grid (so you don't need to install again)
      - to make sure: `module load anaconda` (no output - correct)
   - then download [init_dl](init_dl) and execute `./init_dl` (anywhere is OK)
      - If failed in the middle: check whether HOME folder is full (see the previous section about the quota)
- Start the environment:
   - download [source_dl](source_dl) and `source source_dl` (anywhere is OK), and it will take you to `/tmp/$USER`
   - For each shell window you should `source` before you want to work with Deep Learning every time
   - Put necessary data file inside your tmp folder

After starting the environment:

- `jupyter notebook` to start the notebook (necessary for opening `.ipynb`)
- `python` to start the python shell

## GPU support

Currently init_dl and source_dl only supports CPU computation. GPU support will be tried later.

Because ENG Grid is retiring GPU capabilities, ECE (lab) offered a shared machine for us. Please contact @phy25 for more information.

Login on to Grid jump server (Don't do `qlogin`) > `ssh phy25@ece-hpc-01` > Enter password

```shell
# Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz * 24
phy25@ECE-HPC-01:~$ lspci -vnn | grep VGA -A 12
03:00.0 VGA compatible controller [0300]: NVIDIA Corporation GK107GL [Quadro K420] [10de:0ff3] (rev a1) (prog-if 00 [VGA controller])
        Subsystem: NVIDIA Corporation GK107GL [Quadro K420] [10de:1106]
        Physical Slot: 4
        Flags: bus master, fast devsel, latency 0, IRQ 71
        Memory at f8000000 (32-bit, non-prefetchable) [size=16M]
        Memory at e0000000 (64-bit, prefetchable) [size=256M]
        Memory at f0000000 (64-bit, prefetchable) [size=32M]
        I/O ports at e000 [size=128]
        [virtual] Expansion ROM at f9000000 [disabled] [size=512K]
        Capabilities: <access denied>
        Kernel driver in use: nvidia
        Kernel modules: nvidiafb, nouveau, nvidia_384_drm, nvidia_384
```
