# First thing to know

- Installation of our enviroument in Anaconda: ~3.1 GB
- Home folder quota on ENGNAS: 10GB
- fastest file access on ENG Grid: `/tmp`
- Everything related to Python in our environment should be Python 3(.6)!

To check quota of file storage:

```shell
quota -s
```
To check which folder is large in the current working directory:

```shell
du --max-depth=1 -h
```

# Development Environment

With everything like Jupyter, etc.

We are not using Docker because it needs root access, while anaconda does not.

- Init Anaconda environment to your HOME folder:
   - firstly install Anaconda on the machine, which exists on ENG Grid
   - then download (init_dl)[init_dl] and execute `./init_dl`
      - If failed in the middle: check whether HOME folder is full (see the last section about the quota)
- Start the environment:
   - download (source_dl)[source_dl] and `source source_dl`, and it will take you to `/tmp/$USER`
   - For each shell window you should `source` every time
   - Put necessary data file inside your tmp folder

After starting the environment:

- `jupyter notebook` to start the notebook
- `python` to start the python shell
