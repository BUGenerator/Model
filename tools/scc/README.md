# Welcome to SCC!

## Useful commands

- Jupyter notebook: `jupyter notebook --no-browser --ip ssh-path.bu.edu` where ssh-path.bu.edu is the scc SSH hostname you are connecting to
  - Then you can just copy-paste the URL shown in the SSH

- `source source_dl` to load modules we need
- `module list` see currently-loaded module
- https://acct.bu.edu/cgi-bin/perl/secure/redirect_sccmgmt.pl to see (cached) quota

## Submit the job

Firstly cd to here. (On SCC our path is at `/projectnb/ece601/BUGenerator/Model/tools/scc`)

- `git pull -r` pull with rebase
- `qsub batch.bash` to schedule a job (Specific arguments has been filed inside it)
- `qstat -u MYUSERNAME` to see job status
- `qdel JOBID` to cancel job
- `ls -l` to see stdout file (BUG*)
