# Short SBATCH INTRO
Wow you made it, great, I'm so proud.

## How to submit a job
Submit your job via:
    
```bash 
sbatch job.sh
```

It then takes the job script, reads the SBATCH directives:
 - all the top comments starting with `#SBATCH`

then executes the bash script normally after the first not-commented line.

## Testing
To test on RWTH Cluster: set the partition to 'devel'.
This allows for a quick test of the job.
And it doesn't even count to your time quota.
BUT you can only set o a max of `01:00:00` == 1 hour.

If you need more you must submit a regular job to c23ms.

And devel has no gpus. Only cpus.

## GPU STUFF
To use a GPU you need to specify the partition `c23g` and the number of GPUs you want to use like so:
```bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:<NUMBER>
```
Number may be any integer from 1 to 4.

But remember, using GPU is highly expensive in terms of time quota. 

And if you want to use GPU, remember to set the cpu to a lower count, as not that many may be needed, just for shoveling around data. How many you need idk. At least 4 per GPU, better more.

## More Info
If you have submitted you job, whether test or real, check for it via:
```bash
squeue --me
```

If you want an estimate of the time your job will be executed, use:
```bash
squeue --me --start
```

If you want to see the output of your job, use:
```bash
tail -f <YOUR_LOG_FILE>
```

The log file will be called like the number of the job. You can edit the name of the log file in the job script.
But let the `%j` in there, as it will be replaced by the job number.

## EVEN MORE INFO
Just google yourself. There are many more options and stuff you can do with sbatch.