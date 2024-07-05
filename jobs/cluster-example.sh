#!/bin/bash -x

#SBATCH --job-name="<NAME>"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 #choose as low as possible, but if you want to run on cpu: 96
#SBATCH --partition=c23g #c23g for gpu, c23ms for cpu, devel for test. on devel you can only run cpu for 1h, but free of charge, no tracked time
#SBATCH --gres=gpu:1 #only put in here if the partition has a gpu
#SBATCH --time=24:00:00
#SBATCH --error=/rwthfs/rz/cluster/home/<USER>/SOMEWHERE/%j.err.log
#SBATCH --output=/rwthfs/rz/cluster/home/<USER>/SOMEWHERE/%j.log
#SBATCH --mail-user=<MAIL-ADRESS>
#SBATCH --mail-type=ALL #ALL, NONE, BEGIN, END, FAIL



###### EXEC PATH ########
EXEC_PATH="$HOME/group6" #set to your git repo directory
echo "exec_path: $EXEC_PATH"
cd $EXEC_PATH


###### CONDA ######
CONDA_ENV="learngraph" #the name of the conda environment to use

chmod +x ~/group6/scripts/make_sure_conda.sh #change group6 to your directory
bash ~/group6/scripts/make_sure_conda.sh $CONDA_ENV 3.12
if [ $? != 0 ]; then
    echo "Could not validate or create the conda installation and environment!"
    exit 1
fi
export CONDA_ROOT=$HOME/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate $CONDA_ENV
if [ "" = "$(which python | grep "$CONDA_ENV")" ]; then
    echo $(which python)
    echo "Thought the conda env $CONDA_ENV got activated, but was not!"
    echo "Maybe conda is not installed, but somehow is, clean your garbage up and try again."
    exit 2
fi

python -m pip install -r requirements.txt

##### DOWN BELOW THE PYTHON CALL YOU WANT TO MAKE ####
#python src/main.py --dataset foo --model bar