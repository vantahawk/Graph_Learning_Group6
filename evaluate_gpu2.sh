#!/bin/bash -x

#SBATCH --job-name="graph-learn-torch"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --error=/rwthfs/rz/cluster/home/sg114224/mspraktikum/logs/%j.err.log
#SBATCH --output=/rwthfs/rz/cluster/home/sg114224/mspraktikum/logs/%j.log
#SBATCH --mail-user=benedict.gerlach@rwth-aachen.de
#SBATCH --mail-type=ALL#
#SBATCH --account=thes1673


###### EXEC PATH ########
#get the path to a very unique file, from that reconstruct the exec_path needed
# cd $HOME
# TESTS_PATH=$(dirname "$(realpath $(find -name test_preprocessing.py) | grep "tests" | grep "mspraktikum")")
#give env name to setup
EXEC_PATH="$HOME/mspraktikum"
echo "exec_path: $EXEC_PATH"
cd $EXEC_PATH


###### CONDA ######
#makes sure conda is installed and the environment sg114224env is there
CONDA_ENV="learngraph"

chmod +x ~/ba/experiments/scripts/make_sure_conda.sh
bash ~/ba/experiments/scripts/make_sure_conda.sh $CONDA_ENV 3.12
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

python src/main.py --l node --d cuda --ds Cora
python src/main.py --l node --d cuda --ds Citeseer