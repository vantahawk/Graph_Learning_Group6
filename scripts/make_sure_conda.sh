#make sure conda is installed, if not install, will also call conda init
EXEC_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
chmod +x $EXEC_PATH/conda_install.sh
bash $EXEC_PATH/conda_install.sh
if [[ $? != 0 ]]; then
    echo "Something during the installation went wrong"
    exit 1
fi
export CONDA_ROOT=$HOME/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda config --set auto_activate_base false
#absolete because reload
# #run the rest in a new shell as conda needs a shell reload
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# chmod +x $SCRIPT_DIR/make_sure_conda_env.sh
# bash $SCRIPT_DIR/make_sure_conda_env.sh
if [ "$1"!="" ];then
ENV_NAME=$1

if [ "" = "$(conda info --envs | grep "$ENV_NAME")" ]; then
    echo -e "y\n" | conda create --name $ENV_NAME python=$2 #asks for y/n
fi

#make sure to deactivate all the venvs, may be multiple idk
while [ $(expr "$(which python | grep "venv")" != "") = 1 ]; do
    deactivate
done

sleep 5
conda activate $ENV_NAME

#make sure python is in this directory
if [ "" = "$(which python | grep "$ENV_NAME")" ]; then
    echo $(which python)
    echo "Thought the conda env $ENV_NAME got activated, but was not!"
    echo "Maybe conda is not installed, but somehow is, clean your garbage up and try again."
    exit 2
fi
if [ "" = "$(which pip | grep "$ENV_NAME")" ]; then
    conda install pip
fi
conda config --set auto_activate_base false
conda deactivate

echo "Successfully validated Conda environment '$ENV_NAME'!"
fi
echo "Successfully validated Conda Installation, good to go."
exit 0
