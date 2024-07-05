cd $HOME
#check generally for conda
CPATH="$HOME/miniconda3"
export CONDA_ROOT=$CPATH
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
if [ ! -d "$CPATH" ]; then
    echo "Conda not installed in $CPATH, checking elsewhere."
    CPATH=$(dirname "$(which conda activate)" | grep "conda" | grep "miniconda3" )
fi

if [ "$CPATH" = "" ]; then
    CPATH=$(dirname "$(find -name conda | grep "bin" | grep -v "py" | grep -v "condabin" | grep "miniconda3")" | grep "conda" ) #this should get us only one path
    echo $CPATH
    if [ "$CPATH" = "" ]; then
        #install conda, facepalm
        echo "Installing conda, because could not find it. Please follow the installation process. You will need to scroll through the license agreement or press q, it will be accepted automatically."
        sleep 5
        wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh
        echo -e "\nyes\n" | bash Miniconda3-py311_24.1.2-0-Linux-x86_64.sh
        rm Miniconda3-py311_24.1.2-0-Linux-x86_64.sh*
        #check conda installation #TODO
        CPATH=$(ls $HOME | grep "conda")
        if [ "$CPATH" = "" ]; then
            echo "Something went wrong when installing conda (did not find it in '$HOME'), please try on your own."
            exit 2
        fi
    fi
    export CONDA_ROOT=$HOME/miniconda3
    source $CONDA_ROOT/etc/profile.d/conda.sh
    export PATH="$CONDA_ROOT/bin:$PATH"
fi

#now the conda bin is on path.
conda init