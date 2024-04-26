from kernels import KernelName
from decorators import parseargs

@parseargs(
    kernel={
        "default":"closed_walk", 
        "type":str, 
        "help":"The kernel to use, one of: \"closed_walk\", \"graphlet\", \"wl\""
    }, 
    __description="The entry point of our graph kernel implementation.\nMay be be called this way:\n\tpython src/main.py [--arg value]*", 
     __help=True
)
def main(kernel:KernelName):
    pass





if __name__ == "__main__":
    main()