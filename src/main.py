"""The entry point of our graph kernel implementation."""
#internal imports
from decorators import parseargs

#external imports
import importlib
from typing import *
import psutil
import numpy as np
import pickle, os

# for how to parse args, see the docstring for parseargs

@parseargs(
    __description="The entry point of our graph kernel implementation.\nMay be be called this way:\n\tpython src/main.py [--arg value]*", 
     __help=True
)
def main():
    
    print("Nothing here yet. Come back later.")


if __name__ == "__main__":
    main()