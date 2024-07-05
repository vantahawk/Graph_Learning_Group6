# UTILITY TO CONVERT BETWEEN XYZ AND Z-MATRIX GEOMETRIES
# Copyright 2017 Robert A Shaw
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# Usage: python3 gc.py -xyz [file to convert]
# or     python3 gc.py -zmat [file to convert]
# possible flags for zmatrix printing are:
# --rvar/avar/dvar = True/False
# --allvar = True/False (sets all above)

import numpy as np
import argparse
import gcutil
from gcutil import distance_matrix, zmat_as_ndarray
import networkx as nx

def zmat_from_molecule(graph:nx.Graph)->np.ndarray:
    """Converts a molecule graph to a Z-matrix.
    
    Args:
        graph (nx.Graph): The molecule graph to convert.
    
    Returns:
        np.ndarray: The Z-matrix of the molecule. Shape = (n_atoms, 7)

        7 columns are: 

            1. atom-type, 
            2. bond-partner-id,        3. bond-length, 
            4. bond-angle-partner-id,  5. bond-angle, 
            6. dihedral-partner-id,    7. dihedral-angle 
    """
    atomnames = [node for node in graph.nodes(data="node_label")]
    xyzarr = np.array([node for node in graph.nodes(data="node_attributes")])

    distmat = distance_matrix(xyzarr)
    return zmat_as_ndarray(graph, xyzarr, distmat, atomnames)

###BORROWED HEAVILY FROM THE ABOVE, BUT MODIFIED TO USE DIRECTLY IN THE CODEBASE
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-xyz", dest="xyzfile", required=False, type=str, help="File containing xyz coordinates")
    parser.add_argument("-zmat", dest="zmatfile", required=False, type=str, help="File containing Z-matrix")
    parser.add_argument("--rvar", dest="rvar", required=False, type=bool, default=False, help="Print distances as variables")
    parser.add_argument("--avar", dest="avar", required=False, type=bool, default=False, help="Print angles as variables")
    parser.add_argument("--dvar", dest="dvar", required=False, type=bool, default=False, help="Print dihedrals as variables") 
    parser.add_argument("--allvar", dest="allvar", required=False, type=bool, default=False, help="Print all values as variables")
    args = parser.parse_args()

    xyzfilename = args.xyzfile
    zmatfilename = args.zmatfile
    xyz = np.array
    rvar = args.rvar or args.allvar
    avar = args.avar or args.allvar
    dvar = args.dvar or args.allvar

    if (xyzfilename == None and zmatfilename == None):
        print("Please specify an input geometry")

    elif (zmatfilename == None):
        xyzarr, atomnames = gcutil.readxyz(xyzfilename)
        distmat = gcutil.distance_matrix(xyzarr)
        gcutil.write_zmat(xyzarr, distmat, atomnames, rvar=rvar, avar=avar, dvar=dvar)
    else:
        atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist = gcutil.readzmat(zmatfilename)
        gcutil.write_xyz(atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist)
