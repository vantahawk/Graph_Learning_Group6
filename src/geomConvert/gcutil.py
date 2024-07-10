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
# Utilities for gc.py

import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from math import isnan
# import sys
# sys.setrecursionlimit(5000)
def replace_vars(vlist, variables):
    """ Replaces a list of variable names (vlist) with their values
        from a dictionary (variables).
    """
    for i, v in enumerate(vlist):
        if v in variables:
            vlist[i] = variables[v]
        else:
            try:
                # assume the "variable" is a number
                vlist[i] = float(v)
            except:
                print("Problem with entry " + str(v))

def readxyz(filename):
    """ Reads in a .xyz file in the standard format,
        returning xyz coordinates as a numpy array
        and a list of atom names.
    """
    xyzf = open(filename, 'r')
    xyzarr = np.zeros([1, 3])
    atomnames = []
    if not xyzf.closed:
        # Read the first line to get the number of particles
        npart = int(xyzf.readline())
        # and next for title card
        title = xyzf.readline()

        # Make an N x 3 matrix of coordinates
        xyzarr = np.zeros([npart, 3])
        i = 0
        for line in xyzf:
            words = line.split()
            if (len(words) > 3):
                atomnames.append(words[0])
                xyzarr[i][0] = float(words[1])
                xyzarr[i][1] = float(words[2])
                xyzarr[i][2] = float(words[3])
                i = i + 1
    return (xyzarr, atomnames)

def readzmat(filename):
    """ Reads in a z-matrix in standard format,
        returning a list of atoms and coordinates.
    """
    zmatf = open(filename, 'r')
    atomnames = []
    rconnect = []  # bond connectivity
    rlist = []     # list of bond length values
    aconnect = []  # angle connectivity
    alist = []     # list of bond angle values
    dconnect = []  # dihedral connectivity
    dlist = []     # list of dihedral values
    variables = {} # dictionary of named variables
    
    if not zmatf.closed:
        for line in zmatf:
            words = line.split()
            eqwords = line.split('=')
            
            if len(eqwords) > 1:
                # named variable found 
                varname = str(eqwords[0]).strip()
                try:
                    varval  = float(eqwords[1])
                    variables[varname] = varval
                except:
                    print("Invalid variable definition: " + line)
            
            else:
                # no variable, just a number
                # valid line has form
                # atomname index1 bond_length index2 bond_angle index3 dihedral
                if len(words) > 0:
                    atomnames.append(words[0])
                if len(words) > 1:
                    rconnect.append(int(words[1]))
                if len(words) > 2:
                    rlist.append(words[2])
                if len(words) > 3:
                    aconnect.append(int(words[3]))
                if len(words) > 4:
                    alist.append(words[4])
                if len(words) > 5:
                    dconnect.append(int(words[5]))
                if len(words) > 6:
                    dlist.append(words[6])
    
    # replace named variables with their values
    replace_vars(rlist, variables)
    replace_vars(alist, variables)
    replace_vars(dlist, variables)
    
    return (atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist) 

def distance_matrix(xyzarr):
    """Returns the pairwise distance matrix between atom
       from a set of xyz coordinates 
    """
    return cdist(xyzarr, xyzarr)

def angle(xyzarr, i, j, k):
    """Return the bond angle in degrees between three atoms 
       with indices i, j, k given a set of xyz coordinates.
       atom j is the central atom
    """
    rij = xyzarr[i] - xyzarr[j]
    rkj = xyzarr[k] - xyzarr[j]
    cos_theta = np.dot(rij, rkj)
    sin_theta = np.linalg.norm(np.cross(rij, rkj))
    theta = np.arctan2(sin_theta, cos_theta)
    theta = 180.0 * theta / np.pi 
    return theta

def dihedral(xyzarr, i, j, k, l):
    """Return the dihedral angle in degrees between four atoms 
       with indices i, j, k, l given a set of xyz coordinates.
       connectivity is i->j->k->l
    """
    rji = xyzarr[j] - xyzarr[i]
    rkj = xyzarr[k] - xyzarr[j]
    rlk = xyzarr[l] - xyzarr[k]
    v1 = np.cross(rji, rkj)
    v1 = v1 / np.linalg.norm(v1) if sum(v1) != 0 else np.zeros(3)
    assert not np.isnan(v1).any(), f"v1 is None or NaN: {v1}. cross(rji, rkj) = cross({rji}, {rkj})= {np.cross(rji, rkj)}"
    v2 = np.cross(rlk, rkj)
    v2 = v2 / np.linalg.norm(v2) if sum(v2) != 0 else np.zeros(3)
    assert not np.isnan(v2).any(), f"v2 is None or NaN: {v1}"
    m1 = np.cross(v1, rkj) / np.linalg.norm(rkj) if sum(rkj) != 0 else np.zeros(3)
    assert not np.isnan(m1).any(), f"m1 is None or NaN: {m1}"
    x = np.dot(v1, v2)
    y = np.dot(m1, v2)
    chi = np.arctan2(y, x)
    assert not np.isnan(chi), f"chi is None or NaN: {chi}"
    chi = -180.0 - 180.0 * chi / np.pi
    if (chi < -180.0):
        chi = chi + 360.0
    return chi

def write_zmat(xyzarr, distmat, atomnames, rvar=False, avar=False, dvar=False):
    """Prints a z-matrix from xyz coordinates, distances, and atomnames,
       optionally with the coordinate values replaced with variables.
    """
    npart, ncoord = xyzarr.shape
    rlist = [] # list of bond lengths
    alist = [] # list of bond angles (degrees)
    dlist = [] # list of dihedral angles (degrees)
    if npart > 0:
        # Write the first atom
        print(atomnames[0])
        
        if npart > 1:
            # and the second, with distance from first
            n = atomnames[1]
            rlist.append(distmat[0][1])
            if (rvar):
                r = 'R1'
            else:
                r = '{:>11.5f}'.format(rlist[0])
            print('{:<3s} {:>4d}  {:11s}'.format(n, 1, r))
            
            if npart > 2:
                n = atomnames[2]
                
                rlist.append(distmat[0][2])
                if (rvar):
                    r = 'R2'
                else:
                    r = '{:>11.5f}'.format(rlist[1])
                
                alist.append(angle(xyzarr, 2, 0, 1))
                if (avar):
                    t = 'A1'
                else:
                    t = '{:>11.5f}'.format(alist[0])

                print('{:<3s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, 1, r, 2, t))
                
                if npart > 3:
                    for i in range(3, npart):
                        n = atomnames[i]

                        rlist.append(distmat[i-3][i])
                        if (rvar):
                            r = 'R{:<4d}'.format(i)
                        else:
                            r = '{:>11.5f}'.format(rlist[i-1])

                        alist.append(angle(xyzarr, i, i-3, i-2))
                        if (avar):
                            t = 'A{:<4d}'.format(i-1)
                        else:
                            t = '{:>11.5f}'.format(alist[i-2])
                        
                        dlist.append(dihedral(xyzarr, i, i-3, i-2, i-1))
                        if (dvar):
                            d = 'D{:<4d}'.format(i-2)
                        else:
                            d = '{:>11.5f}'.format(dlist[i-3])
                        print('{:3s} {:>4d}  {:11s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, i-2, r, i-1, t, i, d))
    if (rvar):
        print(" ")
        for i in range(npart-1):
            print('R{:<4d} = {:>11.5f}'.format(i+1, rlist[i]))
    if (avar):
        print(" ")
        for i in range(npart-2):
            print('A{:<4d} = {:>11.5f}'.format(i+1, alist[i]))
    if (dvar):
        print(" ")
        for i in range(npart-3):
            print('D{:<4d} = {:>11.5f}'.format(i+1, dlist[i]))

def zmat_as_ndarray(graph:nx.Graph, xyzarr:np.ndarray, distmat:np.ndarray, atomnames:list[int]):
    """Returns a numpy array of the z-matrix from xyz coordinates
       and a distance matrix.

       Expects a connected graph, with number of nodes == maximum node index
    """
    #build tree of the graph
    class Node:
        def __init__(self, id):
            self.id = id
            self.children = []
        def add_child(self, child):
            self.children.append(child)
        def __str__(self):
            return str(self.id) + " -> " + str(self.children)
        def __repr__(self):
            return str(self.id) + ": {"+ ", ".join([c.__repr__() for c in self.children]) + "}"

    def writeToZmat(n:int, parent:int, zmat:np.ndarray, xyzarr:np.ndarray, distmat:np.ndarray, atomnames:list[int], count):
        if count == 0:
            # Write the first atom
            zmat[n][0] = atomnames[n]
            
        if count == 1:
            # and the second, with distance from first
            zmat[n][0] = atomnames[n]
            zmat[n][1] = parent
            zmat[n][2] = distmat[n][parent]
            
        if count == 2:
            # and the third, with distance from first and angle from first two
            zmat[n][0] = atomnames[n]
            zmat[n][1] = parent
            zmat[n][2] = distmat[n][parent]
            zmat[n][3] = zmat[parent][1]
            # assert isinstance(zmat[n][3], (int, np.int64)), f"zmat[{n}][3] is not an int: {zmat[n][3]} - type: {type(zmat[n][3])}"
            zmat[n][4] = angle(xyzarr, zmat[n][3], n, parent)
            
        if count >= 3:
            # and the rest, with distances and angles from previous atoms
            zmat[n][0] = atomnames[n]
            zmat[n][1] = parent
            zmat[n][2] = distmat[n][parent]
            zmat[n][3] = zmat[parent][1]
            # assert isinstance(zmat[n][3], int), f"zmat[{n}][3] is not an int: {zmat[n][3]} - type: {type(zmat[n][3])}"
            zmat[n][4] = angle(xyzarr, n, zmat[n][3], parent)
            zmat[n][5] = zmat[parent][3]
            # assert isinstance(zmat[n][5], int), f"zmat[{n}][5] is not an int: {zmat[n][5]} - type: {type(zmat[n][5])}"
            dihed = np.float64(dihedral(xyzarr, n, parent, zmat[n][3], zmat[n][5]))
            assert dihed is not None and not isnan(dihed), f"dihed is None or NaN: {dihed}"
            zmat[n][6] = dihed

        #each row is now:
        # ATOM, PARENT, DISTANCE, GRAND-PARENT, ANGLE, GREAT-GRAND-PARENT, DIHEDRAL

    #with dfs
    visited = np.zeros(len(atomnames))
    def build_tree(graph:nx.Graph, node:Node, parent:Node, xyzarr:np.ndarray, distmat:np.ndarray, atomnames:list[int], zmat:np.ndarray):
        
        writeToZmat(node.id, parent.id, zmat, xyzarr, distmat, atomnames, sum(visited))
        visited[node.id] = 1

        for child in graph.neighbors(node.id):
            if visited[child] == 0:
                child_node = Node(child)
                node.add_child(child_node)
                build_tree(graph, child_node, node,
                            xyzarr, distmat, atomnames, zmat)

    tree_root = Node(0)
    zmat = np.zeros([graph.number_of_nodes(), 7], dtype=int)
    build_tree(graph, tree_root, Node(-1), xyzarr, distmat, atomnames, zmat)
    
    return zmat

def write_xyz(atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist):
    """Prints out an xyz file from a decomposed z-matrix"""
    npart = len(atomnames)
    print(npart)
    print('INSERT TITLE CARD HERE')
    
    # put the first atom at the origin
    xyzarr = np.zeros([npart, 3])
    if (npart > 1):
        # second atom at [r01, 0, 0]
        xyzarr[1] = [rlist[0], 0.0, 0.0]

    if (npart > 2):
        # third atom in the xy-plane
        # such that the angle a012 is correct 
        i = rconnect[1] - 1
        j = aconnect[0] - 1
        r = rlist[1]
        theta = alist[0] * np.pi / 180.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a_i = xyzarr[i]
        b_ij = xyzarr[j] - xyzarr[i]
        if (b_ij[0] < 0):
            x = a_i[0] - x
            y = a_i[1] - y
        else:
            x = a_i[0] + x
            y = a_i[1] + y
        xyzarr[2] = [x, y, 0.0]

    for n in range(3, npart):
        # back-compute the xyz coordinates
        # from the positions of the last three atoms
        r = rlist[n-1]
        theta = alist[n-2] * np.pi / 180.0
        phi = dlist[n-3] * np.pi / 180.0
        
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)

        x = r * cosTheta
        y = r * cosPhi * sinTheta
        z = r * sinPhi * sinTheta
        
        i = rconnect[n-1] - 1
        j = aconnect[n-2] - 1
        k = dconnect[n-3] - 1
        a = xyzarr[k]
        b = xyzarr[j]
        c = xyzarr[i]
        
        ab = b - a
        bc = c - b
        bc = bc / np.linalg.norm(bc)
        nv = np.cross(ab, bc)
        nv = nv / np.linalg.norm(nv)
        ncbc = np.cross(nv, bc)
        
        new_x = c[0] - bc[0] * x + ncbc[0] * y + nv[0] * z
        new_y = c[1] - bc[1] * x + ncbc[1] * y + nv[1] * z
        new_z = c[2] - bc[2] * x + ncbc[2] * y + nv[2] * z
        xyzarr[n] = [new_x, new_y, new_z]
            
    # print results
    for i in range(npart):
        print('{:<4s}\t{:>11.5f}\t{:>11.5f}\t{:>11.5f}'.format(atomnames[i], xyzarr[i][0], xyzarr[i][1], xyzarr[i][2]))
