# Import
# -----------------------------------------------------------------------------
import time

import numpy as np
from typing import Any
import sys

import pyscipopt
from pyscipopt import Model, quicksum, Expr
from random import seed
from random import random

def solve_problem(nnode, threshold, vartype):
    nodes = []

    # The list of Nodes.
    for i in range(nnode):
        nodes.append(int(i))

    # The list of Edges.
    adjacency = [[0 for i in range(nnode)] for j in range(nnode)]
    nedge = 0
    edges = []
    for i in range(nnode):
        for j in range(i + 1, nnode):
            if random() <= threshold:
                edge = set([])
                adjacency[i][j] = adjacency[j][i] = True
                edge.add(i)
                edge.add(j)
                nedge += 1
                edges.append(edge)
            else:
                adjacency[i][j] = adjacency[j][i] = False

    print(f"The number of edges = {nedge}")
    #print(edges)

    # Declare the Model.
    m: pyscipopt.Model = pyscipopt.Model(
        problemName = "Graph-Matching",
    )

    """"
    if(vartype == "C"):
         m.setParam("propagating/maxrounds", 0)
         m.setParam("propagating/maxroundsroot", 0)
    """
   

    # Variables and Weights.
    seed(1)
    x: dict[tuple[int, int], Any] = {}
    w = []
    for i in range(nedge):
        if (vartype == "B"):
            x[i] = m.addVar(
                name = f"Var_x({i})",
                vtype = "B"
            )
        else:
            x[i] = m.addVar(
                name = f"Var_x({i})",
                vtype = "C",
                lb = 0,
                ub = 1
            )
        w.append(random())
    
    # print(w)

    w = np.array(w)
    # Objective Value
    time_start: float = time.perf_counter()
    m.setObjective(
        pyscipopt.quicksum(w[i] * x[i] for i in range(nedge)),
        sense="maximize",
    )
    # m.setIntParam("misc/usesymmetry", 0)

    # Constraints.
    print(nnode, nedge)
    constr = [m.addCons(Expr() <= 1) for nodeno in range(nnode)]
    for (cons, nodeno) in zip(constr, range(nnode)):
        for edgeno in range(nedge):
            if nodeno in edges[edgeno]:
                m.addConsCoeff(cons, x[edgeno], 1)

    time_stop: float = time.perf_counter()
    print(f"Construction Time = {time_stop - time_start:.3f} (sec)")
    
    time_start: float = time.perf_counter()
    m.optimize()
    time_stop: float = time.perf_counter()

    print(f"Optimal Value = {m.getStatus()}, ", end="")
    print(f"Objective Value = {m.getObjVal()}, ", end="")
    print(f"Elapsed Time = {time_stop - time_start:.3f} (sec)")
    print(f"Solution: ")
    #for i in range(nedge):
        #print(m.getVal(x[i]))
        #if m.getVal(x[i]) > 0:
        #    print(edges[i])

nnode = int(sys.argv[1])
threshold = float(sys.argv[2])
# solve_problem(nnode, threshold, "C")
solve_problem(nnode, threshold, "B")

