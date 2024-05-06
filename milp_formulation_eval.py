# Import
# -----------------------------------------------------------------------------
import time
from typing import Any

import pyscipopt

from random import seed
from random import random

import itertools

# Main
# -----------------------------------------------------------------------------
def add_constraint_sharp(m, x, y, I):
    m = add_constraint_simple(m, x, y, I)
    f = [-1, 1]
    for r in itertools.product(f, repeat = len(I)):
        m.addCons(
            pyscipopt.quicksum(r[i] * x[i] for i in I) <= 1,
            name=f"Constraint_sharp",
        )

    return m

def add_constraint_simple(m, x, y, I):
    # Constraints
    for i in I:
        for j in I:
            if i != j:
                m.addCons(
                    x[i] <= 1 -y[j],
                    name=f"Constraint_le_{i}_{j}",
                )
                m.addCons(
                    x[i] >= y[j] - 1,
                    name=f"Constraint_ge_{i}_{j}",
                )

    m.addCons(
        pyscipopt.quicksum(y[j] for j in I) == 1,
        name=f"Constraint_eq",
    )
    return m

def add_constraint_ideal(m, x, y, I):
    # Constraints
    for i in I:
        m.addCons(
            x[i] <= y[i],
            name=f"Constraint_le_{i}",
        )
        m.addCons(
            x[i] >= -y[i],
            name=f"Constraint_ge_{i}",
        )

    m.addCons(
        pyscipopt.quicksum(y[j] for j in I) == 1,
        name=f"Constraint_eq",
    )
    return m

def solve_problem(type, size):
    # index list
    I: list[int] = range(size)

    c: list[int] = []
    seed(1)
    max_val = -10
    max_idx = -1
    for i in I:
        c.append(5 - 10 * random())
        if max_val < c[i]:
            max_idx = i
            max_val = c[i]

    print(f"maximum = {max_val}, index = {max_idx}")

    m: pyscipopt.Model = pyscipopt.Model(
        problemName = type,
    )
    # variables
    x: dict[tuple[int], Any] = {}  
    y: dict[tuple[int], Any] = {}  

    for i in I:
        x[i] = m.addVar(
            name=f"Var_x({i})",
            vtype="C",
        )
    for i in I:
        y[i] = m.addVar(
            name=f"Var_y({i})",
            vtype="B",
        )

    # Objective Value
    m.setObjective(
        pyscipopt.quicksum(c[i] * x[i] for i in I),
        sense="maximize",
    )

    if type == "simple":
        m = add_constraint_simple(m, x, y, I)
    elif type == "ideal":
        m = add_constraint_ideal(m, x, y, I)
    elif type == "sharp":
        m = add_constraint_sharp(m, x, y, I)
    elif type == "none":
        start_time = time.time()
        max_val = -10
        max_idx = -1
        for i in I:
            if max_val < c[i]:
                max_idx = i
                max_val = c[i]
        print("Optimization Time = %s seconds ---" % (time.time() - start_time))

    if type != "none":
        time_start: float = time.perf_counter()
        m.optimize()
        time_stop: float = time.perf_counter()

        print(f"Status = {m.getStatus()}, ", end="")
        print(f"Objective Value = {m.getObjVal()}, ", end="")
        print(f"Optimization Time = {time_stop - time_start:.3f} (sec)")
        print(f"Solution x[i]: ")
        for i in I:
            if m.getVal(x[i]) != 0:
                print(f"{x[i]} = {m.getVal(x[i])},  ", end="")
        print(f"")
        return m

solve_problem("simple", 1000)
solve_problem("ideal", 1000)
solve_problem("sharp", 17)
solve_problem("none", 1000000)
