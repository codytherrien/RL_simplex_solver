# Naive simplex solver
# Can be used in place of RL solver to 
# Test the pure performance of the solver

import sys
import Simplex

def main():
    file = sys.stdin
    lp_solver = Simplex.Simplex(file=file)
    if min(lp_solver.df.iloc[:,0]) < 0:
        lp_solver.auxiliary_setup()

        while lp_solver.solution is None:
            lp_solver.naive_pivot()
        lp_solver.check_auxiliary()

    while lp_solver.solution is None:
        lp_solver.naive_pivot()
    lp_solver.get_results()

if __name__ == "__main__":
    main()