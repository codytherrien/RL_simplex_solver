Running project:
To run the project the user will need docker installed. Once docker
is installed follow the steps below

1. Build docker container
To build the container run "docker build -t simplex-agent ."
in the command line.

2. Test LP
Using Bash Shell Run:
docker run -i simplex-agent < <lp input>

Using Windows Powershell Run:
cat <lp input> | docker run -i simplex-agent

Solver Architecture:
This solver uses the standard dictionary based method for solving
LPs. Pivot selection can be done using a naive approach or deep 
reinforcement agent approach (see Extra Features for details).
To solve initially infeasible dictionaries this solver solves 
the auxiliary problem and then proceeds to solve the problem if
the auxiliary problem returns a feasible point.

To prevent cycling the reinforcement learning agent selects the pivot
rule to use at each step. This does not guerentee with 100% success that
the solver will never cycle, but the solver has passed all test cases given.

The naive solver prevents cycling by using blands rule whenever there are any
variables in the solver with a constant value equal to 0. 

Extra Features:
The naive solver impliments largest increase as well was bland's rule
for pivot selection.

The rl_solver uses deep reinforcement learning for pivot selection.
at each set the solver selects an action or pivot. THe solver can 
choose blands rule, dansigs rule, or largest increase for pivot selection.
The rl_solver uses a deep q network to model possible dictionaries. The
model architecture consists of 3 convolutional layers of decreasing size followed
by two fully connected, dense layers, and a final fully connected layer of size 3
with softmax activation for one hot encoding. 

In theory this reinforcement learning agent selects the pivot which will result in
the fewest pivots to solve an LP. This is based on the LPs the model has learned on which
include the test LPs given and an assortment of randomly generated LPs.

NOTE:
The deep CNN used for the reinforcement learning agent is about 400Mb and takes
5-6 seconds to load on the machine tested (which has an ssd). To test the pure performance 
of the solver edit line 13 of the Dockerfile to "COPY run_simplex.py ." and rebuild the
container.