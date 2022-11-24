### This code was inspired by Phil Tabob's Youtube Video on Deep Q Learning
### Using Keras https://www.youtube.com/watch?v=5fHngyN8Qhw
### And the code on Phil's Github https://github.com/philtabor
### https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/main_keras_dqn_lunar_lander.py

# Simplex Agent used for testing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from Simplex import Simplex
from simplex_q_network import SimplexAgent
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

def process_state(input_dims, state):
    arr = np.zeros((input_dims[0], input_dims[1]))
    arr[:state.shape[0], :state.shape[1]] = state
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(arr)
    arr = arr.reshape(1,input_dims[0], input_dims[1],1)
    return arr

def naive_solve(lp_solver):
    # Used if lp is too big for RL model
    if min(lp_solver.df.iloc[:,0]) < 0:
        lp_solver.auxiliary_setup()
        while lp_solver.solution is None:
            lp_solver.naive_pivot()
        lp_solver.check_auxiliary()

    while lp_solver.solution is None:
        lp_solver.naive_pivot()
    lp_solver.get_results()

def rl_solve(lp_solver, agent):
    done = False
    if min(lp_solver.df.iloc[1:,0]) < 0:
        lp_solver.auxiliary_setup()

        observation = lp_solver.get_observation()
        observation = process_state(agent.input_dims, observation)

        while lp_solver.solution is None:
            pivot = agent.choose_action(observation)
            observation_, reward = lp_solver.rl_pivot(pivot)
            observation_ = process_state(agent.input_dims, observation_)
            agent.remember(observation, pivot, reward, observation_, int(done))
            observation = observation_
            #agent.learn() # No need for learning during testing
        lp_solver.check_auxiliary()
            
    else:
        observation = lp_solver.get_observation()
        observation = process_state(agent.input_dims, observation)

    while lp_solver.solution is None:
        pivot = agent.choose_action(observation)
        observation_, reward = lp_solver.rl_pivot(pivot)
        observation_ = process_state(agent.input_dims, observation_)
        agent.remember(observation, pivot, reward, observation_, int(done))
        observation = observation_
        #agent.learn() # No need for learning during testing
    lp_solver.get_results()

def main():
    lr = 0.01
    n_lps = 10
    agent = SimplexAgent(
        gamma=0.99,
        epsilon=0.1,
        alpha=lr,
        input_dims=(100, 100),
        n_actions=3,
        mem_size=100,
        batch_size=12,
        epsilon_min=0.1
    )
    
    file = sys.stdin
    lp_solver = Simplex(file=file)
    if len(lp_solver.df) >= min(agent.input_dims) or len(lp_solver.df.columns) >= min(agent.input_dims):
        naive_solve(lp_solver)
            
    else:
        try:
            agent.load_model()
        except:
            pass
        rl_solve(lp_solver, agent)




if __name__ == '__main__':
    main()