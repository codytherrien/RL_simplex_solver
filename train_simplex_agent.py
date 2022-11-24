### This code was inspired by Phil Tabob's Youtube Video on Deep Q Learning
### Using Keras https://www.youtube.com/watch?v=5fHngyN8Qhw
### And the code on Phil's Github https://github.com/philtabor
### https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/main_keras_dqn_lunar_lander.py

# This script is only used for training the model and is currently set up to 
# train the model on the test cases given.
# There is no need to use this script during testing
# as training stage causes the solver to solve very slowly

from Simplex import Simplex
from simplex_q_network import SimplexAgent
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def process_state(input_dims, state):
    arr = np.zeros((input_dims[0], input_dims[1]))
    arr[:state.shape[0], :state.shape[1]] = state
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(arr)
    arr = arr.reshape(1,input_dims[0], input_dims[1],1)
    return arr

def naive_solve(lp_solver):
    # Used if input LP is too big for RL model
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
            print(f'Using pivot {pivot}')
            observation_, reward = lp_solver.rl_pivot(pivot)
            observation_ = process_state(agent.input_dims, observation_)
            agent.remember(observation, pivot, reward, observation_, int(done))
            observation = observation_
            agent.learn()
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
        agent.learn()
    lp_solver.get_results()


def main():
    lr = 0.01
    n_lps = 10
    agent = SimplexAgent(
        gamma=0.99,
        epsilon=0.8,
        alpha=lr,
        input_dims=(100, 100),
        n_actions=3,
        mem_size=100,
        batch_size=12,
        epsilon_min=0.1
    )

    try:
        agent.load_model()
    except:
        pass

    lp_list = [

    ]

    for i in range(n_lps):
        counter = 0
        for lp in os.listdir('./test_LPs/input/'):
            counter += 1
            lp_solver = Simplex(f'./test_LPs/input/{lp}')
            print(lp)
            print(f'LP number: {counter}')
            print(len(lp_solver.df))
            print(len(lp_solver.df.columns))
            print(min(agent.input_dims))
            if len(lp_solver.df) >= min(agent.input_dims) or len(lp_solver.df.columns) >= min(agent.input_dims):
                naive_solve(lp_solver)
            
            else:
                rl_solve(lp_solver, agent)

        agent.save_model()



if __name__ == '__main__':
    main()