### This code was inspired by Phil Tabob's Youtube Video on Deep Q Learning
### Using Keras https://www.youtube.com/watch?v=5fHngyN8Qhw
### And the code on Phil's Github https://github.com/philtabor
### https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_keras.py
### With a large portion of this script directly from the repo above

from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape=(500,500), n_actions=3):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape[0], input_shape[1]))
        self.new_state_memory = np.zeros((self.mem_size, input_shape[0], input_shape[1]))
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_trasition(self, state, action, reward, state_, done, shape):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state.reshape(shape[0],shape[1])
        self.new_state_memory[index] = state_.reshape(shape[0],shape[1])
        # store 1hot encoding
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    # Building Deep CNN
    model = Sequential()
    model.add(Conv2D(
        128, 
        kernel_size=4, 
        activation='relu',
        input_shape=(input_dims[0], input_dims[1], 1)
    ))
    model.add(Conv2D(
        64, 
        kernel_size=4, 
        activation='relu'
    ))
    model.add(Conv2D(
        16, 
        kernel_size=4, 
        activation='relu'
    ))
    model.add(Flatten())
    model.add(Dense(fc1_dims, activation='relu'))
    model.add(Dense(fc2_dims, activation='relu'))
    model.add(Dense(n_actions, activation=softmax))

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

class SimplexAgent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.9999, epsilon_min=0.0001,
                 mem_size=100, file_name='simplex_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = file_name
        self.input_dims = input_dims
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        # Stores state for future learning
        self.memory.store_trasition(state, action, reward, new_state, done, self.input_dims)



    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            # Exploration action
            action = np.random.choice(self.action_space)
        else:
            # Exploitation action (Uses model)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        
        return action

    def learn(self):
        # This is where the meat of the Deep Reinforcement Learning takes place.
        # This method is not called during testing for performance reasons
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)
            state = state.reshape(self.batch_size,self.input_dims[0],self.input_dims[1],1)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)
            q_eval = self.q_eval.predict(state)
            q_next = self.q_eval.predict(new_state.reshape(self.batch_size,self.input_dims[0],self.input_dims[1],1))
            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = reward + \
                self.gamma*np.max(q_next, axis=1)*done
            
            _ = self.q_eval.fit(state, q_target)

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon*self.epsilon_dec
            
    def save_model(self):
        self.q_eval.save(self.model_file)
        #print('Model saved to {self.model_file}')

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        #print('Model loaded from {self.model_file}')