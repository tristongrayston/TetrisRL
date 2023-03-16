import numpy as np
import random as rnd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from collections import deque

# HYPERPARAMS
REPLAY_BUFFER_MAX_LENGTH = 100000
BUFFER_BATCH_SIZE = 128
BATCH_SIZE = 64 
GAMMA = 0.99
TARGET_UPDATE = 150
EPSILON_START = 0.99 # Epsilon Start
EPSILON_END = 0.2 # Epsilon End
EPSILON_DECAY = 0.00008 # Epsilon Decay
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "model_params_7.pt"
save_to_path = "model_params_8.pt"


class ReplayBuffer():
    
    def __init__(self):
        self.memory = deque(maxlen = REPLAY_BUFFER_MAX_LENGTH)

    def store_memory(self, experience: tuple) -> None:
        self.memory.append(experience)

    def collect_memory(self) -> tuple:
        if np.random.rand() < 0.8:
            return self.memory.pop()
        else:
            return rnd.choice(self.memory)
    
    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent():
    def __init__(self, input_dims, output_dims):
        self.output_dims = output_dims
        self.input_dims = input_dims
        #self.observation_space = observation_space

        # actor and target models:
        self.model = self.create_model(self.input_dims, self.output_dims)
        self.target_model = self.create_model(self.input_dims, self.output_dims)
        self.replay_memory = ReplayBuffer()
        self.update_target_counter = 0
        self.total_actions_taken = 0
        self.actions_taken = 0


        self.optimizer = torch.optim.Adam(self.model.parameters(), LR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self, input_dims, output_dims):
        model = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(56, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dims),
        )
        return model

    # Method for predicting an action 
    def get_action(self, state):
        #print(state.shape)
        #state = np.transpose(state, (2, 0, 1))
        #print(state.shape)
        self.actions_taken += 1
        self.total_actions_taken += 1
        EPSILON = EPSILON_END # (EPSILON_START)*np.exp(-EPSILON_DECAY*self.total_actions_taken)

        with torch.no_grad():
            #input = state_array.reshape(1)
            input_state = torch.Tensor(state).to(device)
            #input_state = input_state.permute(2, 0, 1)
            q_values = self.target_model.forward(input_state)

        #print(q_values)
        if rnd.random() > EPSILON_START - EPSILON:
            # Pick a random action
            action = torch.tensor([rnd.randrange(0, 4)])
        else:
            action = torch.argmax(q_values)
        #print(action)
        return action.item()
    
    def learn(self):
        ''' We pretty much strictly learn from the memory, not 
            specifically from the current experience. '''
        
        # We just pass through the learn function if the batch size has not been reached. 
        if self.replay_memory.__len__() < BUFFER_BATCH_SIZE:
            return

        # DQN uses MSE and stochastic gradient descent
        criterion = nn.SmoothL1Loss()

        state = []
        action = []
        reward = []
        next_state = []
        for _ in range(BATCH_SIZE):
            s, a, r, n = self.replay_memory.collect_memory()

            state.append(torch.tensor(s, dtype=torch.float32).to(device))
            action.append(a)
            reward.append(r)
            next_state.append(torch.tensor(n, dtype=torch.float32).to(device))



        # TENSORS

        # Convert lists to tensors
        #state_tensors = [torch.tensor(s).flatten() for s in state]

        state_tensors = torch.stack(state)
        new_state_tensors = torch.stack(next_state)
        #action = torch.eye(4)[action].to(device)
        action = torch.tensor(action).to(device)
        #next_state = torch.tensor(next_state).to(device)
        # We normalize our rewards for a more stable learning process
        rewards = torch.tensor(reward, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        #print(rewards)
        #print(action)
        #Find our predictions
        with torch.no_grad():
            target_predictions = self.target_model(new_state_tensors).max(dim=1)[0]
        
        model_predictions = self.model(state_tensors).max(dim=1)[0]*action # .max(dim=1)[action]
        
        # Calculate our target
        target = rewards + GAMMA*target_predictions
        
        # Calculate MSE Loss
        loss = criterion(model_predictions, target)
        
        #print(loss)
        
        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        # print(len(self.replay_memory.memory))
        self.optimizer.step()
        #print("learning..")
        self.update_target_counter += 1
        #print(self.update_target_counter)
        if self.update_target_counter % TARGET_UPDATE == 0:
            print('updating...')
            self.target_model.load_state_dict(self.model.state_dict())
            print('Current Epsilon, ', (EPSILON_START - (EPSILON_START)*np.exp(-EPSILON_DECAY*self.total_actions_taken)))

    def save(self):
        torch.save(self.target_model.state_dict(), save_to_path)

    def load(self):
        torch.load(path)

# if __name__ == "__main__":
#     agent = DQNAgent((1, 10, 20), 4)
#     agent.get_action(np.zeros(10, 20))
    