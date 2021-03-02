# Import routines

import numpy as np
import math
import random
from sklearn.preprocessing import OneHotEncoder 

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
        self.state_space = [(loc,time,day) for loc in range(m) for time in range(t) for day in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
#         print(state)
    
        state = np.array(state)

        loc_encoder = OneHotEncoder(sparse=False)
        loc_to_encode = np.array([p for p in range(m)])
        loc_to_encoded = loc_to_encode.reshape(len(loc_to_encode), 1)
        loc_fit = loc_encoder.fit(loc_to_encoded)

        tim_encoder = OneHotEncoder(sparse=False)
        time_to_encode = np.array([r for r in range(t)])
        time_to_encoded = time_to_encode.reshape(len(time_to_encode), 1)
        time_fit = tim_encoder.fit(time_to_encoded)

        day_encoder = OneHotEncoder(sparse=False)
        day_to_encode = np.array([p for p in range(d)])
        day_to_encoded = day_to_encode.reshape(len(day_to_encode), 1)
        day_fit = day_encoder.fit(day_to_encoded)

        out = loc_fit.transform([[state[0]]])
        temp = time_fit.transform([[state[1]]])
        out = np.concatenate((out[0],temp[0])) #first
        temp = day_fit.transform([[state[2]]])
        out = np.concatenate((out,temp[0])) #second

    #     print(out)    
        return out


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
#         print(state)
#         print(action)

        state = np.array(state)
        action = np.array(action)

        loc_encoder = OneHotEncoder(sparse=False)
        loc_to_encode = np.array([p for p in range(m)])
        loc_to_encoded = loc_to_encode.reshape(len(loc_to_encode), 1)
        loc_fit = loc_encoder.fit(loc_to_encoded)

        tim_encoder = OneHotEncoder(sparse=False)
        time_to_encode = np.array([r for r in range(t)])
        time_to_encoded = time_to_encode.reshape(len(time_to_encode), 1)
        time_fit = tim_encoder.fit(time_to_encoded)

        day_encoder = OneHotEncoder(sparse=False)
        day_to_encode = np.array([p for p in range(d)])
        day_to_encoded = day_to_encode.reshape(len(day_to_encode), 1)
        day_fit = day_encoder.fit(day_to_encoded)

        out = loc_fit.transform([[state[0]]])
        temp = time_fit.transform([[state[1]]])
        out = np.concatenate((out[0],temp[0])) #first
        temp = day_fit.transform([[state[2]]])
        out = np.concatenate((out,temp[0])) #second

        temp = loc_fit.transform([[action[0]]])
        out = np.concatenate((out,temp[0])) #third
        temp = loc_fit.transform([[action[1]]])
        out = np.concatenate((out,temp[0])) #fourth

    #     print(out)    
        return out


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        #21 random samples are selected & why 21 that needs to be found
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append((0,0))
        possible_actions_index.append(0)
        
        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        #Calc T1
        t1 = Time_matrix[state[0]][action[0]][state[1]][state[2]]        
        #Calc T2
        tempTime = state[1]+int(t1)
        if tempTime >= 24:
            tempT = tempTime - 24
            tempD = state[2] + 1
            if tempD>=7:
                t2 = Time_matrix[action[0]][action[1]][tempT][tempD-7]
            else:
                t2 = Time_matrix[action[0]][action[1]][tempT][tempD]
        else:
            t2 = Time_matrix[action[0]][action[1]][tempTime][state[2]]
        
        #Calc Reward Accordingly
        if action[0] == 0 and action[1] == 0:
            reward = float(-1 * C)
        else:
            reward = R * t2 - C * (t1 + t2)
        
        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = [0,0,0]
        
        #Calc T1
        t1 = Time_matrix[state[0]][action[0]][state[1]][state[2]]        
        #Calc T2
        tempTime = state[1]+int(t1)
        if tempTime >= 24:
            tempT = tempTime - 24
            tempD = state[2] + 1
            if tempD>=7:
                t2 = Time_matrix[action[0]][action[1]][tempT][tempD-7]
            else:
                t2 = Time_matrix[action[0]][action[1]][tempT][tempD]
        else:
            t2 = Time_matrix[action[0]][action[1]][tempTime][state[2]]
        
        #Update the state and increase the time value by 1hr
        next_state[0] = action[1] #drop loc == current loc
        tempTime = state[1]+int(t1)+int(t2) #curr time
        if tempTime >= 24:
            next_state[1] = tempTime - 24
            tempDay = state[2] + 1
            if tempDay >= 7:
                next_state[2] = tempDay - 7
            else:
                next_state[2] = tempDay
        else:
            next_state[1] = tempTime
            next_state[2] = state[2]
        
        return tuple(next_state)

    
    def reset(self):
        return self.action_space, self.state_space, self.state_init
