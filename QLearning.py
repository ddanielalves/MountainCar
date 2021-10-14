# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:30:25 2020

@author: alves
"""
import numpy as np

class qLearning:
    def __init__(self, env, convert_state_func, qtable_size = None, nr_states = 30):
        self.env = env
        self.convert_state_func = convert_state_func
        self.qtable_size = qtable_size
        self.nr_states = nr_states
        self.stats = {
                "reward": {
                        "min":[],
                        "max":[],
                        "avg":[]
                        }, 
                "episode":[]
                      }

        
    def fit(self, num_episodes, learning_rate = 0.1, discount = 0.95, epsilon = 0.1, 
            stats_every = 0.01, save_every = 0.5, render_every = 0.1): 
        
        if stats_every > 1:
            stats_every = num_episodes//stats_every
        else:
            stats_every = int(1/stats_every)
        
        if save_every > 1:
            save_every = num_episodes//save_every
        else:
            save_every = int(1/save_every)
            
        if render_every > 1:
            render_every = num_episodes//render_every
        else:
            render_every = int(1/render_every)
            
        stats_eps = np.linspace(0, num_episodes, stats_every + 1, dtype=int)
        save_eps = np.linspace(0, num_episodes, save_every + 1, dtype=int)
        render_eps = np.linspace(0, num_episodes, render_every + 1, dtype=int)
 
           
        if self.qtable_size == None:
                 q_table = np.random.uniform(size=self.qtable_size)
        else:
            q_table = np.random.uniform(size = ([self.nr_states]*self.env.observation_space.shape[0] + [self.env.action_space.n]))
        
        # Action value function 
        # A nested dictionary that maps 
        # state -> (action -> action-value). 
        rewards_list = []
        for episode in range(num_episodes):
            
            state_converted = self.convert_state_func(self.env.reset())
            done = False
            total_reward = 0
            
            while not done:
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(q_table[state_converted])
                else:
                    # Get random action
                    action = np.random.randint(0, self.env.action_space.n)
                
                next_state, reward, done, _ = self.env.step(action)
                # Keeps track of useful statistics 
                total_reward+= reward
                next_state_converted = self.convert_state_func(next_state)
                
                action = np.argmax(q_table[state_converted])
                
                
                if not done:        
                    # Maximum possible Q value in next step (for new state)
                    next_q_val = np.max(q_table[next_state_converted])
                
                    # Current Q value (for current state and performed action)
                    current_q = q_table[state_converted + (action,)]
                
                    # And here's our equation for a new Q value for current state and action
                    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * next_q_val)
                    
                    # Update Q table with new Q value
                    q_table[state_converted + (action,)] = new_q
                    
                if episode in render_eps:
                    
                    self.env.render()

                state_converted = next_state_converted
            
            if episode in save_eps:
                pass
            
            if episode in render_eps:
                    print("Episode: {}\nCurrent Reward: {}\n".format(episode, total_reward))
            
            rewards_list.append(total_reward)
            if episode in stats_eps:
                self.stats["reward"]["avg"].append(np.mean(rewards_list))
                self.stats["reward"]["max"].append(np.max(rewards_list))
                self.stats["reward"]["min"].append(np.min(rewards_list))
                self.stats["episode"].append(episode)
                rewards_list = []
        return q_table