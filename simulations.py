import numpy as np
from ADS_Environment import Environment
import threading
import time

def example_execution(env, policy, render=False, stop=False):
    """
    Simulation of the environment without learning.

    :param env: the environment encoding the (MO)MDP
    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """
    max_timesteps = 200
    number_of_simulations = 1000

    n_steps = 0
    n_peatons_run = 0
    n_bumps_coll = 0

    for i in range(number_of_simulations):
        # Initialize environment
        timesteps = 0
        env.hard_reset(env.initial_agent_left_position, env.initial_pedestrian_1_position, env.initial_pedestrian_2_position)
        state = env.get_state()
        done = False

        #print("State :", state)

        env.set_stats(i + 1, 99, 99, 99, 99)
        if render: # if we want to draw the simulations
            if not env.drawing_paused():
                time.sleep(0.5)
                env.update_window()

        while (timesteps < max_timesteps) and (not done):
            timesteps += 1

            actions = list()
            actions.append(policy[state[0], state[1], state[2]])

            if stop:
                actions = [LEFT, RIGHT, RIGHT]

            state, rewards, dones = env.step(actions)
            #print("State :", state)

            #print(rewards)

            if rewards[2] != 0.0:
                n_peatons_run += 1
                #print("ara!")
                #print(rewards[2])
            if rewards[1] != 0.0: n_bumps_coll += 1
            #print(rewards)

            done = dones[0]  # R Agent does not interfere

            if render:
                if not env.drawing_paused():
                    time.sleep(0.5)
                    env.update_window()
        n_steps += timesteps
        #print(n_steps)

    print()
    print('The results of these ' + str(number_of_simulations) + ' simulations are:')
    print(str(n_steps/number_of_simulations) + ' has been the average steps for the car to reach it\'s destination')
    print(str(n_bumps_coll/number_of_simulations) + ' has been the average incidents with bumps while reaching it\'s destination')
    print(str(n_peatons_run/number_of_simulations) + ' has been the average peatons runned over by the car while reaching it\'s destination')


class QLearner:
    """
    A Wrapper for the Q-learning method, which uses multithreading
    in order to handle the game rendering.
    """

    def __init__(self, environment, policy, drawing=False):

        threading.Thread(target=example_execution, args=(environment, policy, drawing,)).start()
        if drawing:
            env.render('Evaluating')

if __name__ == "__main__":

    policy = np.load('new_38_31/policy_lex102.npy')

    initial_pedestrian_1_cell = 31
    initial_pedestrian_2_cell = 38
    isMoreStochastic = False
    env = Environment(isMoreStochastic=isMoreStochastic, initial_pedestrian_1_cell = initial_pedestrian_1_cell, initial_pedestrian_2_cell = initial_pedestrian_2_cell)

    print()
    print("-----------------------------------")
    print("Starting simulations!")
    print("-----------------------------------")
    QLearner(env, policy, drawing=False)