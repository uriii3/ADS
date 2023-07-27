import numpy as np
from ADS_Environment import Environment
import threading
import time
from VI import translate_action
import csv

def example_execution(env, policy, render=False, save = False, stop=False):
    """
    Simulation of the environment without learning.

    :param env: the environment encoding the (MO)MDP
    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """
    max_timesteps = 200
    number_of_simulations = 5

    n_steps = 0
    n_peatons_run = 0
    n_bumps_coll = 0
    n_dangerous = 0
    prev_state = []
    if save:
        f = open('celeste_results/results_ini[2, 1, 0].csv', 'w', newline='')
        writer = csv.writer(f)

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

        history_actions = list()
        history_rewards = list()
        history_state = list()

        while (timesteps < max_timesteps) and (not done):
            timesteps += 1
            v = []
            actions = list()
            actions.append(policy[state[0], state[1], state[2]])
            history_actions.append(policy[state[0], state[1], state[2]])

            if save:
                v.append([env.translate_state_cell(state[0]), env.translate_state_cell(state[1]), env.translate_state_cell(state[2])])
                v.append(translate_action(actions[0]))

            if stop:
                actions = [LEFT, RIGHT, RIGHT]

            state, rewards, dones = env.step(actions)

            if save:
                v.append([env.translate_state_cell(state[0]), env.translate_state_cell(state[1]), env.translate_state_cell(state[2])])
                v.append(rewards.tolist())
                v.append(dones[0])
                v.append([[2, 2], [2, 5], [4, 1]])
                print(v)
                print("-------")
                writer.writerow(v)
            #print("State :", state)
            history_rewards.append(rewards)
            history_state.append(state)

            if rewards[2] == -10.0:
                n_peatons_run += 1
            if rewards[2] == -3.0:
                n_dangerous += 1
            if rewards[2] == -6.0:
                n_dangerous += 2
            if rewards[2] == -13.0:
                n_peatons_run += 1
                n_dangerous += 1
            if rewards[2] == -20.0:
                n_peatons_run += 2
                #print("ara!")
                #print(rewards[2])
            if rewards[1] != 0.0: n_bumps_coll += 1
            #print(rewards)
            prev_state = state
            done = dones[0]  # R Agent does not interfere

            if render:
                if not env.drawing_paused():
                    time.sleep(1.5)
                    env.update_window()

        n_steps += timesteps
        #print(i)
        #if timesteps != 8:
            #print("hola: ", i)
            #print(timesteps)
        #print(n_steps)
    if save:
        f.close()

    print()
    print('The results of these ' + str(number_of_simulations) + ' simulations are:')
    print(str(n_steps/number_of_simulations) + ' has been the average steps for the car to reach it\'s destination')
    print(str(15-(n_steps/number_of_simulations)) + ' total value of obj 0.')
    print()
    print(str(n_bumps_coll/number_of_simulations) + ' has been the average incidents with bumps while reaching it\'s destination')
    print(str((n_bumps_coll/number_of_simulations)*-10) + ' total value regarding bumps')
    print()
    print(str(n_peatons_run/number_of_simulations) + ' has been the average peatons runned over by the car while reaching it\'s destination')
    print(str(n_dangerous/number_of_simulations) + ' has been the average dangerous situations it\'s destination')
    print(str((n_peatons_run/number_of_simulations)*-10 + (n_dangerous/number_of_simulations)*-3) + " the total value of the external safety value")

class QLearner:
    """
    A Wrapper for the Q-learning method, which uses multithreading
    in order to handle the game rendering.
    """

    def __init__(self, environment, policy, drawing=False, save = False):

        threading.Thread(target=example_execution, args=(environment, policy, drawing, save,)).start()
        if drawing:
            env.render('Evaluating')

if __name__ == "__main__":

    policy = np.load('new_more_sto/policy_lex210.npy')

    initial_pedestrian_1_cell = 31
    initial_pedestrian_2_cell = 45
    isMoreStochastic = True
    env = Environment(isMoreStochastic=isMoreStochastic, initial_pedestrian_1_cell = initial_pedestrian_1_cell, initial_pedestrian_2_cell = initial_pedestrian_2_cell)

    print()
    print("-----------------------------------")
    print("Starting simulations!")
    print("-----------------------------------")
    QLearner(env, policy, drawing=True, save = False)