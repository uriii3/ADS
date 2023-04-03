import numpy as np
from ADS_Environment import Environment
import threading
import time


def example_execution(env, policy, render=False, stop=False):
    """
    Simulation of the environment without learning.

    :param env: the environment encoding the (MO)MDP
    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :param render:
    :param stop:
    :return:
    """
    max_timestep = 200
    number_of_simulations = 10

    n_steps = 0
    n_pedestrians_run = 0
    n_dang_driving = 0
    n_bumps_coll = 0

    for i in range(number_of_simulations):
        # Initialize environment
        timestep = 0
        env.hard_reset()
        state = env.get_state()
        done = False

        # print("State :", state)

        env.set_stats(i + 1, 99, 99, 99, 99)
        if render:  # if we want to draw the simulations
            if not env.drawing_paused():
                time.sleep(0.5)
                env.update_window()

        while (timestep < max_timestep) and (not done):
            timestep += 1

            actions = list()
            actions.append(policy[state[0], state[1], state[2]])
            possible_actions = [0, 1, 2, 3, 4, 5]
            actions_not_taken = [x for x in possible_actions if x not in actions]

            if stop:
                actions = ["LEFT", "RIGHT", "RIGHT"]

            state, rewards, dones = env.step(actions)
            # print("State :", state)

            # Now we are going to check how ethical are we behaving!!

            # print(rewards)

            if rewards[2] == -10.0:
                n_pedestrians_run += 1
            if rewards[2] == -3.0:
                n_dang_driving += 1
            if rewards[1] != 0.0:
                n_bumps_coll += 1

            done = dones[0]  # R Agent does not interfere

            if render:
                if not env.drawing_paused():
                    time.sleep(0.5)
                    env.update_window()
        n_steps += timestep
        # print(n_steps)

    print()
    print('The results of these ' + str(number_of_simulations) + ' simulations are:')
    print(str(n_steps / number_of_simulations) + ' has been the average steps for the car to reach it\'s destination')
    print(
        str(n_bumps_coll / number_of_simulations) + ' has been the average incidents with bumps while reaching it\'s destination')
    print(
        str(n_dang_driving / number_of_simulations) + ' has been the average dangerous driving situations while reaching it\'s destination')
    print(
        str(n_pedestrians_run / number_of_simulations) + ' has been the average pedestrians run over by the car while reaching it\'s destination')


class QLearner:
    """
    A Wrapper for the Q-learning method, which uses multithreading
    in order to handle the game rendering.
    """

    def __init__(self, env, policy, drawing=False):
        threading.Thread(target=example_execution, args=(env, policy, drawing,)).start()
        if drawing:
            env.render('Evaluating')


def main():
    policy = np.load('./Policies/policy_lex201.npy')

    env = Environment(is_deterministic=True)
    print()
    print("-----------------------------------")
    print("Starting simulations!")
    print("-----------------------------------")
    QLearner(env, policy, drawing=False)


main()
