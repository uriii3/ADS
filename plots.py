import numpy as np
from ADS_Environment import Environment
import matplotlib.pyplot as plt


def plotting(environment, policies_to_compare):

    """
    Simulation of the environment without learning.

    :param environment: the environment encoding the (MO)MDP
    :param vpolicies: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """
    vpolicies = []
    # We save all policies we will compare inside the vector:
    for i in range(0, len(policies_to_compare)):
        vpolicies.append(np.load('./Policies/policy_lex' + policies_to_compare[i] + '.npy'))

    max_timesteps = 200
    number_of_simulations = 500
    n_policies = len(vpolicies) # number of policies we will take intp account

    n_steps = np.zeros(n_policies, dtype=object)
    n_peatons_run = np.zeros(n_policies, dtype=object)
    n_bumps_coll = np.zeros(n_policies, dtype=object)
    v_steps = np.zeros((n_policies, 14))
    v_peatons_run = np.zeros((n_policies, 4))
    v_bumps_coll = np.zeros((n_policies, 5))

    for i in range(n_policies):
        policy = vpolicies[i]
        for j in range(number_of_simulations):
            # Initialize environment
            timesteps = 0
            int_peatons_run = 0
            int_bumps_coll = 0
            environment.hard_reset()
            state = environment.get_state()
            done = False

            #print("State :", state)

            while (timesteps < max_timesteps) and (not done):
                timesteps += 1

                actions = list()
                actions.append(policy[state[0], state[1], state[2]])

                state, rewards, dones = environment.step(actions)
                #print("State :", state)

                #print(rewards)

                if rewards[2] != 0.0: int_peatons_run += 1
                if rewards[1] != 0.0: int_bumps_coll += 1

                done = dones[0]  # R Agent does not interfere

            n_steps[i] += timesteps
            n_peatons_run[i] += int_peatons_run
            n_bumps_coll[i] += int_bumps_coll

            # Sumem 1 a cada simulació que correspon
            v_steps[i][timesteps] += 1
            v_peatons_run[i][int_peatons_run] += 1
            v_bumps_coll[i][int_bumps_coll] += 1
            #print(n_steps)

    plt.figure()
    for i in range(n_policies):
        plt.subplot(3, 1, 1)
        plt.bar(np.arange(len(v_steps[0])), v_steps[i], label=policies_to_compare[i], alpha = 0.5)
        plt.subplot(3, 1, 2)
        plt.bar(np.arange(len(v_peatons_run[0])), v_peatons_run[i], label=policies_to_compare[i], alpha = 0.5)
        plt.subplot(3, 1, 3)
        plt.bar(np.arange(len(v_bumps_coll[0])), v_bumps_coll[i], label=policies_to_compare[i], alpha = 0.5)
    plt.subplot(3, 1, 1)
    plt.title("Steps to reach destination")
    plt.subplot(3, 1, 2)
    plt.title("Runned peatons in one go")
    plt.subplot(3, 1, 3)
    plt.title("Bumps taken in one go")
    plt.legend()
    plt.show()

    print()
    print('The results of these ' + str(number_of_simulations) + ' simulations are:')
    print(str(n_steps/number_of_simulations) + ' has been the average steps for the car to reach it\'s destination')
    print(str(n_bumps_coll/number_of_simulations) + ' has been the average incidents with bumps while reaching it\'s destination')
    print(str(n_peatons_run/number_of_simulations) + ' has been the average peatons runned over by the car while reaching it\'s destination')


if __name__ == "__main__":

    policies_to_compare = [
        "012",
        "021",
        "210"
    ]

    # Initialize the environment:
    env = Environment(is_deterministic=True)
    print()
    print("-----------------------------------")
    print("Starting simulations!")
    print("-----------------------------------")

    plotting(env, policies_to_compare)