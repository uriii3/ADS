import numpy as np
from ADS_Environment import Environment
import matplotlib.pyplot as plt


def transcribe_label(numeros):
    if numeros == "000" or numeros == "unethical":
        return "Unethical"
    first, second, third = [*numeros]
    dictionary = {"0":"Velocity", "1": "Internal (bumps)", "2": "External (pedestrians)"}
    dictionary2 = {"0":"Velocity", "1": "Internal", "2": "External"}
    return numeros + ": " + dictionary2[first] + " > " + dictionary2[second] + " > " + dictionary2[third]


def plotting(environment, policies_to_compare, initial_state):

    """
    Simulation of the environment without learning.

    :param environment: the environment encoding the (MO)MDP
    :param vpolicies: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """
    #Variables to set:
    max_timesteps = 200
    number_of_simulations = 250

    # Variables taken into account quickly derived
    n_policies = len(policies_to_compare) # number of policies we will take intp account
    unethical = False
    vpolicies = []
    n_unethical_policies = 2

    root = './Policies_' + initial_state + '/'
    # We save all policies we will compare inside the vector:
    if policies_to_compare[len(policies_to_compare)-1] == "unethical":
        unethical = True
        for i in range(0, len(policies_to_compare)-1):
            vpolicies.append(np.load(root + 'policy_lex' + policies_to_compare[i] + '.npy'))
        for i in range(0,n_unethical_policies): # change in case more policies are added
            vpolicies.append(np.load(root + 'policy_lexunethical' + str(i) + '.npy'))

    else:
        for i in range(0, len(policies_to_compare)):
            vpolicies.append(np.load(root + 'policy_lex' + policies_to_compare[i] + '.npy'))

    n_steps = np.zeros(n_policies, dtype=object)
    n_peatons_run = np.zeros(n_policies, dtype=object)
    n_dangerous_driving = np.zeros(n_policies, dtype=object)
    n_bumps_coll = np.zeros(n_policies, dtype=object)
    v_steps = np.zeros((n_policies, 15))
    v_peatons_run = np.zeros((n_policies, 4))
    v_dangerous_driving = np.zeros((n_policies, 4))
    v_bumps_coll = np.zeros((n_policies, 5))

    for i in range(n_policies):
        policy = vpolicies[i]
        for j in range(number_of_simulations):
            if unethical and i==n_policies-1: # we are on the unethical simulation
                if j%(number_of_simulations/n_unethical_policies) == 0:
                    print(j)
                    print(number_of_simulations)
                    print(n_unethical_policies)
                    policy = vpolicies[i+int(j/(number_of_simulations/n_unethical_policies))]

            # Initialize environment
            timesteps = 0
            int_peatons_run = 0
            int_dangerous_driving = 0
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

                if rewards[2] == -10.0: int_peatons_run += 1
                if rewards[2] == -3.0: int_dangerous_driving += 1
                if rewards[1] != 0.0: int_bumps_coll += 1

                done = dones[0]  # R Agent does not interfere

            n_steps[i] += timesteps
            n_peatons_run[i] += int_peatons_run
            n_dangerous_driving[i] += int_dangerous_driving
            n_bumps_coll[i] += int_bumps_coll

            # Sumem 1 a cada simulació que correspon
            v_steps[i][timesteps] += 1
            v_peatons_run[i][int_peatons_run] += 1
            v_dangerous_driving[i][int_dangerous_driving] += 1
            v_bumps_coll[i][int_bumps_coll] += 1
            #print(n_steps)

    space = 1.5
    width = 1/(n_policies+space)
    center = (n_policies-1)/2
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'] # in theory colorblind friendly :)
    CB_hatch_cycle = ['/', 'o', 'x', '..', '*', '', '/', 'o', 'x']
    plt.figure()
    for i in range(n_policies):
        plt.subplot(4, 1, 1)
        plt.bar(np.arange(len(v_steps[0]))+width*(i-center), v_steps[i], label=transcribe_label(policies_to_compare[i]),
                alpha = 0.5, width=width, color = CB_color_cycle[i], hatch=CB_hatch_cycle[i])
        plt.xticks(np.arange(len(v_steps[0])), np.arange(len(v_steps[0])))
        plt.subplot(4, 1, 2)
        plt.bar(np.arange(len(v_peatons_run[0]))+width*(i-center), v_peatons_run[i], label=transcribe_label(policies_to_compare[i]),
                alpha = 0.5, width=width, color = CB_color_cycle[i], hatch=CB_hatch_cycle[i])
        plt.xticks(np.arange(len(v_peatons_run[0])), np.arange(len(v_peatons_run[0])))
        plt.subplot(4, 1, 3)
        plt.bar(np.arange(len(v_dangerous_driving[0])) + width * (i - center), v_dangerous_driving[i], label=transcribe_label(policies_to_compare[i]),
                alpha=0.5, width=width, color=CB_color_cycle[i], hatch=CB_hatch_cycle[i])
        plt.xticks(np.arange(len(v_dangerous_driving[0])), np.arange(len(v_dangerous_driving[0])))
        plt.subplot(4, 1, 4)
        plt.bar(np.arange(len(v_bumps_coll[0]))+width*(i-center), v_bumps_coll[i], label=transcribe_label(policies_to_compare[i]),
                alpha = 0.5, width=width, color = CB_color_cycle[i], hatch=CB_hatch_cycle[i])

    plt.subplot(4, 1, 1)
    plt.title("Steps to reach destination")
    for xx in np.arange(len(v_steps[0])):
        plt.axvline(x=xx+0.5, linestyle='--', alpha = 0.3, color='red', lw=0.5)
    plt.subplot(4, 1, 2)
    plt.title("Runned peatons in one go", y=1.0, pad=-14)
    for xx in np.arange(len(v_peatons_run[0])):
        plt.axvline(x=xx+0.5, linestyle='--', alpha = 0.3, color='red', lw=0.5)
    plt.subplot(4, 1, 3)
    plt.title("Dangerous driving situations in one go", y=1.0, pad=-14)
    plt.legend(loc='center right', bbox_to_anchor=(1.1, 0.8), fancybox=True, shadow=True)
    for xx in np.arange(len(v_dangerous_driving[0])):
        plt.axvline(x=xx + 0.5, linestyle='--', alpha=0.3, color='red', lw=0.5)
    plt.subplot(4, 1, 4)
    plt.title("Bumps taken in one go", y=1.0, pad=-14)
    for xx in np.arange(len(v_bumps_coll[0])):
        plt.axvline(x=xx+0.5, linestyle='--', alpha = 0.3, color='red', lw=0.5)

    plt.show()

    print()
    print('The results of these ' + str(number_of_simulations) + ' simulations are:')
    print(str(n_steps/number_of_simulations) + ' has been the average steps for the car to reach it\'s destination')
    print(str(n_bumps_coll/number_of_simulations) + ' has been the average incidents with bumps while reaching it\'s destination')
    print(str(n_peatons_run/number_of_simulations) + ' has been the average peatons runned over by the car while reaching it\'s destination')
    print(str(n_dangerous_driving/number_of_simulations) + ' has been the average dangerous driving situations while reaching it\'s destination')


def main():
    initial_state = "38_31"

    policies_to_compare = [
        "210",
        "120",
        "102",
        "201",
        "012",
        "021",
        "unethical"
    ]

    # Initialize the environment:
    env = Environment(is_deterministic=True) # he d'anar amb compte amb això!!
    env.initial_pedestrian_1_position = env.translate_state_cell(31)
    env.initial_pedestrian_2_position = env.translate_state_cell(38)
    print()
    print("-----------------------------------")
    print("Starting simulations!")
    print("-----------------------------------")

    plotting(env, policies_to_compare, initial_state)


main()