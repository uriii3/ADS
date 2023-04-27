"""
We have a convex hull (v_function.pickle, from 13/4/23 at 12:09) and will use it for different initial positions.
We will have the set of initial positions defined by the 2 positions that the pedestrians take:
p. ex: Policies_45_31 correspond to those done in the first part of the project and are supposed to work already.
We will add Policies_38_31 and maybe some other if the first one fits good in our methodology.
To do so, we will use 3 types of vectors, lexicographic order, final theoretical values and weights.
"""

from Learning import q_learning
from ADS_Environment import Environment
import numpy as np


def main():
    v_lexico = ['102', '120', '012', '021', '210', '201', 'unethical1']
    v_good = [True, True, True, True, True,
              True, True]  # if some lexicographic order has already a good policy mark it here

    # Basics things to change manually:
    pedestrian1_position = 31
    pedestrian2_position = 45

    v_weights = [[1.0, 1.33567, 0.00121],
                 [1.0, 1.898, 18.062],
                 [1.0, 0.0552239, 0.04267],
                 [1.0, 0.026691358024687736, 0.09333333333332157],
                 [1.0, 3.794431680730658, 36.09607370767785],
                 [1.0, 0.0072263374485599455, 0.8551440329218155],
                 [1.0, 0.0, 0.0]]

    v_values = [[7.0, 0.0, -4.72799497],
                [5.28332746, 0.0, 0.0],
                [10.0, -20.0, -4.48148148],
                [10.0, -30.0, -1.40740741],
                [5.28332746,  0.0, 0.0],
                [9.26646091, -22.5, 0.0],
                [10, -30, 0.0]] # no estic segur dels valors hehe

    # Preparations before loop
    env = Environment(is_deterministic=True)
    max_weights = 15000
    alpha = 0.8
    print(env.translate_state_cell(pedestrian1_position))
    env.initial_pedestrian_1_position = env.translate_state_cell(pedestrian1_position)
    env.initial_pedestrian_2_position = env.translate_state_cell(pedestrian2_position)

    for order, is_good, weights, exp_value in zip(v_lexico, v_good, v_weights, v_values):
        print('hola')
        if not is_good:
            print("Learning order ...", order)
            policy, v, q = q_learning(env, weights, alpha=alpha, max_weights=max_weights, max_episodes=70000)
            print("-------------------")

            print("The Learnt Policy has the following Value:")
            print(v[43, pedestrian2_position, pedestrian1_position])
            error = v[43, pedestrian2_position, pedestrian1_position] - exp_value
            if sum(abs(error)) < 0.5:
                print("This policy works!: ", order)
            # Save policy
            np.save("./more_stochasticity/policy_lex" + order + ".npy", policy)
            print("Saved!")


main()
