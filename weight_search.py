from Main import Ethical_Environment_Designer
import numpy as np

from ADS_Environment import Environment

lex_ordering = [1, 0, 2]  # order the correct values!! [1,2,0]
initial_states = [[43, 38, 31]]
env = Environment()
epsilon = 0.062
w_E = [0, 0, 0]
while w_E != [None, None, None]:
    epsilon += 0.0001
    discount_factor = 1.0
    max_iterations = 15

    w_E = Ethical_Environment_Designer(env, lex_ordering, epsilon, discount_factor, max_iterations,
                                       initial_states=[[43, 38, 31]])
    print("Ethical weights found: ", w_E)
    print(epsilon)
