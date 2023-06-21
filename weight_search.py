from Main import Ethical_Environment_Designer

from ADS_Environment import Environment

lex_ordering = [1, 0, 2]  # order the correct values!!
initial_pedestrian_1_cell = 31
initial_pedestrian_2_cell = 45
isMoreStochastic = True # remember to change the file you are grabbing from Main.py!!
env = Environment(isMoreStochastic=isMoreStochastic, initial_pedestrian_1_cell=initial_pedestrian_1_cell,
                  initial_pedestrian_2_cell=initial_pedestrian_2_cell)
initial_states = [[43, initial_pedestrian_2_cell, initial_pedestrian_1_cell]]
epsilon = 0.0
w_E = [0, 0, 0]
while w_E != [None, None, None]:
    epsilon += 0.01
    discount_factor = 1.0
    max_iterations = 15

    w_E = Ethical_Environment_Designer(env, lex_ordering, epsilon, discount_factor, max_iterations,
                                       initial_states=initial_states)
    print("Ethical weights found: ", w_E)
    print(epsilon)
