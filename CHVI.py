import numpy as np
import convexhull
from ADS_Environment import Environment

agent_2_actions = Environment.pedestrian_move_map



def Q_function_calculator(env, state, V, discount_factor, model_used=None, epsilon=-1.0):
    """

    Calculates the (partial convex hull)-value of applying each action to a given state.
    Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the new convex obtained after checking for each action (this is the operation hull of unions)
    """

    state_translated = env.translate_state(state)
    hulls = list()
    action2 = agent_2_actions[state_translated[1][0]][state_translated[1][1]]
    action3 = agent_2_actions[state_translated[2][0]][state_translated[2][1]]


    dividing_factor = float(1./(len(action2)*len(action3)))
    for action in range(env.n_actions):
        hull_sa_all = []

        for act2 in action2:

            for act3 in action3:

                if model_used is not None:
                    all_things = model_used[state[0], state[1], state[2], action, act2, act3]

                    next_state = [int(all_things[i]) for i in range(3,6)]
                    rewards = all_things[:3]



                else:
                    env.easy_reset(state_translated[0], state_translated[1], state_translated[2])
                    next_state, rewards, _ = env.step([action, act2, act3])


                V_state = dividing_factor*V[next_state[0]][next_state[1]][next_state[2]].copy()
                alt_rewards = dividing_factor*rewards
                hull_sa = convexhull.translate_hull(alt_rewards, discount_factor, V_state)
                hull_sa_all = convexhull.sum_hulls(hull_sa, hull_sa_all, epsilon=epsilon)

        for point in hull_sa_all:
            hulls.append(point)

    hulls = np.unique(np.array(hulls), axis=0)

    new_hull = convexhull.get_hull(hulls, epsilon=epsilon)

    return new_hull

def partial_convex_hull_value_iteration(env, discount_factor=1.0, max_iterations=20, model_used=None):
    """
    Partial Convex Hull Value Iteration algorithm adapted from "Convex Hull Value Iteration" from
    Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Calculates the partial convex hull for each state of the MOMDP

    :param env: the environment encoding the MOMDP
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: value function storing the partial convex hull for each state
    """

    n_cells = env.map_num_cells

    V = list()
    for i in range(n_cells):
        V.append(list())
        for j in range(n_cells):
            V[i].append(list())
            for k in range(n_cells):
                V[i][j].append(np.array([]))

    iteration = 0
    print("Number of states:", len(env.states_agent_left) * len(env.states_agent_right) ** 2)
    eps = origi_eps = 0.7
    while iteration < max_iterations:
        iteration += 1
        n_states = 0
        if origi_eps >= 0.0:
            eps *= origi_eps
        # Sweep for every state

        for cell_L in env.states_agent_left:
            for cell_R in env.states_agent_right:
                for cell_J in env.states_agent_right:
                    if cell_R >= cell_J:
                        n_states += 1
                        if n_states % 1000 == 0:
                            pass
                            #print(n_states)

                        V[cell_L][cell_R][cell_J] = Q_function_calculator(env, [cell_L, cell_R, cell_J], V, discount_factor, model_used, eps)

        print("Iterations: ", iteration, "/", max_iterations)
        #print(V[43][38][31])
    return V


def learn_and_do():
    env = Environment()
    v = partial_convex_hull_value_iteration(env, model_used=None)
    np.save("v_function.npy", v)


if __name__ == "__main__":

    learn = True

    if learn:
        learn_and_do()

    # Returns partial convex hull of initial state
    v_func = np.load("v_function.npy", allow_pickle=True)
    #print(v_func[43][31][31])
    print("--")
    #print(v_func[43][38][38])
    print(v_func[43][38][31])



