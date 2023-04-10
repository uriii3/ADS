from scipy.spatial import ConvexHull
import numpy as np


def non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            dominated_points = (np.asarray(solutions[is_efficient]) <= c).all(axis=1)
            is_efficient[is_efficient] = np.invert(dominated_points)
            # keep the point itself, otherwise we would get an empty list
            is_efficient[i] = 1

    return solutions[is_efficient]


def get_hull(points, CCS=True):


    # if we want only the positive half of the hull we do this preprocessing step
    # WARNING: This step is NOT necessary, but it seems to speed up execution
    if CCS:
        points = non_dominated(np.array(points))


    # Compute new hull (try-except for handling hulls with less than 3 points)

    dimension_redux_worked = False
    try:
        if CCS:
            d = 0
            while not dimension_redux_worked:
                if np.max(points[:,d]) == np.min(points[:,d]):
                    hull = ConvexHull(points[:,:d])
                    dimension_redux_worked = True
                else:
                    d += 1

        if not dimension_redux_worked:
                hull = ConvexHull(points)

        hull_points = [points[vertex] for vertex in hull.vertices]
    except:
        hull_points = points


    # Now, if we want only the positive half of the hull, remove points that are optimal for negative weights
    # WARNING: This step IS necessary for computing the half hull
    if CCS:
        vertices = non_dominated(np.array(hull_points))
    else:
        vertices = hull_points

    return np.array(vertices)



def translate_hull(point, gamma, hull):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Translation and scaling operation of convex hulls (definition 1 of the paper).

    :param point: a 2-D numpy array
    :param gamma: a real number
    :param hull: a set of 2-D points, they need to be numpy arrays
    :return: the new convex hull, a new set of 2-D points
    """

    if len(hull) == 0:
        hull = [point]
    else:

        hull = np.multiply(hull, gamma, casting="unsafe")

        if len(point) > 0:
            hull = np.add(hull, point, casting="unsafe")

    return hull


def sum_hulls(hull_1, hull_2):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Sum operation of convex hulls (definition 2 of the paper)

    :param hull_1: a set of 2-D points, they need to be numpy arrays
    :param hull_2: a set of 2-D points, they need to be numpy arrays
    :return: the new convex hull, a new set of 2-D points
    """
    if len(hull_1) == 0:
        return hull_2
    elif len(hull_2) == 0:
        return hull_1

    new_points = None

    for i in range(len(hull_1)):
        if new_points is None:
            new_points = translate_hull(hull_1[i].copy(), 1,  hull_2.copy())
        else:
            new_points = np.concatenate((new_points, translate_hull(hull_1[i].copy(), 1, hull_2.copy())), axis=0)

    return get_hull(new_points)

def max_q_value(weight, hull):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Extraction of the Q-value (definition 3 of the paper)

    :param weight: a weight vector, can be simply a list of floats
    :param hull: a set of 2-D points, they need to be numpy arrays
    :return: a real number, the best Q-value of the hull for the given weight vector
    """
    scalarised = []

    for i in range(len(hull)):
        f = np.dot(weight,hull[i])
        #print(f)
        scalarised.append(f)

    scalarised = np.array(scalarised)

    return np.max(scalarised)


if __name__ == "__main__":

    points = np.random.rand(42, 2)   # 15 random points in 2-D

    puntitos = [[-4.,0.,0.],
     [-3.75,-1.25,0.],
    [-3.625,-1.40625,0.],
    [-3.34375,-1.5625,0.],
    [-3.21875,-1.71875,0.],
    [-2.84375,-1.875,0.],
    [-2.65625,-2.03125,0.],
    [-2.375,-2.1875,0.],
    [-1.5,-2.5,0.],
    [-1.34375,-2.8125,0.],
    [-1.21875,-2.96875,0.],
    [-0.84375,-3.125,0.],
    [-0.65625,-3.28125,0.],
    [-0.375,-3.4375,0.],
    [0.25,-3.75,0.],
    [0.375,-3.90625,0.],
    [0.65625,-4.0625,0.],
    [0.78125,-4.21875,0.],
    [1.15625,-4.375,0.],
    [1.34375,-4.53125,0.],
    [1.625,-4.6875,0.],
    [3.,-5.,0.],
    [3.15625,-5.625,0.],
    [3.34375,-5.78125,0.],
    [3.625,-5.9375,0.],
    [4.25,-6.25,0.],
    [4.375,-6.40625,0.],
    [4.65625,-6.5625,0.],
    [4.78125,-6.71875,0.],
    [5.15625,-6.875,0.],
    [5.34375,-7.03125,0.],
    [5.625,-7.1875,0.],
    [6.5,-7.5,0.],
    [6.65625,-7.8125,0.],
    [6.78125,-7.96875,0.],
    [7.15625,-8.125,0.],
    [7.34375,-8.28125,0.],
    [7.625,-8.4375,0.],
    [8.25,-8.75,0.],
    [8.375,-8.90625,0.],
    [8.65625,-9.0625,0.],
    [8.78125,-9.21875,0.],
    [9.15625,-9.375,0.],
    [9.34375, -9.53125,0.],
    [9.625,-9.6875,0.],
    [12.,-10.,0.]]
    puntitos = np.array(puntitos)
    print(puntitos)

    v_function = [[3., 2.],
                  [5., -1.5],
                  [-20., 4.]]

    v_functionE = [[-3., 0.],
                   [4., 2.],
                   [5., 3.]]

    v_function = np.array(v_function)
    v_functionE = np.array(v_functionE)

    vertices = get_hull(puntitos)

    print("resulting vertices")
    print(vertices)
    import matplotlib.pyplot as plt

    plt.plot(vertices[:,0], vertices[:,1], 'k-')
    #max_q_value([1.0,0.4],vertices)
    plt.show()
