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

    if CCS:
        if len(vertices) > 4:
            new_vertices = [vertices[0]]
            for i in range(1, len(vertices) - 1):
                dist1 = np.linalg.norm(new_vertices[-1] - vertices[i])
                dist2 = np.linalg.norm(vertices[i+1] - vertices[i])
                dist3 = np.linalg.norm(new_vertices[-1] - vertices[i+1])
                if np.abs(dist1 + dist2 - dist3) > 0.001:
                    new_vertices.append(vertices[i])



            #for p in hull_points:
            #    dist1 = np.linalg.norm(vertices[-1] - p) + np.linalg.norm(vertices[0] - p)
            #    if np.abs(dist1 - dist2) > 0.00001:
            #        new_vertices.append(p)

            vertices = new_vertices


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

    puntitos = [[ 10. ,        -10.,           0.        ],
 [  9.77783203,  -9.99755859 ,  0.        ],
 [ -4.         ,  0.         ,  0.        ],
 [ -3.63049316 , -2.36999512 ,  0.        ],
 [ -3.57580566 , -2.40905762 ,  0.        ],
 [ -2.29089355 , -3.29345703 ,  0.        ],
 [  3.52178955 , -7.16156006 ,  0.        ],
 [  5.57537842 , -8.43353271 ,  0.        ],
 [  6.63787842 , -9.05853271 ,  0.        ],
 [  7.40209961 , -9.46655273 ,  0.        ],
 [  7.9118042  , -9.73724365 ,  0.        ],
 [  9.05957031 , -9.94628906 ,  0.        ],
 [  9.61132812 , -9.99023438,   0.        ]]
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
