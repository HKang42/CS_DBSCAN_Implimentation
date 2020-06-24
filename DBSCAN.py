### Naive implementation

"""
Given some data set, Array, generate and return a model that clusters the points based on:
 - Epsilon, a maximum distance for points to be considered connected
 - Min_points, a minimum number of points required for a group of points to be a cluster
"""
from Cluster_class import Cluster
import numpy as np
import matplotlib.pyplot as plt

def get_distances(point, arr, epsilon):
    """
    Given a point and an n x m array, calculate the distance between that point and every other point in the array
    Record the distances in a new n x 2 array. 
    [n, 0] contains the distance value. 
    [n, 1] is 1 if the distance is less or equal to epsilon (is a connection) and 0 otherwise.

    Returns a tuple where the first entry is the array of distances and the second is the number of connections
    """
    distances = np.zeros((len(arr), 2))
    neighbor = 0
    for i in range(len(arr)):
        #print("a", point, "b",arr[i], "c",np.linalg.norm(point - arr[i]))
        dist = np.linalg.norm(point - arr[i])
        distances[i, 0] = dist
        
        if dist <= epsilon:
            distances[i, 1] = 1
            neighbor += 1

    return distances, neighbor

"""
Build model:

loop through Array
    
    Check if point is a cluster:
        If yes, then skip
    
    for a point, 
        calculate its distance to every other point
        get the number of points that are less than or equal to Epsilon

    If number of points is less than min_pts:
        label the point as noise
        move on to next point in Array

    else:
        generate a cluster.
        
"""



def DBSCAN(arr, min_points, epsilon):

    clusters = Cluster(Array)
    Cluster_num = 1
    
    for i, point in enumerate(arr):
        
        if clusters.labels[i] != 0:
            continue

        # get distances between point and all other points
        distances = get_distances(point, arr, epsilon)
        #print(distances[1])

        if distances[1] < min_points:
            clusters.labels[i] = "N"
            continue
        
        else: 
            clusters.labels[i] = Cluster_num
            Cluster_num += 1

            # create cluster starting from the given point


            #pass

    return clusters


Array = np.array([[0,0], [0,1], [0,2], [1,1], [5,1], [9,1], [10,0], [10,1], [10,2]])

# x, y= Array[:,0], Array[:,1]
# plt.scatter(x,y)
# plt.show()


min_points = 4
epsilon = 3
output = DBSCAN(Array, min_points, epsilon)
#print(output)




"""
grow a cluster starting from some point and given a max distances and min number of points


"""

def create_cluster(point, arr, epsilon, min_points, cluster = []):

    distances, connections = get_distances(point, arr, epsilon)

    # base case, we run out of points
    if connections == 0:
        return
    

    # continue to jump to connecting points
    # each time we jump, we shorten the input array
    # remove the points that have been checked

    # Filter the array down to only connecting points
    # Remember that the indices for arr and distances correpsond to the same points
    connecting_points = arr[distances[:,1]==1]

    # Filter the array down to only connecting points and grab the indices
    # We need the indices to act as a primary key between arr, distances, and the clusters object
    #connecting_points = np.where(distances[:,1]==1)[0]

    """
    maybe I should assume that the point values can be used to link back to the clusters object.
    
    then I can just create a list of points and append after each loop
    OR 
    then I can search the cluster object for the corresponding point(s) and assign their labels
    """


    for p in connecting_points:
        pass
    
    return connecting_points


Array = np.array([[0,0], [0,1], [0,2], [1,1], [5,1], [9,1], [10,0], [10,1], [10,2]])

q = create_cluster(Array[5], Array, 3, 4)
print(q)

q = Cluster(Array)
print(q)
# print(q[0])
# q.labels[0] = 2
# print(q)