### Naive implementation

"""
Given some data set, Array, generate and return a model that clusters the points based on:
 - Epsilon, a maximum distance for points to be considered connected
 - Min_points, a minimum number of points required for a group of points to be a cluster
"""
from Cluster_class import Cluster
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

class DBSCAN():
    def __init__(self, epsilon, min_points, cluster = Cluster()):
        self.epsilon = epsilon
        self.min_points = min_points
        self.cluster = cluster

    def get_distances(self, point, arr):
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
            
            if dist <= self.epsilon:
                distances[i, 1] = 1
                neighbor += 1

        return distances, neighbor


    def create_cluster(self, point, arr, cluster, c):
        """
        Recursively grow a cluster given a starting point, an array, a max distances, and a minimum number of points.
        Modifies the input cluster object by setting all points within the cluster to c.
        """
        distances, connections = self.get_distances(point, arr)

        # Every time we call this function, we add the point to the cluster
        cluster.set_label(point, c)

        # Recursion base case, we have run out of connecting points (reach a terminating point/leaf)
        if connections == 0:
            return 

        # If we are not at base case:
        # Continue jumping to connecting points and labeling c.
        # Each time we jump, we shorten the input array by
        # removing the connecting points from the input array.


        # Filter the array down to only connecting points
        # We use the fact that the indices for arr and distances correpsond to the same points
        connecting_points = arr[ distances[:,1] == 1 ]


        # We generate the new input array.
        # This is the array of points minus the connecting points (includes the original point itself)
        # We must subtract all connecting points instead of just the inpout point to prevent points from 
        # connecting back and forth with each other
        new_arr = arr[ distances[:,1] == 0 ]


        for p in connecting_points:
            
            self.create_cluster(p, new_arr, cluster, c)
            
        return None


    def fit(self, arr):

        clusters = Cluster(Array)
        Cluster_num = 0
        
        for i, point in enumerate(arr):
            
            # If the point has already been assigned a cluster, skip it
            if clusters.labels[i] != 0:
                continue

            # Get distances between point and all other points
            # Function returns a tuple where the first element is an 
            # array of distances and the second is the number of connections
            distances = self.get_distances(point, arr)

            if distances[1] < self.min_points:
                clusters.labels[i] = -1
                continue
            
            else: 
                clusters.labels[i] = Cluster_num
                Cluster_num += 1

                # create cluster starting from the given point
                self.create_cluster(point, arr, clusters, Cluster_num)

        self.cluster = clusters
        return self.cluster

    def __str__(self):
        """ Print cluster using the Cluster class's __str__ method. """
        return self.cluster.__str__()

if __name__ == "__main__": 

    Array = np.array([[0,0], [0,1], [0,2], [1,1], [5,1], [9,1], [10,0], [10,1], [10,2]])

    # x, y= Array[:,0], Array[:,1]
    # plt.scatter(x,y)
    # plt.show()

    epsilon = 3
    min_points = 4
    output = DBSCAN(epsilon, min_points)
    output.fit(Array)
    print(output)

    # x, y = output.data[:,0], output.data[:,1]
    # plt.scatter(x,y)
    # plt.show()

    q = output.cluster

    sns.scatterplot(x = q.data[:,0], y = q.data[:,1], hue = q.labels, palette=['green','orange','brown'])
    plt.show()