import numpy as np

class Cluster(tuple):
    def __init__(self, data = None, labels = None):
        self.data = data
        if self.data.all() == None:
            self.data = np.array([])
        
        self.labels = [0] * len(self.data)
    
    def __str__(self):
        print_str = ''
        for i in range(len(self.data)):
            print_str += "{}\t{:>4}\n".format(str(self.data[i]), str(self.labels[i]))
        return print_str



# Array = np.array([[0,0], [0,1], [0,2], [1,1], [5,1], [9,1], [10,0], [10,1], [10,2]])
# clusters = Cluster(Array)

# print(clusters)
