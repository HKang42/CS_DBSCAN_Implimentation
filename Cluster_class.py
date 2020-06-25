import numpy as np

class Cluster():
    def __init__(self, data = None, labels = None, index = None):
        self.data = data
        if type(self.data) is np.ndarray:
            self.labels = [0] * len(self.data)
            self.index = [i for i in range(0, len(data))]
        
        else: 
            self.data = np.array([])
        

    def __str__(self):
        print_str = "{:>5}{:>12}{:>11}\n".format("Index", "Value", "Label")
        for i in range(len(self.data)):
            print_str += "{:>5}{:>12}{:>11}\n".format(str(self.index[i]), str(self.data[i]), str(self.labels[i]))
        return print_str


    def __getitem__(self, i):
        return self.data[i]


    def __setitem__(self, i, data_val):
        """
        Set the data value and, optionally, the label for a given index.
        Input should be a numpy array or a tuple of an array and an integer.
        If a label is not provided, the default value of 0 is used.
        """
        if type(data_val) is tuple:
            
            if type(data_val[0]) is not np.ndarray:
                raise TypeError(f"Expected type {np.ndarray}, got {type(data_val[0])} instead.")
            self.data[i] = data_val[0]

            if (type(data_val[1]) is not int) and (type(data_val[1]) is not str):
                raise TypeError(f"Expected type {int} or {str}, got {type(data_val[1])} instead.")
            self.labels[i] = data_val[1]

        else:
            if type(data_val) is not np.ndarray:
                raise TypeError(f"Expected type {np.ndarray} or {tuple}, got {type(data_val[0])} instead.")
            self.data[i] = data_val
            self.labels[i] = 0


    def set_label(self, value, label, assign = 'All'):
        """
        Set the label for a given index or data value. 
        If multiple points have the same value, the method defaults to setting the label for each one.
        This can be overridden by setting the 'assign' parameter to integer, n. 
        The method will then assign only the first n points with the same value.
        """
        if type(value) is int:
            self.labels[value] = label

        elif type(value) is np.ndarray:

            # Get the index for every data value that matches the input value
            indices = np.all(self.data == value, axis=1).nonzero()[0]

            if assign == 'All':
                for i in indices:
                    self.labels[i] = label
            
            else:
                i = 0
                while i < assign and i < len(indices):
                    self.labels[i] = label
                    i+=1

        else:
            raise TypeError(f"Expected type {np.ndarray} or {int}, got {type(value)} instead.")


if __name__ == "__main__": 

    # Create Object
    Array = np.array([[0,0], [0,1], [0,2], [1,1], [5,1], [9,1], [10,0], [10,1], [10,2]])
    clusters = Cluster(Array)

    # Test print function
    print(clusters)


    # # Test __setitem__
    # clusters[0] = Array[8], '1'
    # print(clusters)

    # Test set_label 
    clusters.set_label(Array[8], 'test 1')
    print(clusters)

    clusters[3] = Array[8]
    clusters[4] = Array[8]
    clusters[5] = Array[8]

    clusters.set_label(Array[8], 'test 2')
    print(clusters)

