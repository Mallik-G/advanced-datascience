import numpy as np

height = [10, 20, 30]
weight = [100,200,300]

weight + height
height ** 2
height > 20

np_height = np.array(height)
np_weight = np.array(weight)

np_height + np_weight

np_height ** 2
np_height > 20
np_height[np_height>20]

list1 = [10, True, 'abc']
array1 = np.array(list1)

np.mean(height)

a1 = np.array([[1,2,3], [4,5,6]])
a1[1,1]
a1[1:2,1]

