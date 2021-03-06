import numpy as np

height = [10, 20, 30]
weight = [100,200,300]

weight + height
height ** 2
height > 20

np_height = np.array(height)
np_weight = np.array(weight)
np_height.shape

np_height[0]
np_height[0:2]

np_height + np_weight
np_height.dot(np_weight)

np_height ** 2
np_height > 20
np_height[np_height>20]

np_height.min()
np_height.mean()
np_height.argmin()
np_height.sort()

zero1 = np.zeros(2, int)
type(zero1)
zero1.shape

ones1 = np.ones(3)
ones1.shape

ones2 = np.ones(3, int)
ones2.shape

list1 = [10, True, 'abc']
array1 = np.array(list1)

