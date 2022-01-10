'''
Descripttion: 
version: 
Date: 2022-01-03 00:07:15
LastEditTime: 2022-01-03 01:39:10
'''
# import numpy as np

# a = np.array([1, 2, 3, 4])
# b = np.array((5, 6, 7, 8))
# c = np.array([[11, 2, 8, 4], [4, 52, 6, 17], [2, 8, 9, 100]])

# print(a)
# print(b)
# print(c)

# print(np.argmin(c))
# print(np.argmin(c, axis=0)) # 按每列求出最小值的索引
# print(np.argmin(c, axis=1)) # 按每行求出最小值的索引
# # 最小的话 min换成max

import numpy

a = numpy.array(([3,2,1],[2,5,7],[4,7,8]))

itemindex = numpy.argwhere(a == 7)

for item in itemindex:
    print(item)

print (itemindex)

print(a)