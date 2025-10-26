# def fibonacci(num:int):
#     if num == 1:
#         return 0
#     elif num == 2:
#         return 1
#     else:
#         return fibonacci(num-1) + fibonacci(num-2)
#
# if __name__ == "__main__":
#     print(fibonacci(5))
#     series = [fibonacci(i) for i in range(1,5+1)]
#     # We can replace the value of 5 with n or something like that
#     print(series)
from functools import lru_cache
import random
from time import time

from numpy.ma.core import swapaxes


## fibonacci with DP

# @lru_cache(maxsize=64)
# def fibo_dp(n:int)-> int:
#
#     if n == 0:
#         return 0
#     elif n == 1:
#         return 1
#     else:
#         f = [0] * (n+1)
#         f[0], f[1] = 0, 1
#         for i in range(2,n+1):
#             f[i] = f[i-1] + f[i-2]
#     return f[n]
#
# if __name__ == "__main__":
#     print(fibo_dp(6))


# def mergeSort(arr:list):
#     if len(arr) <= 1:
#         return arr
#
#     mid = len(arr)//2
#     left_side = arr[:mid]
#     right_side = arr[mid:]
#
#     left_side = mergeSort(left_side)
#     right_side = mergeSort(right_side)
#
#     return merge(left_side,right_side)
#
#
# def merge(left,right):
#     i=k=0
#     res = []
#     left_len = len(left)
#     right_len = len(right)
#     while i < left_len and k < right_len:
#         if left[i] < right[k]:
#             res.append(left[i])
#             i+= 1
#         else:
#             res.append(right[k])
#             k+= 1
#
#     while i < left_len:
#         res.append(left[i])
#         i+= 1
#     while k < right_len:
#         res.append(right[k])
#         k+= 1
#
#
#     return res
#
#
#
# if __name__ == "__main__":
#     t1 = time()
#     an_array = [random.randint(1,123) for _ in range(12)]
#     print(an_array)
#
#     res = mergeSort(an_array)
#     print(res)
#
#     t2 = time()
#     print(f"Time taken for this code completion is : {t2 - t1}")

# QuickSort Practice

import random

# class QuickSort:
#
#     def partition(self,array,low,high):
#         kingpin = array[high]
#         i = low - 1
#         for j in range(low,high):
#             if array[j] < kingpin:
#                 i += 1
#                 self.swap(array,i,j)
#         self.swap(array,i+1,high)
#
#         return i+1
#
#     def swap(self,array,a,b):
#         array[a],array[b] = array[b],array[a]
#
#
#     def quickSort(self,array,low,high):
#         if len(array) <= 1:
#             return array
#
#         if low < high:
#             pi = self.partition(array,low,high)
#             self.quickSort(array,low,pi-1)
#             self.quickSort(array,pi+1,high)
#
#         return array
#
#
# if __name__ == "__main__":
#     sol = QuickSort()
#     an_array = [random.randint(1,16) for x in range(22)]
#     l = 0
#     h = len(an_array)-1
#     print(sol.quickSort(an_array,l,h))

def gen():
    for n in range(100):
        yield n


numbers = gen()
print(next(numbers))
print(50 in numbers)
print(next(numbers))













