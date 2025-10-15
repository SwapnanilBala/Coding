# a_tuple = (1,4,2,5,3,6,7,3,4,8,3,5,6,2,8)
from functools import lru_cache
from time import time
# for j in range (0,len(a_list)-1):
#     min_pos = j
#     for k in range(j+1,len(a_list)):
#         if a_list[k] > a_list[min_pos]:
#             a_list[k], a_list[min_pos] = a_list[min_pos], a_list[k]
#
# print(a_list[::-1])


# for index,value in enumerate(a_list):
#     i = 0
#     if i == index:
#         i +=1
#         pass
#     else:
#         if value > a_list[i]:
#             a_list[i],value = value,a_list[i]
#
#
# print(a_list)
# empty_list = []
#
# for i in range(len(a_list)-1):
#     minimum = min(a_list)
#     empty_list.append(minimum)
#     for i in range()
# print(empty_list)

# Selection Sort


# from time import time
# from functools import lru_cache
#
# def swap(arr, i, j):
#     arr[i], arr[j] = arr[j], arr[i]
#
# t_1 = time()
#
# @lru_cache(maxsize=64)
# # we must use tuple, unless we want that stupid unhashable list type
# def selection_sort(n):
#     arr = list(n)  # Converting to list for better handling, because swapping is not allowed in tuples
#     for i in range(0, len(arr)-1):
#         for j in range(i+1, len(arr)):
#             if arr[i] > arr[j]:
#                 swap(arr, i, j)
#     return tuple(arr)
#
# t_2 = time()
#
# a_list = (5, 2, 9, 1, 7,14,2,16,41,16,17,8,46)
# print(selection_sort(a_list))
# print("Time Taken:", t_2-t_1)

# Merge Sort through Divide and Conqueror Method

a_list = [5,2,3,7,1,8,2,3,8,4,2,7,2,8,23,1]

# for now, we will only write the code for a list that has even numbers of integers

try:
    cal = len(a_list)
    if cal%2 == 0:
            first_half = [a_list[i] for i in range(0,cal,2)]
            second_half = [a_list[i+1] for i in range(0,cal,2)]
    else:
        print("Odd number of items sorry can't do shit")
finally:
    print("I am confused")
    print(first_half)
    print("")
    print(second_half)

