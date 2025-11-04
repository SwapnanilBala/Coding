# def partition(array, low, high):
#     kingpin = array[high]
#     i = low - 1
#     for j in range(low, high):
#         if array[j] < kingpin:
#             i = i + 1
#             quickswap(array, i, j)
#     quickswap(array, i+1, high)
#     return i+1
#
#
#
# def quickswap(array, start, end):
#     array[start], array[end] = array[end], array[start]
#
#
# def quicksort(array, low, high):
#     if low < high:
#         pi = partition(array, low, high)
#         quicksort(array, low, pi-1)
#         quicksort(array, pi+1, high)
#     return array
#
#
# # if __name__ == '__main__':
# #     arr = list(map(int,input().split()))
# #     quicksort(arr, 0, len(arr)-1)
# #     print(arr)
#
#
# # Let's do some Breadth First Search Shortest Distance
#
# from collections import deque
#
# def bfs_shortest_path(graph, start, end):
#     visited = set()
#     queue = deque([(start, [start])])  # Store tuples of (current_node, path)
#
#     while queue:
#         node, path = queue.popleft()  # Get the current node and the path taken
#
#         if node == end:
#             return path  # Return the path if we reached the end
#
#         for neighbor in graph[node]:
#             if neighbor not in visited:
#                 visited.add(neighbor)  # Mark the neighbor as visited
#                 # Append the neighbor and the new path to the queue
#                 queue.append((neighbor, path + [neighbor]))
#
#     return None  # Return None if no path found
#
# if __name__ == '__main__':
#     graph = {
#         'A': ['B', 'C'],
#         'B': ['A', 'D', 'E'],
#         'C': ['A', 'F'],
#         'D': ['B'],
#         'E': ['B', 'F'],
#         'F': ['C', 'E']
#     }
#
#     start_node = 'A'
#     end_node = 'F'  # Specify the end node
#     path = bfs_shortest_path(graph, start_node, end_node)
#     print(f"Shortest path from node '{start_node}' to '{end_node}': {path}")
from functools import lru_cache
from inspect import stack

# def two_sum(a_list: list, target: int) -> list:
#     result_set = set()
#     for i in range(len(a_list)-1):
#         complement = target - a_list[i]
#         for index, value in enumerate(a_list[i::]):
#             if value == complement:
#                 result_set.add(tuple(sorted([i,index])))
#
#     return list(result_set)
#
#
# if __name__ == "__main__":
#     print(two_sum([1,2,3,4,5,6,7,8,9], 10))


# from functools import lru_cache
#
# @lru_cache(maxsize=64)
# def two_sum(a_tuple: tuple, target: int) -> list:
#     a_list = list(a_tuple)  # convert back to list if needed
#     result_set = set()
#     for i in range(len(a_list)-1):
#         complement = target - a_list[i]
#         for index, value in enumerate(a_list[i:]):
#             if value == complement:
#                 result_set.add(tuple(sorted([i, i+index])))  # careful with indices
#
#     return list(result_set)
#
#
# if __name__ == "__main__":
#     print(two_sum(tuple([1,2,3,4,5,6,7,8,9]), 10))

# from time import time
# from functools import lru_cache
#
# @lru_cache(maxsize=64)
# def reversal(num: int) -> int:
#     if num == 0:
#         return 0
#     elif num > 2**31:
#         return 0
#     elif num < 2**31:
#         return 0
#     elif num < 0:
#         str_con = str(num)[1::]
#         rev = str_con[::-1]
#         output =""
#         for i in range(len(rev)):
#             if rev[i] == "0":
#                 i+= 1
#             else:
#                 output += rev[i::]
#                 break
#         return -1 * int(output)
#     else:
#         str_con = str(num)
#         rev = str_con[::-1]
#         output = ""
#         for i in range(len(rev)):
#             if rev[i] == "0":
#                 i += 1
#             else:
#                 output += rev[i::]
#                 break
#         return  int(output)
#
#
# if __name__ == "__main__":
#     t1 = time()
#     print(reversal(-1203400100))
#     t2 = time()
#     print(f"Time taken: {t2 - t1}")


# from functools import lru_cache
#
# @lru_cache(maxsize=64)
# def process_nums(nums: list[int]) -> list[int]:
#     # Ensure nums is hashable for caching
#     uniques = set()
#     duplicates = 0
#     for item in sorted(nums):
#         if item in uniques:
#             duplicates += 1
#         else:
#             uniques.add(item)
#
#     output = []
#     for items in uniques:
#         output.append(items)
#     for _ in range(duplicates):
#         output.append('_')
#
#     return output
#
#
# if __name__ == "__main__":
#     nums = [3, 1, 2, 2, 3, 1]
#     print(process_nums(nums))  #

# Selection Sort

## WRONG SOLUTION ##

# random_list = [5,2,7,3,9,1,4,2,46,2,35,27,25,0]

# reps = 0
# for _ in range(len(random_list)-2):
#     reps = reps + 1
#     first = random_list[reps]
#     target = min(random_list[reps+1::])
#     if first > target:
#         swap(first,target)
#
# print(random_list)

# def swap(a_list,a,b):
#     a_list[a],a_list[b] = a_list[b],a_list[a]
#
#
# def selection_sort(a_list : list):
#         for i in range(len(a_list)-1):
#             target = a_list[i]
#             target_index = a_list.index(target)
#             minimum = min(a_list)
#             min_index = a_list.index(minimum)
#             if target > minimum:
#                 swap(a_list,target_index,min_index)
#
#         return a_list
#
# if __name__ == "__main__":
#     print(selection_sort(random_list))

## WRONG SOLUTION ##

# True Solution #
## Selection Sort ##
#
# def swap(a_list, a, b):
#     a_list[a], a_list[b] = a_list[b], a_list[a]
#
# def selection_sort(a_list: list):
#     n = len(a_list)
#     for i in range(n - 1):
#         # Find index of the smallest element in the unsorted part
#         min_index = i
#         for j in range(i + 1, n):
#             if a_list[j] < a_list[min_index]:
#                 min_index = j
#         # Swap if a smaller element is found
#         if min_index != i:
#             swap(a_list, i, min_index)
#     return a_list
#
# if __name__ == "__main__":
#     random_list = [5, 3, 8, 1, 2]
#     print(selection_sort(random_list))


# def quick_swap(a_list, a, b):
#         a_list[a], a_list[b] = a_list[b], a_list[a]
#
#
# def mergesort_even_numbered_array(an_array):
#     sorted_list = []
#     for i in range(0,len(an_array),2):
#         pairs = [an_array[i],an_array[i+1]]
#         left = pairs[0]
#         right = pairs[1]
#         if left > right:
#             quick_swap(pairs,0,1)
#         sorted_list.append(pairs)
#
#     return sorted_list
#
# def merge_them(the_array):
#         left = [items[0] for items in the_array]
#         right = [items[1] for items in the_array]
#         result_array = []
#         min_left = left.pop(min(left))
#         min_right = right.pop(min(right))
#         if min_left > min_right:
#             result_array.append(min_right)
#             result_array.append(min_left)
#
#         return result_array
#
# unique_list = [4,1,5,2,17,9,8,6,14,23,16,12]
#
# print(mergesort_even_numbered_array(unique_list))
# print(merge_them(mergesort_even_numbered_array(unique_list)))

import numpy as np
import random


## Quick Sort ##

# def quick_swap(arr, a, b):
#     arr[a],arr[b]= arr[b],arr[a]
#
#
# def partition(arr,low,high) -> int:
#     kingpin = arr[high]
#     x = low - 1
#     for y in range(low,high):
#         if arr[y] < kingpin:
#             x+= 1
#             quick_swap(arr,x,y)
#
#     quick_swap(arr,x+1,high)
#
#     return x + 1
#
# def quicksort(arr: list,low,high)-> list:
#     if low < high:
#
#         pi = partition(arr,low,high)
#
#         quicksort(arr,low,pi-1)
#         quicksort(arr,pi+1,high)
#
#     return arr
#
# if __name__ == '__main__':
#     # unordered_list = [np.random.randint(1,14,10)]
#     ymca = [6,2,7,3,8,1,9,0]
#     quicksort(ymca,0,len(ymca)-1)
#     print(ymca)


# from functools import lru_cache
# from time import time
#
#
# t1 = time()
#
# def quickswap(arr, a, b):
#     # This function works directly on the list
#     arr[a], arr[b] = arr[b], arr[a]
#
#
# def partition(arr, low, high) -> int:
#     pivot = arr[high]
#     i = low - 1
#     for j in range(low, high):
#         if arr[j] < pivot:
#             i += 1
#             quickswap(arr, i, j)
#     quickswap(arr, i + 1, high)
#     return i + 1
#
#
# def quicksort_recursive(arr, low, high):
#     if low < high:
#         pi = partition(arr, low, high)
#         quicksort_recursive(arr, low, pi - 1)
#         quicksort_recursive(arr, pi + 1, high)
#
#
# def sort_tuple(a_tuple) -> tuple:
#     # 1. Convert the tuple to a list
#     mutable_list = list(a_tuple)
#
#     # 2. Sort the list in place
#     quicksort_recursive(mutable_list, 0, len(mutable_list) - 1)
#
#     # 3. Convert the sorted list back to a tuple and return it
#     return tuple(mutable_list)
#
# t2 = time()
#
# if __name__ == "__main__":
#     an_array = (7, 6, 5, 4, 3, 2, 1, 0)
#     sorted_array = sort_tuple(an_array)
#     print(f"Original tuple: {an_array}")
#     print(f"Sorted tuple:   {sorted_array}")
#     print(f'time taken: {t2-t1}')

# def swap(random_list,a,b):
#     random_list[a],random_list[b] = random_list[b], random_list[a]
#
#
#
# def selection_sort(a_list):
#     n = len(a_list)
#     for j in range(0,n-1):
#         min_pos = j
#         for k in range(j+1,n):
#             if a_list[k] < a_list[j]:
#                 min_pos = k
#         swap(a_list,min_pos,j)
#
#     return a_list
#
# if __name__ == "__main__":
#     sp_list = [5,4,3,2,1,0]
#     print(selection_sort(sp_list))

# def inversions(arr: list) -> list:
#     unique_duos = set()
#     for i in range(0,len(arr)-2,1):
#         for j in range(i,len(arr)-1,1):
#             if i < j and arr[i] > arr[j]:
#                 unique_duos.add(tuple(sorted([arr[i], arr[j]])))
#
#
#     return list(unique_duos)
#
#
# if __name__ == "__main__":
#     a_list = [1,4,2,3,1]
#     print(inversions(a_list))


# def inversion_shorter(arr:list) -> list:
#     unique_combos = set()
#     for index, value in enumerate(arr):
#         for _ in arr:
#             if value > _ and index < arr.index(_):
#                 unique_combos.add(tuple(sorted([value,_])))
#
#     return list(unique_combos)
#
# if __name__ == "__main__" :
#     arra = [2,4,1,5,2,6,7,3,1]
#     print(inversion_shorter(arra))


# def inversion_counter(array: list) -> int:
#     n = len(array)
#     counter = 0
#     for i in range(0,n-1,1):
#         for j in range(n-1,0,-1):
#             if i == j:
#                 pass
#             else:
#                 if i  < j and array[i] > array[j]:
#                     counter+= 1
#
#     return counter
#
# if __name__ == "__main__":
#     a_list = [2,5,3,1,4,6,7,2,3,8,3]
#     print(inversion_counter(a_list))


# def inversion_counter_ezz(arra: list) -> int:
#     n = len(arra)
#     counter = 0
#
#     for x in range(n // 2):
#         y = n - 1 - x
#         if arra[x] < arra[y]:
#             counter += 1
#
#     return counter
#
#
# if __name__ == "__main__":
#     arr = [5, 2, 3, 7, 1, 6, 8, 2, 9, 4, 2, 10]
#     print(inversion_counter_ezz(arr))

# l = [0] * 4
# print(l)

# def good_pairs(arr:list, target:int) -> int:
#     counter = 0
#     seen = set()
#
#     for value in arr:
#         complement = target - value
#         if complement in seen:
#             counter += 1
#         seen.add(value)
#
#     return counter
#
# if __name__ == "__main__":
#     numbers = [3,2,5,4,1,6,2,3,5,7,3,7,4,5,6,2,8,4,5,6]
#     print(good_pairs(numbers,6))

# def mergesort(arr):
#     if len(arr) <= 1:
#         return arr
#
#     mid = len(arr) // 2
#     left = arr[:mid]
#     right = arr[mid:]
#
#     sorted_left = mergesort(left)
#     sorted_right = mergesort(right)
#
#     return merge(sorted_left,sorted_right)
#
#
# def merge(left:list , right:list):
#     result = []
#     i = j = 0
#
#     while i < len(left) and j < len(right):
#         if left[i] < right[j]:
#             result.append(left[i])
#             i += 1
#         else:
#             result.append(right[j])
#             j += 1
#
#     result.extend(left[i:])
#     result.extend(right[j:])
#
#     return result
#
#
# if __name__ == "__main__":
#     unsortedArr = [3, 7, 6, -10, 15, 23.5, 55, -13]
#     sortedArr = mergesort(unsortedArr)
#     print("Sorted array:", sortedArr)

# knap_sack Problem

# def ks_dpp(max_cap: int, stuff: list):
#     max_two_combos = []
#     seen = set()
#     for index, value in enumerate(stuff):
#         if value > max_cap:
#             continue
#         else:
#             remaining = max_cap - value
#             if remaining in seen:
#                 continue
#             else:
#                 for j in stuff[index:]:
#                     if j <= remaining:
#                         seen.add(j)
#                         max_two_combos.append(tuple(sorted([value, j])))  # FIXED
#                     else:
#                         continue
#     return set(max_two_combos)
# 
# 
# if __name__ == "__main__":
#     cap = 7
#     bagged_items = [4, 2, 3, 5, 1, 0]
#     print(ks_dpp(cap, bagged_items))

# merge_sort algorithm

# def mergesort(array):
#     # The base case is that if the array has 1 or 0 elements, it's already sorted.
#     if len(arr) <= 1:
#         return arr
#
#     # finding the middle point
#     mid = len(arr)//2
#
#     # divide the array into two halves
#
#     left = array[:mid] # start to middle
#     right = array[mid:] # from middle to the end
#
#     # recursively sort both halves, down you see
#
#     left = mergesort(left)
#     right = mergesort(right)
#
#     # the following above are the recursions
#
#     return merge(left, right)
#
# def merge(left,right):
#     result = []
#     i = 0 # this is the pointer for the left array
#     j = 0 # this is the pointer for the right array
#
#     # Now we are comparing from left to right, adding smaller one to the result
#     while i < len(left) and j < len(right):
#         if left[i] <= right[j]: # this one makes sure to check the comparable elements before adding
#             result.append(left[i])
#             i +=1
#         else:
#             result.append(right[j])
#             j += 1
#
#     while i < len(left):
#         result.append(left[i])
#         i+=1
#
#     while j < len(right):
#         result.append(right[j])
#         j+= 1
#
#     return result
#
#
# if __name__ == "__main__":
#     arr = [38, 27, 43, 3, 9, 82, 10, 5, 2, 7, 4,11]
#     sorted_arr = mergesort(arr)
#     print(sorted_arr)

# the knapsack problem

# def knapsack(valuables: dict, capacity: int):
#     value_ratios = []
#     i = 0
#     records = dict()
#     for i in range(0,len(valuables)):
#         key = list(valuables.keys())[i]
#         val = valuables[key]
#         ratio = val/key
#         value_ratios.append(ratio)
#
#     j = 0
#     while j <= len(valuables)-1:
#         records[j] = value_ratios[j]
#         j+= 1
#
#     priority_list = []
#     reverse_check = {a:b for b,a in records.items()}
#
#     total_weight = 0
#     total_value = 0
#
#     rev_sorted_values = (sorted(reverse_check.keys()))[::-1]
#
#     while total_weight <= capacity:
#         i = 0
#         total_value += rev_sorted_values[i]
#         total_weight += rev_sorted_values[i]
#         while total_weight > capacity:
#             total_value -= rev_sorted_values[i]
#             total_weight -= rev_sorted_values[i]
#             i += 1
#
#
#
#
#     return [total_weight,total_value]
#
#
# if __name__ == "__main__":
#     w_val = {3:500, 1: 200, 4:600, 2:100, 5: 1000 }
#     print(knapsack(w_val,13))

# def knapsack(valuables: dict, capacity: int):
#     # Computing Values and Weight Rations
#     val_ratio = {}
#     for weight,value in valuables.items():
#         ratio = value/ weight
#         val_ratio[weight] = ratio
#
#     # Sorting by ration from high to low
#     sorted_items = sorted(val_ratio.items(), key = lambda x:x[1],reverse = True)
#
#     # will output these in res
#     total_weight = 0
#     total_value = 0
#
#     for weight,ratio in sorted_items:
#         if total_weight + weight <= capacity:
#             total_weight += weight
#             total_value += valuables[weight]
#
#         else:
#             break
#
#     return [total_weight,total_value]
#
# if __name__ == "__main__":
#     w_val = {3: 500, 1: 200, 4: 600, 2: 100, 5: 1000}
#     print(knapsack(w_val, 13))

# Some random Leet-Code Grind



# @lru_cache(maxsize=64)
# def Max_Diff_Evn_Odd(a_string) -> int :
#     uni_alph = set()
#     word_counter = dict()
#     for char in a_string:
#         if char not in uni_alph:
#             uni_alph.add(char)
#             word_counter[char] = 1
#         else:
#             word_counter[char] += 1
#     rev_wc = {a:b for  b,a in word_counter.items()}
#
#     a = max(key for key, val in rev_wc.items() if key % 2 == 0)
#     b = max(key for key, val in rev_wc.items() if key % 2 != 0)
#     return abs(a-b)
#
# if __name__ == "__main__":
#     night_wing = "abcabcab"
#     print(Max_Diff_Evn_Odd(night_wing))


# class Solution:
#     lru_cache(maxsize=128)
#     def strStr(self, haystack: str, needle: str) -> int:
#         for i in range(len(haystack) - len(needle) + 1):
#             word = haystack[i:i+len(needle)]
#             if word == needle:
#                 return i
#
#         return -1
#
# if __name__ == "__main__":
#     sol = Solution()
#     print(sol.strStr("IamGogol","Gogol"))

# Insertion sort on a sorted array

# import numpy as np
#
# def Ins_srt(array: list, target: int) -> int:
#     left = 0
#     right = len(array) - 1
#
#     while left <= right:
#         mid = (left + right) // 2
#         if array[mid] == target:
#             return mid
#         elif array[mid] > target:
#             right = mid - 1
#         else:
#             left = mid + 1
#
#     return -1
#
#
# if __name__ == "__main__":
#     array = [1,2,3,4,5,6,7,8,9]
#     print(Ins_srt(array, 7))


# def twoSum( nums, target):
#     for index, value in enumerate(nums):
#         for i in range(index + 1, len(nums)):
#             if value + nums[i] == target:
#                 return [index, i]
#
#     return None
#
#
# if __name__ == "__main__":
#     arr = [4,2,1,3,5,2,6,3,6,4,5,2,9,3,6]
#     tar = 5
#     print(twoSum(arr,tar))

# Roman to Integer

# symbol_and_meaning = {
#                     "I": 1,
#                     "V": 5,
#                     "X": 10,
#                     "L": 50,
#                     "C": 100,
#                     "D": 500,
#                     "M": 1000
#                       }
# def roman_conv(a_wrd:str):
#     res = 0
#     prev_value = 0
#     for i in a_wrd[::-1]:
#         val = symbol_and_meaning[i]
#         if val < prev_value:
#             res -= val
#         else:
#             res += val
#         prev_value = val
#     return res
#
# if __name__ == "__main__":
#     word = "MCXIV"
#     print(roman_conv(word))

# Longest Common Prefix


# @lru_cache(maxsize = 32)
# def longest_common_prefix(low:tuple):
#     our_list = [low]
#     n = len(low)
#     uniques = set()
#     uniques_count = dict()
#     for items in our_list:
#         for char in items:
#             if char not in uniques:
#                 uniques_count[char] = 1
#                 uniques.add(char)
#             else:
#                 uniques_count[char] += 1


# Time Complexity == O(n^4) and Space Complexity == O(n^4), classic 4-sum problem

# def four_sum(nums, target):
#     l = len(nums)
#     ans = set()
#     if l >= 4:
#         for i in range(0,l):
#             for j in range(i+1,l):
#                 for k in range(j+1,l):
#                     for o in range(k+1,l):
#                         if nums[i]+nums[j]+nums[k]+nums[o] == target:
#                             ans.add(tuple(sorted([nums[i],nums[j],nums[k],nums[o]])))
#         if len(ans) > 0:
#             return [[x] for x in ans]
#
#     return None
#
# if __name__ == "__main__":
#     numbers = [4,2,3,1,5,2,-2,6,2,3,-1,7,8]
#     print(four_sum(numbers, 8))



# Time Complexity == O(n^3) and Space Complexity == O(n^3), Modified 4-sum problem


# def four_sum(nums: list[int], target: int)-> list:
#     abso = sorted(nums)
#     res = set()
#     l = len(nums)
#     for i in range(0, l-4):
#         for j in range(i+1, l-3):
#             left = j+1
#             right = l-1
#             complement = target - (abso[i]+abso[j])
#             while left < right:
#                 current_num = abso[left] + abso[right]
#                 if  current_num == complement:
#                     res.add(tuple(sorted([abso[i],abso[j],abso[left],abso[right]])))
#                     left += 1
#                 elif current_num  > complement:
#                     right -= 1
#                 else:
#                     left += 1
#
#             return [list(x) for x in res]
#
#     return [None]
#
# if __name__ == "__main__":
#     an_array = [1,2,3,4,5,6,7,8,9,10,-5,-4,-3,-2,-1,0]
#     tar = 6
#     print(four_sum(an_array,tar))

# @lru_cache(maxsize=32)
# def fibo_with_dp(nth_element:int)-> int:
#     if nth_element == 1:
#         return 0
#
#     elif nth_element == 2:
#         return 1
#
#     else:
#         dp_series = [0] * nth_element
#         dp_series[0] = 0
#         dp_series[1] = 1
#         for i in range(2,nth_element):
#             dp_series[i] = dp_series[i-1] + dp_series[i-2]
#
#         return dp_series[nth_element-1]
#
# if __name__ == "__main__":
#     print(fibo_with_dp(7))

# def fourSum(nums: list[int], target: int) -> list[list[int]]:
#     nums.sort()
#     l = len(nums)
#     unique_combos = []
#
#     for i in range(0, l - 4):
#         if i > 0 and nums[i] == nums[i-1]:
#             continue
#         for j in range(i + 1, l - 3):
#             if nums[j] == nums[j-1]:
#                 continue
#
#             left = j + 1
#             right = l - 1
#             while left < right:
#                 complement = target - (nums[i] + nums[j])
#                 summation = nums[left] + nums[right]
#                 if summation == complement:
#                     unique_combos.append(sorted([nums[i], nums[j], nums[left], nums[right]]))
#                     left += 1
#                 elif summation > complement:
#                     right -= 1
#                 else:
#                     left += 1
#
#     return [list(x) for x in unique_combos]
#
# if __name__ == "__main__":
#     nums_ = [1,0,-1,0,-2,2]
#     target = 0
#     print(fourSum(nums_,target))


# longest Common Prefix between the words in a given list

# def longest_common_prefix(words: list)-> str:
#     first_word = words[0]
#     last_word = words[-1]
#     target = ""
#     for i in range(0,len(first_word)):
#         for j in range(i+1,len(last_word)):
#             if first_word[i:j] == last_word[i:j]:
#                 curr = first_word[i:j]
#                 if len(curr) > len(target):
#                     target = curr
#
#     if len(target) > 1:
#         return target
#
#     return "No Matching Strings"
#
# if __name__ == "__main__":
#     wrds = ["flower","flow","flight","chafli","chifli"]
#     print(longest_common_prefix(wrds))


# longest common prefix

# def longest_common_prefix(words:list):
#     smallest_word = words[0]
#     substr = ""
#     for wrds in words:
#         if len(wrds) < len(smallest_word):
#             smallest_word = wrds
#     for x in words:
#         if x == smallest_word:
#             continue
#         else:
#             for u in x:
#                 if smallest_word in x:
#                     return smallest_word
#
#                for i in range(0,len(smallest_word,1)):
#                        curr = smallest_word[i:j]

# def removeDuplicates(nums:list[int]):
#     l = len(nums)
#     dp_list = [0] * (l)
#     uniques = set()
#     i = 0
#     j = (l - 1)
#     for items in nums:
#         if items not in uniques:
#             uniques.add(items)
#             dp_list[i] = items
#             i+= 1
#         else:
#             dp_list[j] = "_"
#             j-= 1
#
#     k = (len(nums)-i)
#
#     return k, dp_list
#
#
# if __name__ == "__main__":
#     arr = [0,0,1,1,1,2,2,3,3,4]
#     print(removeDuplicates(arr))

# def removeElement(nums: list[int], val:int):
#     l = len(nums)
#     dp_l = [0] * l
#     i = 0
#     triggers = 0
#     for items in nums:
#         if items != val:
#             dp_l[i] = items
#             i+= 1
#         else:
#             dp_l.remove(items)
#             dp_l.append("_")
#             i+= 1
#
#     return dp_l
#
# if __name__ == "__main__":
#     nums = [0, 1, 2, 2, 3, 0, 4, 2]
#     val = 2
#     print(removeElement(nums,val))

# def plusOne(digits: list[int]) -> list[int]:
#     num_str = ""
#     res = []
#     for x in digits:
#         num_str += str(x)
#
#     num = int(num_str) + 1
#     num_str = str(num)
#     res = [int(x) for x in num_str]
#
#     return res
#
# if __name__ == "__main__":
#     num = [1,2,3,4]
#     print(plusOne(num))

# def addBinary(a: str, b: str) -> str:
#     res = ""
#     A, B = len(a), len(b)
#
#     # pad shorter string
#     if A > B:
#         b = "0" * (A - B) + b
#     elif B > A:
#         a = "0" * (B - A) + a
#
#     carry = 0
#     for i in range(len(a) - 1, -1, -1):
#         first = int(a[i])
#         second = int(b[i])
#
#         total = first + second + carry
#         res += str(total % 2)
#         carry = total // 2
#
#     if carry:
#         res += "1"
#
#     return res[::-1]
#
# if __name__ == "__main__":
#     u = "101011"
#     v = "00101"
#     print(addBinary(u,v))

# Pascals Triangle

# def singleNumber(nums: list[int]) -> int:
#     hash_dict = {}
#     for num in nums:
#         hash_dict[num] = hash_dict.get(num, 0) + 1
#
#     rev_dict = {v: k for k, v in hash_dict.items()}
#     return rev_dict[1]
#
# if __name__ == "__main__":
#     numbers = [1,2,3,4,5,4,3,2,5,2,6,434,3,6,3,4,1]
#     print(singleNumber(numbers))

# class Solution:
#     def isHappy(self,n:int) -> bool:
#         tracker = 0
#         while tracker < 8:
#             tracker += 1
#             goal = self.conv(n)
#             if goal == 1:
#                 return True
#             n = goal
#
#         return False
#
#
#     def conv(self,num):
#         res = sum(pow(int(x),2) for x in str(num))
#         return res
#
#
# if __name__ == "__main__":
#     sol = Solution()
#     n = 19
#     print(sol.isHappy(n))

# Valid Palindrome
# def palindrome(s: str) -> bool:
#     ln = len(s)
#     if ln == 1:
#         return True
#
#     for i in range(ln // 2):
#         if s[i] != s[ln - 1 - i]:
#             return False
#
#     return True
#
#
# if __name__ == "__main__":
#     word = "racecar"
#     print(palindrome(word))

import re

from jsonschema.benchmarks.const_vs_enum import value

from Insertion_Sort_Practice import target_index_finder
from Random_Practice import Solution
import random

# Special Palindrome Check
#
# class Soju:
#     import re
#     def isPalindrome(self,s:str)-> bool:
#         # Cleaning the String First
#         s = re.sub(r'[^A-Za-z0-9]','',s).lower()
#         # Assigning a pointer to the length of our new string
#         length = len(s)
#         if length <= 1:
#             return True
#
#         for i in range(0,length//2):
#             if s[i] != s[length - 1 - i]:
#                 return False
#
#         return  True
#
# if __name__ == "__main__":
#     sol = Soju()
#     ward = "aabbccbbaa"
#     print(sol.isPalindrome(ward))

# Depth-First_Search


# graph = {
#     1: [2, 3],
#     2: [6],
#     3: [4, 5],
#     4: [5],
#     5: [8],
#     6: [],
#     7: [8],
#     8: [7]  # (example directed/undirected depends on edges you add)
# }
#
# def dfs_no_recursion(graph, start):
#     visited = set()
#     order = []
#     stacking = [start]
#     while stacking:
#         node = stacking.pop()
#         if node in visited:
#             continue
#         visited.add(node)
#         order.append(node)
#         # dict.get(key, default if not found)
#         for neigh in reversed(graph.get(node, [])):
#             if neigh not in visited:
#                 stacking.append(neigh)
#
#     return order
#
# print(dfs_no_recursion(graph, 1))


# This is a wrong solution

# class Solution:
#     def myPow(self,x:float,n:int)-> float:
#         while n >=0:
#             output = x**n
#             return output
#
#         y = (1/x)
#         output = y**n
#         return output
#
# if __name__ == "__main__":
#     sol = Solution()
#     print(sol.myPow(-2.100,-3))

# 3Sum Closest

# class Three_sum_closest:
#         def threeSumClosest(self, nums: list[int], target: int) -> int:
#             nums.sort()
#             closest_ori = 99999
#             total = 0
#             the_total = 0
#             l = len(nums)
#             for i in range(l - 2):
#                 left = i + 1
#                 right = l - 1
#                 while left < right:
#                     total = nums[i] + nums[left] + nums[right]
#                     closest_curr = abs(total - target)
#                     if closest_curr < closest_ori:
#                         closest_ori = closest_curr
#                         the_total = total
#                     if total < target:
#                         left += 1
#                     else:
#                         right -= 1
#
#             return the_total
#
# if __name__ == "__main__":
#         sol = Three_sum_closest()
#         print(sol.threeSumClosest([1,2,3,4,5,7,6],7))


# Threesum Smaller Algorithm

# class ThreeSumCloser:
#
#     def threeSumSmaller(self, nums: list[int], target: int) -> int:
#         nums.sort()
#         l = len(nums)
#         spikes = 0
#         for i in range(l-2):
#             left = i+1
#             right = l - 1
#             while left < right:
#                 total = nums[i] + nums[left] + nums[right]
#                 if total < target:
#                     spikes += (right-left)
#                     left += 1
#                 else:
#                     right-= 1
#
#
#         return spikes
#
# if __name__ == "__main__":
#     sol = ThreeSumCloser()
#     array_n = [random.randint(-2,4) for x in range(5)]
#     print(f"The array looks like {array_n}")
#     tar = 5
#     print(sol.threeSumSmaller(array_n,tar))


# class solution:
#     def numSubArrayProductLessThanK(self,num: list[int], k: int) ->int:
#         if k<= 1:
#             return 0
#
#         prod = 1.0
#         ans = 0
#         left = 0
#
#         for right in range(len(num)):
#             prod *= num[right]
#
#             while prod >= k and left <= right:
#                 prod /= num[left]
#                 left += 1
#
#             ans += (right - left + 1)
#
#         return ans

# I did not understand the above question, let's not include that for understanding

# shortest_path using bfs,

from collections import deque

# def bfs_shortest_path(graph,start,end):
#     traversed = set()
#     queue = deque([start])
#     while queue:
#         path = queue.popleft()
#         node = path[-1]
#
#         if node == end:
#             return path
#
#         if node not in traversed:
#             traversed.add(node)
#             for adjacent_nodes in graph.get(node,[]):
#                 new_path = list(path)
#                 new_path.append(adjacent_nodes)
#                 queue.append(new_path)
#
#     return None
#
# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'E': ['F'],
#     'F': ['G']
# }
#
# print(bfs_shortest_path(graph, 'A', 'G'))

# Longest Common Sequence


# failed attempt

# def lcs(word_1:str, word_2:str)-> str:
#     small_word = min(word_1,word_2)
#     big_word = max(word_1,word_2)
#     longest = 0
#     sub_word = ""
#     for index, value in enumerate(small_word):
#         j = 0
#         while small_word[j] == big_word[j]:
#             j+= 1
#         j+= 1
#         if j - index > longest:
#             longest = (j-index)
#             sub_word = small_word[index:j+1]
#
#     return sub_word
#
# if __name__ == "__main__":
#     word_one = 'racecar'
#     word_two = 'suscar'
#     print(lcs(word_one,word_two))


# Longest Commmon Sequence

# Didn't Understand a thing with this one

# def lcs(s1: str, s2: str) -> str:
#     l1, l2 = len(s1), len(s2)
#     dp = [[0] * (l1 + 1) for _ in range(l2 + 1)]
#
#     # Fill DP table
#     for i in range(1, l2 + 1):
#         for j in range(1, l1 + 1):
#             if s2[i - 1] == s1[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1] + 1
#             else:
#                 dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
#
#     # Reconstruct LCS
#     i, j = l2, l1
#     out = []
#     while i > 0 and j > 0:
#         if s2[i - 1] == s1[j - 1]:
#             out.append(s2[i - 1])
#             i -= 1
#             j -= 1
#         elif dp[i - 1][j] >= dp[i][j - 1]:
#             i -= 1
#         else:
#             j -= 1
#
#     return "".join(reversed(out))
#
#
# # Test
# print(lcs("abcdef", "acbcf"))      # -> "abcf"
# print(lcs("AGGTAB", "GXTXAYB"))    # -> "GTAB"

# Selection Sort:
# def ss(array:list[int]):
#     for i in range(len(array)-1):
#         for j in range(i+1, len(array)):
#             if array[i] > array[j]:
#                 swap(array,i,j)
#
#     return array
#
#
# def swap(array,a,b):
#     array[b],array[a] = array[a], array[b]
#
#
# if __name__ == "__main__":
#     arr = [7,6,5,2,4,3,1,7,4,5,2,8,5,3,6,2,0]
#     print(ss(arr))

# import numpy as np
#
# list_1 = [1,2,3,4,5]
# 
# list_2 = np.array([1,2,3,4,5])
#
# print(f"normal multiplication : {list_1*5}, and vector multiplicaton {list_2*5}" )

# In this tab we will try to replicate the mergeSort Algorithm, lets hit it
# Also going to import numpy to make sure to stress the shit out of this algorithm
# And my laptop as well, the reason is to generate a huge array and also import time


import numpy as np
import time
import random

# class MergeSort:
#
#       def mergeSort(self,array):
#             if len(array) <= 1:
#                   return array
#             mid = len(array)//2
#             left_side = array[:mid]
#             right_side = array[mid:]
#
#             left_sorted = self.mergeSort(left_side)
#             right_sorted = self.mergeSort(right_side)
#
#             return self.merge(left_sorted,right_sorted)
#
#       def merge(self,left,right):
#             a = b = 0
#             result = []
#             left_len = len(left)
#             right_len = len(right)
#             while a < left_len and b < right_len:
#                   if left[a] < right[b]:
#                         result.append(left[a])
#                         a+= 1
#                   else:
#                         result.append(right[b])
#                         b+= 1
#
#             while a < left_len:
#                   result.append(left[a])
#                   a+= 1
#             while b < right_len:
#                   result.append(right[b])
#                   b+= 1
#
#             return result
#
# if __name__ == "__main__":
#       a_huge_array = [random.randint(1,11111) for x in  range(2021)]
#       sol = MergeSort()
#       t1 = time.time()
#       res = sol.mergeSort(a_huge_array)
#       t2 = time.time()
#       print(f"The Sorted array is : {res} \nand the time it took was {t2-t1}")


# def binary_search(arr : list,target: int):
#     l = 0
#     r = len(arr) - 1
#
#     while l <= r:
#         mid = (l+r) // 2
#
#         if arr[mid] == target:
#             return  mid
#         if arr[mid] < target:
#             l =  mid+1
#         else:
#             r = mid - 1
#
#     return -1
#
#
# if __name__ == "__main__":
#     an_array = [8,7,6,5,4,3,2,1]
#     target_number = 6
#     sorted_array = an_array[::-1]
#     index_res = binary_search(sorted_array,target_number)
#     print(f'The sorted array will look like this - {sorted_array}')
#     print(f"The position of our target number will be: {index_res} ")

# Fibonacci Series with Tabulation

# def fibo(number:int) -> int:
#     if number <= 1:
#         return number
#     dp_array = [0] * (number+1)
#     dp_array[1] = 1
#     for i in range(2,number+1):
#         dp_array[i] = dp_array[i-1] + dp_array[i-2]
#
#     return dp_array[number-1]
#
# if __name__ == "__main__":
#     print(f"The {7}th item in the fibonacci series is {fibo(7)}")


# Now time to reduce the space complexity

# def fibo(num: int) -> int:
#     if num <= 1:
#         return num
#
#     prev, curr = 0, 1
#
#     for i in range(2, num + 1):
#         next_val = prev + curr
#         prev = curr
#         curr = next_val
#
#     return prev
#
#
# if __name__ == "__main__":
#     print(f"So our fibonacci number will be: {fibo(7)}")

# DP climbing stairs

# def climb_stairs(steps: int)-> int:
#     dp_st = [0] * (steps+1)
#     dp_st[1] = 1
#     dp_st[2] = 2
#     if 0 < steps <= 2:
#         return dp_st[steps]
#     else:
#         for i in range(3,steps+1):
#             dp_st[i] = dp_st[i-1] + dp_st[i-2]
#
#         return dp_st[steps]
#
# if __name__ == "__main__":
#     print(f'The number of ways we could climb the stairs are: {climb_stairs(4)}')

# Min Cost Climbing Stairs

# def stairs_cost(cost: list[int]) -> int:
#     l = len(cost)
#     dp_cost = [0] * (l+1)
#     dp_cost[0] = 0
#     dp_cost[1] = 0
#     for i in range(2, l+1):
#         dp_cost[i] = min(dp_cost[i-1] + cost[i-1], dp_cost[i-2] + cost[i-2])
#
#     return dp_cost[l]
#
# if __name__ == "__main__" :
#     print(stairs_cost([10, 15, 20]))
#     print(stairs_cost([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))

# Robbing Houses,

# def rob(houses:  list[int])-> int:
#     n = len(houses)
#     if n ==0:
#         return 0
#     if n == 1:
#         return houses[0]
#
#     dp = [0] * n
#
#     dp[0] = houses[0]
#     dp[1] = max(houses[0], houses[1])
#
#     for i in range(2,n):
#         dp[i] = max(dp[i-1], dp[i-2] + houses[i])
#
#     return dp[-1]
#
#
# if __name__ == "__main__":
#     print(rob([1, 2, 3, 4, 5]))

# Maximum Sub_Array

# def max_subarray(nums: list[int])-> list[int]:
#     n = len(nums)
#     dp_list = [0] * n
#     dp_list[0] = nums[0]
#     dp_sum = nums[0]
#     for i in range(1,n):
#         dp_list[i] = max(nums[i], dp_list[i-1] + nums[i])
#         dp_sum = max(dp_sum,dp_list[i])
#
#     return dp_sum
#
# if __name__ == "__main__":
#     numbers_random = [2,3,4,5,1,6,2,3,7,3,7,5,8,9,0,-5,-3,-5,-7,0]
#     print(max_subarray(numbers_random))


# dp = [[0]*4]*3
# print(dp)
#
# target = 'abc'
#
# for index_1, value_1 in enumerate('abdc'):
#     for index_2,value_2 in enumerate('dcba'):

def exist(board: list[list[str]], target):
    connecter = dict()
    l = len(board[0])
    for i in range(len(board)-1):



