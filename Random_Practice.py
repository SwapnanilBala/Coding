import random
from functools import lru_cache
from collections import deque
from symtable import Class
from time import time

from Algo_Random_Prac import mergeSort
from Binary_Search import quick_sort


# First lets do Quick sort:

class Quick_Sort:

    def partition(self,array,low,high):
        kingpin = array[high]
        i = low - 1
        if len(array) <= 1:
            return array
        for j in range(low,high):
            if array[j] < array[high]:
                i+= 1
                self.swap(array,i,j)
        self.swap(array,i+1,high)

        return i+1



    def swap(self,array,a,b):
        array[a],array[b] = array[b], array[a]


    def quickSort(self,array,low,high):
        if low < high:
            pi = self.partition(array,low,high)

            self.quickSort(array,low,pi-1)
            self.quickSort(array,pi+1,high)


        return array

if __name__ == "__main__":
    an_array = [random.randint(1,33) for x in range(22)]
    Algo_1 = Quick_Sort()
    print(Algo_1.quickSort(an_array,0,len(an_array)-1))

# Second lets fo Merge Sort

class Merge_Sort:

    def merge(self,left,right):
        res = []
        a=b=0
        ls = len(left)
        rs = len(right)
        while a<ls and b<rs:
            if left[a] < right[b]:
                res.append(left[a])
                a+= 1
            else:
                res.append(right[b])
                b+= 1

        while a<ls:
            res.append(left[a])
            a+= 1
        while b<rs:
            res.append(right[b])
            b+= 1

        return res

    def mergeSort(self,array):
        if len(array) <= 1:
            return array

        mid = len(array)//2
        left_side = array[:mid]
        right_side = array[mid:]

        left_sorted = self.mergeSort(left_side)
        right_sorted = self.mergeSort(right_side)

        return self.merge(left_sorted,right_sorted)

if __name__ == "__main__":
    ran_array = [random.randint(2,45) for x in range(34)]
    Algo_2 = Merge_Sort()
    print(Algo_2.mergeSort(ran_array))


# Third lets do Bubble Sort

class Bubble_Sort:

    def bubbleSort(self,array):
        le = len(array)
        for i in range(le):
            for j in range(0,le-i-1):
                if i == j:
                    continue

                else:
                    if array[j] > array[j+1]:
                        array[j], array[j + 1] = array[j + 1], array[j]

        return array


if __name__ == "__main__":
    Algo_3 = Bubble_Sort()
    arr = [random.randint(5,100) for x in range(33)]
    print(Algo_3.bubbleSort(arr))

# Fourth lets do Depth First Search

class Depth_First_Search:

    def dfs(self,graph:dict,start):
        traversed = set()
        total_path = []
        curr_path = [start]
        while curr_path:
            present_node = curr_path.pop()
            traversed.add(present_node)
            total_path.append(present_node)
            for adjacent_nodes in reversed(graph.get(present_node,[])):
                if adjacent_nodes not in traversed:
                    traversed.add(adjacent_nodes)
                    curr_path.append(adjacent_nodes)

        return total_path

if __name__ == "__main__":
    Algo_4 = Depth_First_Search()
    graph_lit = {
        1: [2, 3],
        2: [6],
        3: [4, 5],
        4: [5],
        5: [8],
        6: [],
        7: [8],
        8: [7]
    }
    print(Algo_4.dfs(graph_lit,1))



# Fifth lets do Breadth First Search

class Breadth_First_Search:

    def bfs(self,graph,start):
        traversed = set()
        curr_path = deque([start])
        while curr_path:
            node = curr_path.popleft()
            if node not in traversed:
                traversed.add(node)
                print(node,end=" ")
                for adjacent_nodes in graph.get(node,[]):
                    curr_path.append(adjacent_nodes)


if __name__ == "__main__":
    Algo_5 = Breadth_First_Search()
    graph_sp = {
        0: [1, 2, 3],
        1: [5, 8],
        2: [4],
        3: [10],
        4: [3, 6],
        5: [2],
        6: [7],
        7: [9],
        8: [2],
        9: [10],
        10: []
    }
    print(Algo_5.bfs(graph_sp,0))


class Solution:
    def myPow(self,x:float,n:int)-> float:
        while n >=0:
            output = x**n
            return output

        y = (1/x)
        output = y**n
        return output

if __name__ == "__main__":
    sol = Solution()
    print(sol.myPow(2.100,-3))



