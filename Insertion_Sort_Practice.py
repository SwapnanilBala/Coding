# def insertion_sort(sorted_array:list, number: int) -> list :
#     for i in range(0,len(sorted_array)):
#         j = i - 1
#         if number > sorted_array[i] and i < len(sorted_array) - 1:
#             continue
#         elif number == sorted_array[i]:
#             sorted_array.insert(j,number)
#             break
#         elif number > max(sorted_array):
#             sorted_array.append(number)
#             break
#         else:
#             sorted_array.insert(0,number)
#
#
#     return sorted_array
#
# if __name__ == "__main__":
#     sorted_array = [1,2,3,4,5,6,7]
#     print(insertion_sort(sorted_array,8))
from importlib.metadata import requires
# Some Stupid Ass easy LeetCode Problems

from time import time


def target_index_finder(arr:list, target: int):
    seen = set()
    i = 0
    for j in range(i+1,len(arr)):
        required = target - arr[i]
        if required in arr[j:]:
            seen.add(tuple(sorted([i,j])))
            i += 1
        else:
            i += 1

    res = list(seen)

    if len(seen) < 1:
        return None
    else:
        return res


if __name__ == "__main__":
    t1 = time()
    nums = [
4, 6, 13, 24, 8, 28, 35, 35, 42, 46, 39, 47, 21, 6, 45, 9, 3, 3, 28, 50,
1, 33, 2, 42, 48, 36, 5, 43, 11, 29, 41, 37, 40, 3, 36, 44, 8, 12, 14, 28,
50, 4, 27, 34, 16, 42, 19, 3, 50, 22, 1, 10, 31, 41, 37, 20, 47, 46, 23, 35,
33, 25, 5, 9, 21, 48, 28, 13, 12, 20, 27, 19, 15, 38, 2, 36, 43, 49, 22, 44,
18, 23, 37, 14, 11, 15, 26, 7, 47, 39, 1, 5, 48, 32, 4, 30, 25, 21, 46, 49,
12, 13, 31, 9, 18, 16, 24, 41, 10, 19, 44, 28, 26, 27, 40, 43, 33, 6, 15, 32,
20, 23, 35, 42, 14, 45, 2, 38, 29, 39, 7, 17, 22, 30, 8, 16, 34, 50, 18, 7
]
    target = 18
    print(target_index_finder(nums,target))
    t2 = time()
    print(f"Time Taken: {t2 - t1}")


# Alternate Approach, using Hashmaps

def tu_sum(arr: list, target: int) ->  list:
    seen = {}
    try:
        for index, value in enumerate(arr):
            required = target - value
            if required not in seen:
                seen[value] = index
            else:
                continue
    finally:
        if len([seen]) < 1:
            return ["N/A"]
        else:
            return [seen]

if __name__ == "__main__":
    t3 = time()
    nums  = [
4, 6, 13, 24, 8, 28, 35, 35, 42, 46, 39, 47, 21, 6, 45, 9, 3, 3, 28, 50,
1, 33, 2, 42, 48, 36, 5, 43, 11, 29, 41, 37, 40, 3, 36, 44, 8, 12, 14, 28,
50, 4, 27, 34, 16, 42, 19, 3, 50, 22, 1, 10, 31, 41, 37, 20, 47, 46, 23, 35,
33, 25, 5, 9, 21, 48, 28, 13, 12, 20, 27, 19, 15, 38, 2, 36, 43, 49, 22, 44,
18, 23, 37, 14, 11, 15, 26, 7, 47, 39, 1, 5, 48, 32, 4, 30, 25, 21, 46, 49,
12, 13, 31, 9, 18, 16, 24, 41, 10, 19, 44, 28, 26, 27, 40, 43, 33, 6, 15, 32,
20, 23, 35, 42, 14, 45, 2, 38, 29, 39, 7, 17, 22, 30, 8, 16, 34, 50, 18, 7
]
    target = 18
    print(tu_sum(nums,target))
    t4 = time()
    print(f"Estimate Time taken: {t4 - t3}")


# The classic Leet-code approach

def all_two_sums(arr:list, target:int) -> list:
    seen = set()
    pairs = []
    for i,v in enumerate(arr):
        needed = target - v
        if needed not in seen and needed in arr[i+1:]:
            seen.add(tuple(sorted([arr.index(needed),i])))
            pairs.append(tuple(sorted([arr.index(needed),i])))
        else:
            continue

    return pairs

if __name__ == "__main__":
    t5 = time()
    nums = [
        4, 6, 13, 24, 8, 28, 35, 35, 42, 46, 39, 47, 21, 6, 45, 9, 3, 3, 28, 50,
        1, 33, 2, 42, 48, 36, 5, 43, 11, 29, 41, 37, 40, 3, 36, 44, 8, 12, 14, 28,
        50, 4, 27, 34, 16, 42, 19, 3, 50, 22, 1, 10, 31, 41, 37, 20, 47, 46, 23, 35,
        33, 25, 5, 9, 21, 48, 28, 13, 12, 20, 27, 19, 15, 38, 2, 36, 43, 49, 22, 44,
        18, 23, 37, 14, 11, 15, 26, 7, 47, 39, 1, 5, 48, 32, 4, 30, 25, 21, 46, 49,
        12, 13, 31, 9, 18, 16, 24, 41, 10, 19, 44, 28, 26, 27, 40, 43, 33, 6, 15, 32,
        20, 23, 35, 42, 14, 45, 2, 38, 29, 39, 7, 17, 22, 30, 8, 16, 34, 50, 18, 7
    ]
    target = 13
    print(all_two_sums(nums,target))
    t6 = time()
    print(f"Time taken: {t6-t5}")

# Bad Time Complexity


## Perfect Time Complexity


