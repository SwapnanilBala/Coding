# def mergesort(arr):
#     if len(arr) <= 1:
#         return arr
#
#     mid = len(arr)//2
#     left = arr[:mid]
#     right = arr[mid:]
#
#     left = mergesort(left)
#     right = mergesort(right)
#
#     return merge(left,right)
#
#
# def merge(left,right):
#     x = y = 0
#     res = []
#     le = len(left)
#     ri = len(right)
#     while x < le and y < ri:
#         if left[x] <= right[y]:
#             res.append(left[x])
#             x += 1
#         else:
#             res.append(right[y])
#             y += 1
#
#     while x < le:
#         res.append(left[x])
#         x+= 1
#
#     while y < ri:
#         res.append(right[y])
#         y+= 1
#
#     return res
#
# if __name__ == "__main__":
#     an_array = [6,5,4,3,2,1,0]
#     print(mergesort(an_array))

# word = "Gogol"
# print(word[::-1])