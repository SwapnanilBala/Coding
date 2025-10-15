# def swap(array,a,b):
#     array[a],array[b] = array[b],array[a]
#
# def partition(arr,low,high):
#     kingpin = arr[high]
#     i = low - 1
#     for j in range(low,high):
#         if arr[j] < kingpin:
#             i+= 1
#             swap(arr,i,j)
#
#     swap(arr,i+1,high)
#     return i+1
#
# def quicksort(array,low,high):
#     if low < high:
#         pi = partition(array,low,high)
#         quicksort(array,low,pi-1)
#         quicksort(array,pi+1,high)
#
#     return array
#
# if __name__ == "__main__":
#     an_array = [8,7,6,5,4,3,2,1]
#     print(quicksort(an_array,0,len(an_array)-1))
