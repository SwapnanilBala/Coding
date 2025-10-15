# Will do binary search with quicksort, sp basically our function will first take in an unsorted_array,
# then it will sort it, and finally it will search for the index if an element's index

def partition(arr : list,low,high) -> int:
    kingpin = arr[high]
    i = low - 1
    for j in range(low,high):
        if arr[j] < kingpin:
            i+=1
            swap(arr,i,j)
    swap(arr,i+1,high)
    return i+1



def swap(arr,a,b):
    arr[a],arr[b] = arr[b],arr[a]


def quick_sort(arr:list,low,high) -> list:
    if low < high:

        pi = partition(arr,low,high)

        quick_sort(arr,low,pi-1)
        quick_sort(arr,pi+1,high)


    return arr

def binary_search(arr : list,target: int):
    l = 0
    r = len(arr) - 1

    while l <= r:
        mid = (l+r) // 2

        if arr[mid] == target:
            return  mid
        if arr[mid] < target:
            l =  mid+1
        else:
            r = mid - 1

    return -1


if __name__ == "__main__":
    an_array = [8,7,6,5,4,3,2,1]
    target_number = 6
    sorted_array = quick_sort(an_array,0,len(an_array)-1)
    index_res = binary_search(sorted_array,target_number)
    print(f'The sorted array will look like this - {sorted_array}')
    print(f"The position of our target number will be: {index_res} ")