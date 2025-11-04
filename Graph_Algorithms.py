# Depth First Search
from email.policy import default

from numpy.f2py.crackfortran import traverse

graph = {
    1: [2, 3],
    2: [6],
    3: [4, 5],
    4: [5],
    5: [8],
    6: [],
    7: [8],
    8: [7]  # (example directed/undirected depends on edges you add)
}

from collections import deque

# def dfs_base(graph : dict, start):
#     path = []
#     traversed = set()
#     order = deque()
#     order.append(start)
#     while order:
#         node = order.pop()
#         if node not in traversed:
#             traversed.add(node)
#             path.append(node)
#
#             for neighbor in reversed(graph.get(node,[])):
#                 if neighbor not in traversed:
#                     order.append(neighbor)
#
#     return path
#
# if __name__ == "__main__":
#     print(dfs_base(graph,1))

# def dfs(graph: dict, start: int)-> list:
#     res = []
#     traversed = set()
#     curr_path = [start]
#     while curr_path:
#         adjacent = curr_path.pop()
#         if adjacent not in traversed:
#             traversed.add(adjacent)
#             res.append(adjacent)
#             for adjacent_nodes in reversed(graph.get(adjacent, [])):
#                 if adjacent_nodes not in traversed:
#                     curr_path.append(adjacent_nodes)
#
#     return res
#
# if __name__ == "__main__":
#     graph = {
#         1: [2, 3],
#         2: [6],
#         3: [4, 5],
#         4: [5],
#         5: [8],
#         6: [],
#         7: [8],
#         8: [7]  # (example directed/undirected depends on edges you add)
#     }
#     print(dfs(graph, 1))
#
#
# print(list(reversed(graph.get(2))))
#
#
#
#
# # DFS
#
# def depth_first_search(graph:dict, start: str) -> list[str]:
#     paths = []
#     traversed = set()
#     curr_path = [start]
#     while curr_path:
#         step = curr_path.pop()
#         if step not in traversed:
#             traversed.add(step)
#             paths.append(step)
#             for adjacent_nodes in reversed(graph.get(step,[])):
#                 if adjacent_nodes not in traversed:
#                     curr_path.append(adjacent_nodes)
#
#     return paths
#
# if __name__ == "__main__":
#     graph_str = {
#         'A': ['B', 'C', 'D'],
#         'B': ['E', 'F'],
#         'C': ['G', 'H'],
#         'D': ['I'],
#         'E': ['J', 'K'],
#         'F': ['L'],
#         'G': ['M', 'N'],
#         'H': ['O'],
#         'I': ['P', 'Q'],
#         'J': ['R'],
#         'K': ['S', 'T'],
#         'L': ['U'],
#         'M': ['V', 'W'],
#         'N': ['X'],
#         'O': ['Y', 'Z'],
#         'P': ['AA'],
#         'Q': ['AB', 'AC'],
#         'R': ['AD'],
#         'S': [],
#         'T': ['AE', 'AF'],
#         'U': [],
#         'V': ['AG'],
#         'W': ['AH', 'AI'],
#         'X': [],
#         'Y': ['AJ'],
#         'Z': [],
#         'AA': [],
#         'AB': ['AK'],
#         'AC': ['AL', 'AM'],
#         'AD': [],
#         'AE': [],
#         'AF': ['AN'],
#         'AG': [],
#         'AH': ['AO'],
#         'AI': [],
#         'AJ': [],
#         'AK': ['AP', 'AQ'],
#         'AL': [],
#         'AM': ['AR'],
#         'AN': ['AS'],
#         'AO': [],
#         'AP': [],
#         'AQ': ['AR'],
#         'AR': ['AS'],
#         'AS': []
#     }
#     print(depth_first_search(graph_str, 'A'))
#
#
#
#
# # 3rd Try:
#
# def Depth_first_Search(graph:dict, start:int):
#     whole_path = []
#     been_there = set()
#     present_path = [start]
#     while present_path:
#         step = present_path.pop()
#         if step not in been_there:
#             been_there.add(step)
#             whole_path.append(step)
#             for neighbors in reversed(graph.get(step, [])):
#                 if neighbors not in been_there:
#                     present_path.append(neighbors)
#
#     return whole_path
#
#
# if __name__ == "__main__":
#     print(Depth_first_Search(graph,1))


# import numpy as np
#
#
# array = np.random.normal(loc= 10000, scale= 1, size= 1000000)
#
# ans = len(array)/len(np.unique(array))
# print(ans)
#
#
#
#
# pop = np.arange(100000)  # finite population
# sample = np.random.choice(pop, size=100000, replace=True)
# ratio = len(sample) / len(np.unique(sample))
# print(ratio)


# class DFS:
#     def depth_first_search(self,graph:dict,start: int):
#         path = []
#         traversed = set()
#         curr_path = [start]
#         while curr_path:
#             node = curr_path.pop()
#             if node not in traversed:
#                 path.append(node)
#                 traversed.add(node)
#                 for adjacent_nodes in reversed(graph.get(node, [])):
#                     if adjacent_nodes not in traversed:
#                         curr_path.append(adjacent_nodes)
#
#         return path
#
# if __name__ == "__main__":
#     sol = DFS()
#     graph_0 = {
#         1: [2, 3],
#         2: [4, 5, 6],
#         3: [7],
#         4: [],
#         5: [8],
#         6: [9, 10],
#         7: [],
#         8: [],
#         9: [],
#         10: []
#     }
#     print(sol.depth_first_search(graph_0,1))

from collections import deque

class Breadth_First_Search:

    def bfs(self, graph: dict, start: int, end: int) -> list:
        visited = set()
        queue = deque([[start]])  # store paths as lists

        while queue:
            path = queue.popleft()
            node = path[-1]

            if node == end:
                return path

            if node not in visited:
                visited.add(node)

                for neighbor in graph.get(node, []):
                    new_path = path + [neighbor]
                    queue.append(new_path)

        return [None]


if __name__ == "__main__":
    sol = Breadth_First_Search()
    graph_1 = {
        1: [2, 3, 4],
        2: [5, 6],
        3: [7, 8, 9],
        4: [10],
        5: [11, 12],
        6: [13],
        7: [14, 15],
        8: [16],
        9: [17, 18],
        10: [19],
        11: [],
        12: [20, 21],
        13: [22],
        14: [23],
        15: [24, 25],
        16: [],
        17: [26],
        18: [27, 28],
        19: [],
        20: [],
        21: [],
        22: [29],
        23: [],
        24: [30],
        25: [],
        26: [],
        27: [],
        28: [],
        29: [],
        30: []
    }
    print(sol.bfs(graph_1, 1, 30))


