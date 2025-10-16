# Depth First Search

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

def dfs_base(graph : dict, start):
    path = []
    traversed = set()
    order = deque()
    order.append(start)
    while order:
        node = order.pop()
        if node not in traversed:
            traversed.add(node)
            path.append(node)

            for neighbor in reversed(graph.get(node,[])):
                if neighbor not in traversed:
                    order.append(neighbor)

    return path

if __name__ == "__main__":
    print(dfs_base(graph,1))
