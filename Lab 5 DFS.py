# Using a Python dictionary to act as an adjacency list
graph = {
  'A': ['B', 'C', "D"],
    'B': ['E', "F"],
    'C': ['G', "H"],
    'D': ["I"],
    'E': [],
    "F": [],
    "G": [],
    "H": [],
    "I": []
}

visited = set() # Set to keep track of visited nodes of graph.

def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

# Driver Code
print("Following is the Depth-First Search")
dfs(visited, graph, 'A')