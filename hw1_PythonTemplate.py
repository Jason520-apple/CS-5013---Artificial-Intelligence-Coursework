# Josiah Abraham, Jason Vo
# CS 5013 - Fall 2026, HW1

# Problem: Implement the Breadth-First Search (BFS), Depth-First Search (DFS) 
# and Greedy Best-First Search (GBFS) algorithms on the graph from Figure 1 in hw1.pdf.


# Instructions:
# 1. Represent the graph from Figure 1 in any format (e.g. adjacency matrix, adjacency list).
# 2. Each function should take in the starting node as a string. Assume the search is being performed on
#    the graph from Figure 1.
#    It should return a list of all node labels (strings) that were expanded in the order they where expanded.
#    If there is a tie for which node is expanded next, expand the one that comes first in the alphabet.
# 3. You should only modify the graph representation and the function body below where indicated.
# 4. Do not modify the function signature or provided test cases. You may add helper functions. 
# 5. Upload the completed homework to Gradescope, it must be named 'hw1.py'.

# Examples:
#     The test cases below call each search function on node 'S' and node 'A'
# -----------------------------

# Step 1: store the graph of n nodes using an n-by-n adjacency matrix G, whose element
# G(i, j) stores the weight between node i and node j. 2 dimensional list

# based off figure 2, expanded to include i j k l m n p q
# -1 means that there is NOT a connection between the two nodes

from collections import deque
import heapq


adjacency_matrix = [
    #  A   B   C   D   E   F   G   H   I   J   K   L   M   N   P   Q   S
    [  0,  4, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # A
    [  4,  0,  2, -1, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # B
    [ -1,  2,  0, -1, -1, -1, -1,  4, -1, -1, -1, -1, -1, -1, -1, -1,  3], # C
    [ -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1,  8, -1, -1, -1, -1,  2], # D
    [  1, -1, -1, -1,  0,  3, -1, -1,  6, -1, -1, -1, -1, -1, -1, -1, -1], # E
    [ -1,  2, -1, -1,  3,  0, -1, -1, -1,  6,  4, -1, -1, -1, -1, -1, -1], # F
    [ -1, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1,  4,  4, -1, 10, -1], # G
    [ -1, -1,  4, -1, -1, -1, -1,  0, -1, -1,  3,  7, -1, -1, -1, -1, -1], # H
    [ -1, -1, -1, -1,  6, -1, -1, -1,  0,  1, -1, -1,  5, -1, -1, -1, -1], # I
    [ -1, -1, -1, -1, -1,  6, -1, -1,  1,  0,  3, -1, -1,  3, -1, -1, -1], # J
    [ -1, -1, -1, -1, -1,  4, -1,  3, -1,  3,  0,  9, -1, -1,  3, -1, -1], # K
    [ -1, -1, -1,  8, -1, -1, -1,  7, -1, -1,  9,  0, -1, -1, -1, 10, -1], # L
    [ -1, -1, -1, -1, -1, -1,  4, -1,  5, -1, -1, -1,  0, -1, -1, -1, -1], # M
    [ -1, -1, -1, -1, -1, -1,  4, -1, -1,  3, -1, -1, -1,  0,  2, -1, -1], # N
    [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, -1,  2,  0, -1, -1], # P
    [ -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, -1, 10, -1, -1, -1,  0, -1], # Q
    [ -1, -1,  3,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]  # S
]

# for accessing each row in adj matrix
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'S']

# for use with GBFS
h_values = {
    'A': 10, 'B': 9, 'C': 16, 'D': 21, 
    'E': 13, 'F': 9, 'G': 0, 'H': 12, 'I': 9, 
    'J': 5, 'K': 8, 'L': 18, 'M': 3, 'N': 4, 
    'P': 6, 'Q': 9, 'S': 17
}

# Breadth first search: FIFO queue
def BFS(start: str) -> list:
    # START: Your code here
    
    # need to output serarch sequence, sequence
    # of nodes being visited by your search algorithm
    
    # have two queues, visited and frontier
    # return visisted queue at the end
    # the BFS will start at arbitrary starting node but terminate at 'G'
    
    visited = [] # add the 
    frontierQueue = deque() #start with the given parameter
    frontierQueue.append(start)
    
    # idea: 'str' is our starting node, take its whole row, and look for entries that are != -1, > 0
    # then those nodes will be added to the frontier queue
    # base on alphabetical order, then 
    # whatever node is popped from frontier will add it's adjacent nodes to frontier, then move to visited
    
    while frontierQueue:
    
        # iterate through the row, if > 0, then add to frontier 
        current = frontierQueue.popleft()
        visited.append(current)
        
        # current is a string, get its equivalent index (ex: A -> 0, C -> 2)
        rowIndex = nodes.index(current)
        
        # will be incremented and used to track node index
        counter = 0
        
        # loop through the row in adjacency matrix for values > 0, if so get there string and add to frontierQueue
        for entry in adjacency_matrix[rowIndex]:
            
            currentNode = nodes[counter]
            
            if (entry > 0 and currentNode not in visited and currentNode not in frontierQueue):
                frontierQueue.append(currentNode)
                
                # terminate at G if it is found inside of row
                if currentNode == 'G':
                    visited.append(currentNode)
                    print(visited)

                    return visited
                
            counter += 1 #increment as we iterate to track for next node's index in the matrix
        
    print(visited)
    return visited

    # END: Your code here

# Depth first search: LIFO / Stack
def DFS(start: str) -> list:
    # START: Your code here
    return []
    # END: Your code here

# Greedy Best first search: priority queue
def GBFS(start: str) -> list:
    # START: Your code here
    
    visited = [] # add the 
    frontierQueue = [] #start with the given parameter
    heapq.heappush(frontierQueue, (0, start)) #starting node is A and = 0
    
    while frontierQueue:
        # serve smallest element in frontier priority queue
        current = heapq.heappop(frontierQueue)[1] # want to pop to get the string of the node, 2nd part of tuple (ex: 'A')

        visited.append(current)
        
        # current is a string, get its equivalent index (ex: A -> 0, C -> 2)
        rowIndex = nodes.index(current)
        
        # will be incremented and used to track node index
        counter = 0
        
        # loop through the row in adjacency matrix for values > 0, if so get there string and add to frontierQueue
        for entry in adjacency_matrix[rowIndex]:
            
            currentNode = nodes[counter] #will be used in calclating priorityValue
            # priorityValue is the current node's path + h(n)
            # ex: b's is 4 + 9, f is 2 + 9, we are going for lowest since greedy
        
            if (entry > 0 and currentNode not in visited and currentNode not in frontierQueue):
                
                priorityValue = entry + h_values[currentNode] # priorityvalue = path + h_values cost
                print(priorityValue)
                
                heapq.heappush(frontierQueue, (priorityValue, currentNode))
                
                # terminate at G if it is found inside of row
                if currentNode == 'G':
                    visited.append(currentNode)
                    print(visited)
                    return visited
                
            counter += 1 #increment as we iterate to track for next node's index in the matrix
        
    print(visited)
    return visited
    # the higher/max value from the current node's neighbor will be selected as next node
    
    # END: Your code here



# test cases - DO NOT MODIFY THESE
def run_tests():
    # Test case 1: BFS starting from node 'A'
    assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
    
    # Test case 2: BFS starting from node 'S'
    assert BFS('S') == ['S', 'C', 'D', 'B', 'H', 'L', 'A', 'F', 'K', 'Q', 'G'], "Test case 2 failed"

    # Test case 3: DFS starting from node 'A'
    assert DFS('A') == ['A', 'B', 'C', 'H', 'K', 'F', 'E', 'I', 'J', 'N', 'G'], "Test case 3 failed"
    
    # Test case 4: DFS starting from node 'S'
    assert DFS('S') == ['S', 'C', 'B', 'A', 'E', 'F', 'J', 'I', 'M', 'G'], "Test case 4 failed"

    # Test case 5: GBFS starting from node 'A'
    assert GBFS('A') == ['A', 'B', 'F', 'J', 'N', 'G'], "Test case 5 failed"
    
    # Test case 6: GBFS starting from node 'S'
    assert GBFS('S') == ['S', 'C', 'B', 'F', 'J', 'N', 'G'], "Test case 6 failed"

    
    
    print("All test cases passed!")

if __name__ == '__main__':
    run_tests()
