import networkx as nx


#create graph
G=nx.Graph()
G.add_edges_from([(0,1),(0,2),(0,4),(0,3),(0,5),(1,7),(1,10),(1,11),(1,12),(2,4),(2,5),(2,3),(3,4),(5,8),(5,6),(6,8),(6,9),(6,7),(7,9),(7,10),(10,11),(10,12),(11,13),(12,13)])


clusteringCoefficientOfNode = []

def clustering_coefficient(G,n):
    # this will store the mapping of node/coefficient
    clusteringDict = {}
    for node in G:

        neighboursOfNode = []
        nodesWithMutualFriends = []

        # store all neighbors of the node in an array so we can compare
        for neighbour in G.neighbors(node):
            neighboursOfNode.append(neighbour)

        for neighbour in G.neighbors(node):
            for second_layer_neighbour in G.neighbors(neighbour):
                # compare if any second degree neighbour is also a first degree neighbour (this makes a triangle)
                # if so, append it to the mutual friends list
                if second_layer_neighbour in neighboursOfNode:
                    nodesWithMutualFriends.append(second_layer_neighbour)

        # filter duplicates from the mutual friend array
        nodesWithMutualFriends = list((nodesWithMutualFriends))

        # apply coefficient formula to calculate
        if len(nodesWithMutualFriends):
            clusteringCoefficientOfNode.append((float(len(list(nodesWithMutualFriends))))/((float(len(list(G.neighbors(node)))) * (float(len(list(G.neighbors(node)))) - 1))))
        clusteringDict[node] = clusteringCoefficientOfNode
    print(clusteringCoefficientOfNode[n])


clustering_coefficient(G,2)
