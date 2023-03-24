import networkx as nx

# create empty graph
G = nx.Graph()
# add nodes
G.add_nodes_from(range(1, 6))
# visualise

# add edges
# G.add_edges_from([(1, 2), (2, 5), (2, 4), (4, 3), (3, 5), (2, 3)])  # Question 1
# G.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5), (4, 5)])  # Question 2
# G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 5), (3, 4), (4, 5)])  # Question 3
# G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 5), (4, 5), (2, 4)])
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 5), (3, 4), (4, 5), (1, 4)])  # Question 3
# G.add_edges_from([(1, 2), (2, 3), (2, 5), (2, 5), (3, 4), (3, 5)])

# visualise
nx.draw(G, with_labels=True)

# nx.clustering(G, 4)
result = nx.eigenvector_centrality(G)
# nx.betweenness_centrality(G)
# print(nx.betweenness_centrality(G))
# divide result's key 3 by key 2
# print(result[1] / result[3])

result8 = nx.pagerank(G, alpha=1)
print(result8[2] / result8[4])
