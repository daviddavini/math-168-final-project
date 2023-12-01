import numpy as np
from traffic_animation import speed_matrix
from networkx_graph import networkx_graph
import networkx as nx

V = speed_matrix(228)
G = networkx_graph(228)

# def eigenvector_centralities(N):
#     assert N in [228, 1026]
#     G = networkx_graph(N, weighted=False)
#     return nx.eigenvector_centrality(G)


average_speeds = V.mean(axis=0)
# get numpy array of degrees for each node
degrees = np.array([G.degree(n) for n in G.nodes()])

# find correlation between the two
correlation = np.corrcoef(average_speeds, degrees)[0,1]
print("Correlation between average speed and degree:", correlation)

from scipy.stats import linregress
from scipy.stats.stats import pearsonr
res = linregress(average_speeds, degrees)
print(res)
res = pearsonr(average_speeds, degrees)
print(res)


# plot the two
import matplotlib.pyplot as plt
plt.scatter(average_speeds, degrees)
plt.xlabel("Average speed")
plt.ylabel("Degree")
plt.savefig("speed_degree_correlation.png", dpi=300)

# compute eigenvector centrality
G = networkx_graph(228)
# plt.clf()
# nx.draw(G, node_color='red', edge_color='blue', node_size=20)
# plt.show()
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')
for edge in G.edges():
  print(edge)
# get numpy array of eigenvector centralities for each node
eigenvector_centrality = np.array([eigenvector_centrality[n] for n in G.nodes()])

# find correlation between the two
correlation = np.corrcoef(average_speeds, eigenvector_centrality)[0,1]
print("Correlation between average speed and eigenvector centrality:", correlation)

# plot the two
plt.clf()
plt.scatter(average_speeds, eigenvector_centrality)
plt.xlabel("Average speed")
plt.ylabel("Eigenvector centrality")
plt.savefig("speed_eigenvector_centrality_correlation.png", dpi=300)

# compute katz centrality
katz_centrality = nx.katz_centrality_numpy(G)
# get numpy array of katz centralities for each node
katz_centrality = np.array([katz_centrality[n] for n in G.nodes()])

# compute degree centrality
degree_centrality = nx.degree_centrality(G)
# get numpy array of degree centralities for each node
degree_centrality = np.array([degree_centrality[n] for n in G.nodes()])

# map the centrality using traffic_animation.py
from traffic_animation import setup_map_plot, info
def map_values(c, label, title, filename):
    setup_map_plot()
    # plot all of the stations
    plt.scatter(info['Longitude'], info['Latitude'], s=50, c=c, edgecolor='black', vmin=0, vmax=c.max())

    cbar = plt.colorbar(fraction=0.025)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label(label, rotation=270)
    plt.title(title)

    plt.savefig(filename, dpi=300)
    plt.clf()

map_values(eigenvector_centrality, "Eigenvector centrality", "Eigenvector centrality of all stations", "map_eigenvector_centrality.png")
map_values(katz_centrality, "Katz centrality", "Katz centrality of all stations", "map_katz_centrality.png")
map_values(degree_centrality, "Degree centrality", "Degree centrality of all stations", "map_degree_centrality.png")