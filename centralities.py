import os
from matplotlib import pyplot as plt
import numpy as np
from traffic_animation import avg_day_speed_at_time, speed_matrix, map_values
from networkx_graph import networkx_graph
import networkx as nx
from scipy.stats.stats import pearsonr

data_dir = "images"

V = speed_matrix(228)
G = networkx_graph(228)

average_speeds = V.mean(axis=0)

G = networkx_graph(228)

def plot_correlation(x, y, xlabel, ylabel, filename):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(data_dir, filename), dpi=300)
    plt.clf()

def correlation(x, y, xname, yname, filename):
    res = pearsonr(x, y)
    print("Correlation between {} and {}:\n".format(xname, yname), "{: .4f}".format(res[0]), "p-value:", "{: .4f}".format(res[1]))
    plot_correlation(x, y, xname, yname, filename)

degree_centrality = nx.degree_centrality(G)
degree_centrality = np.array([degree_centrality[n] for n in G.nodes()])

eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')
eigenvector_centrality = np.array([eigenvector_centrality[n] for n in G.nodes()])

katz_centrality = nx.katz_centrality_numpy(G)
katz_centrality = np.array([katz_centrality[n] for n in G.nodes()])

page_rank_centrality = nx.pagerank(G)
page_rank_centrality = np.array([page_rank_centrality[n] for n in G.nodes()])

correlation(average_speeds, degree_centrality, "Average speed", "Degree", "correlation_speed_degree.png")
correlation(average_speeds, eigenvector_centrality, "Average speed", "Eigenvector centrality", "correlation_speed_eigenvector_centrality.png")
correlation(average_speeds, katz_centrality, "Average speed", "Katz centrality", "correlation_speed_katz_centrality.png")
correlation(average_speeds, page_rank_centrality, "Average speed", "Page rank centrality", "correlation_speed_page_rank_centrality.png")


speeds_at_7_50 = avg_day_speed_at_time(7, 50)
correlation(speeds_at_7_50, degree_centrality, "Speed at 7:50", "Degree", "correlation_speed_7_50_degree.png")
correlation(speeds_at_7_50, page_rank_centrality, "Speed at 7:50", "Page rank centrality", "correlation_speed_7_50_page_rank_centrality.png")
speed_at_17_30 = avg_day_speed_at_time(17, 30)
correlation(speed_at_17_30, degree_centrality, "Speed at 17:30", "Degree", "correlation_speed_17_30_degree.png")
correlation(speed_at_17_30, page_rank_centrality, "Speed at 17:30", "Page rank centrality", "correlation_speed_17_30_page_rank_centrality.png")

map_values(eigenvector_centrality, "Eigenvector centrality", "Eigenvector centrality of all stations", "map_eigenvector_centrality.png")
map_values(katz_centrality, "Katz centrality", "Katz centrality of all stations", "map_katz_centrality.png")
map_values(degree_centrality, "Degree centrality", "Degree centrality of all stations", "map_degree_centrality.png")
map_values(page_rank_centrality, "Page rank centrality", "Page rank centrality of all stations", "map_page_rank_centrality.png")