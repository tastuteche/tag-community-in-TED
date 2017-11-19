import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
import community
from tabulate import tabulate
from itertools import combinations
import math
import numpy as np
b_dir = './ted-talks/'

df = pd.read_csv(b_dir + 'ted_main.csv')
import datetime
df['film_date'] = df['film_date'].apply(
    lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%d-%m-%Y'))
df['published_date'] = df['published_date'].apply(
    lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%d-%m-%Y'))

df['tags_eval'] = df['tags'].apply(lambda x: ast.literal_eval(x))


def get_dic_node_edge(df, tag_list_col):
    dic_edge = {}
    dic_node = {}
    for tag_list in df[tag_list_col]:
        tag_list = sorted(tag_list)
        for source, target in combinations(tag_list, 2):
            dic_edge.setdefault((source, target), 0)
            dic_edge[(source, target)] += 1
        for node in tag_list:
            dic_node.setdefault(node, 0)
            dic_node[node] += 1
    return (dic_node, dic_edge)


dic_node, dic_edge = get_dic_node_edge(df, 'tags_eval')


def get_G(dic_node, dic_edge):
    G = nx.Graph()
    G.add_weighted_edges_from([(source, target, weight)
                               for (source, target), weight in dic_edge.items()])
    return G


G = get_G(dic_node, dic_edge)


def plot_G(G, dic_node, dic_edge):
    plt.figure(figsize=(25, 25))
    list_node, list_node_size = [], []
    for k, v in dic_node.items():
        list_node.append(k)
        list_node_size.append(math.sqrt(v))
    list_edge, list_edge_width = [], []
    for k, v in dic_edge.items():
        list_edge.append(k)
        list_edge_width.append(math.sqrt(v))
    pos = nx.spring_layout(G, k=0.15)
    nx.draw_networkx_nodes(G, pos, list_node, node_size=list_node_size,
                           node_color='blue')
    nx.draw_networkx_edges(G, pos, edgelist=list_edge,
                           width=list_edge_width, alpha=0.5)


def get_edge_width(edge):
    if edge in dic_edge:
        return dic_edge[edge]
    else:
        source, target = edge
        return dic_edge[(target, source)]


from networkx.drawing.nx_agraph import graphviz_layout


def draw_partition_N(category, G, partition, n):
    plt.figure(figsize=(15, 15))
    list_node = [nodes for nodes in partition.keys()
                 if partition[nodes] == n]
    dic_node_weight = {node: dic_node[node] for node in list_node}
    sorted_node_list, sorted_node_weight = zip(*sorted(dic_node_weight.items(),
                                                       key=lambda item: item[1], reverse=True))
    # print(sorted_node_weight)
    list_node = sorted_node_list[:100]
    list_node_size = [math.sqrt(dic_node[node]) * 2 for node in list_node]
    data3d = np.random.uniform(low=0.0, high=1.0, size=(len(list_node), 3))
    H = G.subgraph(list_node)
    #pos = nx.spring_layout(H, k=0.45, weight='weight')  #
    # pos = nx.circular_layout(H)
    pos = graphviz_layout(H, prog='neato')
    list_edge = list(H.edges())
    list_edge_width = [math.sqrt(get_edge_width(edge))
                       * 2 for edge in list_edge]
    # nx.draw_networkx(H, pos, nodelist=list_node, node_size=list_node_size,
    #                  edgelist=list_edge, width=1.0, node_color='blue')
    nx.draw_networkx_nodes(H, pos, nodelist=list_node, node_size=list_node_size,
                           node_color=data3d, alpha=1)
    nx.draw_networkx_edges(H, pos, edgelist=list_edge,
                           width=list_edge_width, alpha=0.1)
    nx.draw_networkx_labels(H, pos, font_size=11, font_color='r')
    # plt.show()
    plt.savefig('%s_partition_%s.png' % (category, str(n)), dpi=200)
    plt.clf()
    plt.cla()
    plt.close()


# first compute the best partition
partition_tag = community.best_partition(G)
# drawing


par_tag_pd = pd.DataFrame.from_dict(partition_tag, orient='index')
par_tag_pd.columns = ['partition']
par_tag_pd.groupby('partition').size().sort_values(ascending=False)
sorted_par_tag = par_tag_pd.groupby(
    'partition').size().sort_values(ascending=False)
print(tabulate(sorted_par_tag.reset_index(), tablefmt="pipe", headers="keys"))
draw_partition_N('tag', G, partition_tag, 4)
