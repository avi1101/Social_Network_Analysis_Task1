__author__ = "Avi Elly 308000017, Elad Cohen- 308468784"

import pandas as pd
import heapq
from operator import itemgetter


adjacency_list, page_rank, reverse_adjacency_list, last_page_rank = {}, {}, {}, {}
nodes = []


def load_graph(path):
    """
    This function loads a graph given the csv of said graph
    :param path: URL path for the csv folder
    :return: None
    """
    global adjacency_list  # source -> list of destinations
    global reverse_adjacency_list  # destination -> list of sources
    global nodes  # list of all nodes
    df = pd.read_csv(header=None, index_col=None, filepath_or_buffer=path, names=['Source', 'Dest'])
    nodes = pd.unique(df[['Source', 'Dest']].values.ravel())
    reverse_adjacency_list = dict.fromkeys(nodes)
    adjacency_list = dict.fromkeys(nodes)
    for index, row in df.iterrows():    # fill adjacency_list
        if adjacency_list[row[0]] is not None:
            adjacency_list[row[0]].append(row[1])
        else:
            adjacency_list[row[0]] = [row[1]]
        if reverse_adjacency_list[row[1]] is not None:  #fill reverse_adjacency_list
            reverse_adjacency_list[row[1]].append(row[0])
        else:
            reverse_adjacency_list[row[1]] = [row[0]]


def calc_pagerank_iteration(beta):
    """
    Does one iteration of PageRank value update for all the nodes
    :param beta: the beta variable
    :return: None
    """
    global adjacency_list
    global reverse_adjacency_list
    global page_rank
    global last_page_rank
    global nodes
    for node in adjacency_list.keys():
        if node not in reverse_adjacency_list or reverse_adjacency_list[node] is None:
            page_rank[node] = 0.0
        else:
            current_node_value = 0.0
            for in_node in reverse_adjacency_list[node]:
                current_node_value += float(last_page_rank[in_node]) / float(len(adjacency_list[in_node])) * beta
                page_rank[node] = current_node_value
    delta = sum({key: abs(page_rank[key] - last_page_rank[key]) for key in page_rank if key in last_page_rank}.values())
    leak_value = float(1-sum(page_rank.values())) / float(len(nodes))
    for node, current_node_value in page_rank.items():
        current_node_value += leak_value
        page_rank[node] = current_node_value
    return delta


def calculate_page_rank(beta=0.85, epsilon=0.001, maxIterations=20):
    """
    calculates the PageRank of the graph, given the beta , epsilon and max number of iteration
    :param beta: parameter of the change for the PageRank to work according to route, non “teleportation” probability
    :param epsilon: parameter for the min difference between an 2 consecutive iterations PageRank values
    :param maxIterations: parameter for max number of iteration
    :return: None
    """
    global adjacency_list
    global reverse_adjacency_list
    global page_rank
    global last_page_rank
    global nodes
    last_page_rank = dict.fromkeys(adjacency_list.keys(),
                                   1.0 / float(len(nodes)))  # set initial values of all PageRank to 1/N
    page_rank = dict.fromkeys(adjacency_list.keys(), 0.0)  # values of PageRank will be stored here
    iteration = 1
    delta_sum = calc_pagerank_iteration(beta)
    last_page_rank = page_rank.copy()
    iteration += 1
    while iteration < maxIterations and delta_sum > epsilon:  # loop to update PageRank
        last_page_rank = page_rank.copy()
        delta_sum = calc_pagerank_iteration(beta)
        iteration += 1


def get_PageRank(node_name):
    """
    get the PageRank of a certain node given its name
    :param node_name: name of the node
    :return: integer value of its rank, -1 if it does not exist
    """
    global page_rank
    try:
        return page_rank[node_name]
    except KeyError:
        return -1


def get_top_nodes(n):
    """
    return the n nodes with the top PageRank value
    :param n: number of nodes to return
    :return: list of tuples : (node, PageRank value)
    """
    global page_rank
    top_items = heapq.nlargest(n, page_rank.items(), key=itemgetter(1))
    top_items = dict(top_items)
    print(top_items)
    return top_items


def get_all_PageRank():
    """
    returns the PageRank of all nodes in the graph
    :return: list of tuples : (node, PageRank value)
    """
    global page_rank
    return sorted(page_rank.items(), key=itemgetter(1), reverse=True)


if __name__ == '__main__':
    print("***"*25+"Wikipedia_votes"+"***"*25)
    load_graph('Wikipedia_votes.csv')
    calculate_page_rank()
    get_top_nodes(10)
    print("***" * 25 + "soc-Epinions1" + "***" * 25)
    load_graph('soc-Epinions1.csv')
    calculate_page_rank()
    get_top_nodes(10)
