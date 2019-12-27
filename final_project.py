# %%
"""
normal version:

initial a set of subgraph, every single node is a subgraph
while number of subgraph > 1
    for subgraph in subgraph set
        find every subgraph's shortes neighbour node and   if the node not in subgraph add the node and the edge in the graph
    merge the subgraph if they have the same node and renew the set of subgraph

parallel version:

initial a set of subgraph, every single node is a subgraph
while number of subgraph > 1
    paraller find every subgraph's shortes neighbour node and   if the node not in subgraph add the node and the edge in the graph
    merge the subgraph if they have the same node and renew the set of subgraph
"""
import networkx as nx
import time
from random import randint, sample, seed
import multiprocessing as mp


class Graph:
    """
    generate graph class
    """

    def __init__(self):
        self.graph = {'nodes': [], 'edges': []}

    def add_edge(self, X, Y):
        self.graph['edges'].append((X, Y))
        self.graph['edges'].append(((Y, X)))
        self.graph['edges'] = list(set(self.graph['edges']))

    def add_node(self, node):
        self.graph['nodes'].append(node)
        self.graph['nodes'] = list(set(self.graph['nodes']))

    def add(self, X, Y):
        self.add_edge(X, Y)
        self.add_node(X)
        self.add_node(Y)

    def node(self):
        return self.graph['nodes']

    def edge(self):
        return self.graph['edges']

    def union(self, G):
        self.graph['nodes'] = list(set.union(set(G.node()), set(self.graph['nodes'])))
        self.graph['edges'] = list(set.union(set(G.edge()), set(self.graph['edges'])))

    def join_exist(self, G):
        """
        is there any same nodes in two subgraph?
        :param G:
        :return:
        """
        if len(set(G.node()) & set(self.graph['nodes'])) != 0:
            return True
        else:
            return False


class BA:
    """
    BA 算法
    """

    def __init__(self):
        self.G = nx.Graph()
        self.graph_dict = nx.to_dict_of_dicts(self.G)
        self.graph_dict_key_list = list(self.graph_dict.keys())

    def add_edges(self, edge_list):
        self.G.add_weighted_edges_from(edge_list)
        self.graph_dict = nx.to_dict_of_dicts(self.G)

        for node, adj in self.graph_dict.items():
            for sub_node, value in self.graph_dict[node].items():
                self.graph_dict[node][sub_node] = self.graph_dict[node][sub_node]['weight']
        self.graph_dict_key_list = list(self.graph_dict.keys())

    def union_subgraph(self, subgraph_set):
        """
        union subgraph
        :param subgraph_set:
        :return:
        """
        new_subgrsubgraph_set = {}  # 新的输出集合

        subgraph_key_list = list(subgraph_set.keys())
        # print('subgraph_key_list', subgraph_key_list)
        absorb_key = []
        index = 0
        for first_key in subgraph_key_list:
            index += 1
            for seconde_key in subgraph_key_list[index:len(subgraph_key_list) + 1]:
                if seconde_key in absorb_key:
                    pass
                else:
                    if subgraph_set[first_key].join_exist(subgraph_set[seconde_key]):
                        subgraph_set[first_key].union(subgraph_set[seconde_key])
                        absorb_key.append(seconde_key)
                    else:
                        pass

        for key, graph in subgraph_set.items():
            if key not in absorb_key:
                new_subgrsubgraph_set[key] = graph

        return new_subgrsubgraph_set

    def init_subgraph_sets(self):
        """
        initial subgraph sets
        :return:
        """
        subgraph_set = {}
        for node in list(self.G.node()):
            graph = Graph()
            graph.add_node(node)
            subgraph_set[node] = graph
        return subgraph_set

    def graph_grow(self, graph):
        """
        graph growing
        :param edge_dict:
        :return:
        """
        mini_value = None
        mini_node = None
        start_node = None
        for node in list(set(graph.node()) & set(self.graph_dict_key_list)):

            for adj_node in list(set(self.graph_dict[node]) - set(graph.node())):

                if mini_value is None or mini_node is None or start_node is None:
                    mini_value = self.graph_dict[node][adj_node]
                    mini_node = adj_node
                    start_node = node
                else:
                    if self.graph_dict[node][adj_node] < mini_value:
                        mini_value = self.graph_dict[node][adj_node]
                        mini_node = adj_node
                        start_node = node
        graph.add_node(mini_node)
        graph.add_edge(start_node, mini_node)
        return graph

    def boruvkaMST(self):
        """
        run algorithm
        :return:
        """
        subgraph_set = self.init_subgraph_sets()
        while len(subgraph_set) > 1:
            for key, graph in subgraph_set.items():
                new_graph = self.graph_grow(graph)
                subgraph_set[key] = new_graph

            subgraph_set = self.union_subgraph(subgraph_set=subgraph_set)

        self.mini_tree = list(subgraph_set.values())[0]

        return subgraph_set

    def calculate_min_tree_lenght(self):
        """
        calculate the lenght of mini tree
        :return:
        """
        sum_path = 0
        for edge in self.mini_tree.edge():
            sum_path += self.graph_dict[edge[0]][edge[1]]
        sum_path = sum_path / 2
        return sum_path


class Parallel_BA:
    """
    BA 算法
    """

    def __init__(self, max_parallel):
        self.G = nx.Graph()
        self.graph_dict = nx.to_dict_of_dicts(self.G)
        self.graph_dict_key_list = list(self.graph_dict.keys())
        self.max_parallel = max_parallel

    def add_edges(self, edge_list):
        self.G.add_weighted_edges_from(edge_list)
        self.graph_dict = nx.to_dict_of_dicts(self.G)

        for node, adj in self.graph_dict.items():
            for sub_node, value in self.graph_dict[node].items():
                self.graph_dict[node][sub_node] = self.graph_dict[node][sub_node]['weight']
        self.graph_dict_key_list = list(self.graph_dict.keys())

    def union_subgraph(self, subgraph_set):
        """
        union subgraph
        :param subgraph_set:
        :return:
        """
        new_subgrsubgraph_set = {}  # 新的输出集合

        subgraph_key_list = list(subgraph_set.keys())
        # print('subgraph_key_list', subgraph_key_list)
        absorb_key = []
        index = 0
        for first_key in subgraph_key_list:
            index += 1
            for seconde_key in subgraph_key_list[index:len(subgraph_key_list) + 1]:
                if seconde_key in absorb_key:
                    pass
                else:
                    if subgraph_set[first_key].join_exist(subgraph_set[seconde_key]):
                        subgraph_set[first_key].union(subgraph_set[seconde_key])
                        absorb_key.append(seconde_key)
                    else:
                        pass

        for key, graph in subgraph_set.items():
            if key not in absorb_key:
                new_subgrsubgraph_set[key] = graph

        return new_subgrsubgraph_set

    def init_subgraph_sets(self):
        """
        initial subgraph sets
        :return:
        """
        subgraph_set = {}
        for node in list(self.G.node()):
            graph = Graph()
            graph.add_node(node)
            subgraph_set[node] = graph
        return subgraph_set

    def graph_grow(self, parallel_input):
        """
        graph growing
        :param edge_dict:
        :return:
        """
        mini_value = None
        mini_node = None
        start_node = None
        for node in list(set(parallel_input[1].node()) & set(self.graph_dict_key_list)):

            for adj_node in list(set(self.graph_dict[node]) - set(parallel_input[1].node())):

                if mini_value is None or mini_node is None or start_node is None:
                    mini_value = self.graph_dict[node][adj_node]
                    mini_node = adj_node
                    start_node = node
                else:
                    if self.graph_dict[node][adj_node] < mini_value:
                        mini_value = self.graph_dict[node][adj_node]
                        mini_node = adj_node
                        start_node = node
        parallel_input[1].add_node(mini_node)
        parallel_input[1].add_edge(start_node, mini_node)
        return (parallel_input[0], parallel_input[1])

    def boruvkaMST(self):
        """
        run algorithm
        :return:
        """
        subgraph_set = self.init_subgraph_sets()
        while len(subgraph_set) > 1:
            poll = mp.Pool(processes=self.max_parallel)
            input_collection = [(key, graph) for key, graph in subgraph_set.items()]
            output_collection = poll.map(self.graph_grow, input_collection)
            subgraph_set = {result[0]: result[1] for result in output_collection}
            subgraph_set = self.union_subgraph(subgraph_set=subgraph_set)

        self.mini_tree = list(subgraph_set.values())[0]

        return subgraph_set

    def calculate_min_tree_lenght(self):
        """
        calculate the lenght of mini tree
        :return:
        """
        sum_path = 0
        for edge in self.mini_tree.edge():
            if self.graph_dict[edge[0]][edge[1]] is not None:
                sum_path += self.graph_dict[edge[0]][edge[1]]
        sum_path = sum_path / 2
        return sum_path


def generate_random_graph(node_num=1000, link_num=50):
    """
    generate test graph
    :param node_num:
    :param link_num:
    :param p:
    :return:
    """
    seed(1)
    random_graph = nx.gnm_random_graph(node_num, link_num)
    W_E = list(nx.to_edgelist(random_graph))
    #     random_weight = np.random.randint(1, 200, len(W_E))
    random_weight = sample(range(10 * link_num), link_num)
    edge_list = []
    for weight, edge in zip(random_weight, W_E):
        edge_list.append((edge[0], edge[1], weight))
    return edge_list


def graph_generator(n, m):
    """
    generate connected weighted undirected graphs with n nodes and m edges
    input: n, m: integers
    output: returns adjacency matrix: numpy array
    """
    seed(1)
    G = nx.gnm_random_graph(n, m)
    while not nx.is_connected(G):
        G = nx.gnm_random_graph(n, m)
    weight_set = sample(range(10 * m), m)
    for i, (u, v) in enumerate(G.edges()):
        G.edges[u, v]['weight'] = weight_set[i]

    return G


if __name__ == "__main__":
    import pandas as pd

    edge_list = generate_random_graph(5000, 2500000)
    mf = BA()
    mf.add_edges(edge_list)
    start = time.time()
    last_set = mf.boruvkaMST()
    end = time.time()
    print('normal:', end - start)
    print(mf.calculate_min_tree_lenght())
    output_G = nx.Graph()
    output_G.add_edges_from(last_set[0].edge())
    edge_list = list(output_G.edges())
    edge_df = pd.DataFrame(edge_list)
    edge_df.to_csv('series_result_edge.csv')

    edge_list = generate_random_graph(5000, 2500000)
    p_mf = Parallel_BA(6)
    p_mf.add_edges(edge_list)
    start = time.time()
    p_last_set = p_mf.boruvkaMST()
    end = time.time()
    print('parallel', end - start)
    print(p_mf.calculate_min_tree_lenght())
    output_G = nx.Graph()
    output_G.add_edges_from(p_last_set[0].edge())
    edge_list = list(output_G.edges())
    edge_df = pd.DataFrame(edge_list)
    edge_df.to_csv('parallel_result_edge.csv')
