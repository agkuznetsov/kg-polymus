import os

import networkx as nx

# from textii.gvec import GVector


def path(*elem, expanduser=True, initial_path='e:\\'):
    if expanduser:
        return os.path.expanduser(os.path.join('~', *elem))
    else:
        return os.path.join(initial_path, *elem)

class Graph():
    def __init__(self):
        # self.g = GVector()
        pass

    def load(self, *name):
        self.graph = nx.read_graphml(path('data', 'graph', 'Polytech_graphs', *name))
        
    def get_edge_labels(self, node=None):
        labels = {}
        for n in self.graph.nodes:
            for x in g.graph[n]:
                try:
                    labels.update({(n, x): g.graph[n][x]['label']})
                except Exception as e:
                    pass

        return labels

    def get_node_labels(self, node=None):
        labels = {}
        for node, prop in self.graph.nodes.items():
            try:
                labels.update({node: prop['label']})
            except Exception as e:
                pass

        return labels


if __name__ == '__main__':
    try:
        g = Graph()
        g.load('Polytech_total.graphml')

        # nx.draw(g.graph)
        # plt.show()
        
        res_n = g.get_node_labels()
        res_e = g.get_edge_labels()

        # sim= g.g.similar('физика')

        pass
    except Exception as e:
        print(e)
