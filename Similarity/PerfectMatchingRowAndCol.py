import pandas as pd
import sys
import networkx
from networkx.algorithms.mis import maximal_independent_set
sys.setrecursionlimit(50000)
class CR:
    def __init__(self, Matrix):
        self.Matrix =Matrix

        self.list_nodes_row =[]
        self.list_nodes_col =[]

        self.list_edges = []


    def add_nodes_row(self):
        """
        Node for row
        :return:
        """
        for i in range(len(self.Matrix)):
            self.list_nodes_row.append(str(i)+"r")

    def add_nodes_col(self):
        """
        Node for col
        :return:
        """
        for i in range(len(self.Matrix)):
            self.list_nodes_col.append(str(i)+"c")

    def add_edegs(self):
        for r in range(len(self.Matrix)):
            for c in range(len(self.Matrix)):
                if self.Matrix[r,c]==0:
                    row = str(r)+"r"
                    col =  str(c)+"c"
                    self.list_edges.append((row,col))

    def ret_center(self, match):
        new_dict=[]
        for pair in match:
            if "c" in pair[0]:
                pair = (pair[1], pair[0])
            new_dict.append(pair)
        new_dict= dict(new_dict)
        return new_dict

    def create_graph(self):
        G = networkx.Graph()
        G.add_edges_from(self.list_edges)
        P=networkx.max_weight_matching(G)
        #p = maximal_independent_set(G, seed=0)
        print(len(P))

        return dict(P)

    def get_res(self):
        G = networkx.Graph()
        G.add_edges_from(self.list_edges)
        P=networkx.max_weight_matching(G)
        #p = maximal_independent_set(G, seed=0)

        res = self.ret_center(P)
        return res











