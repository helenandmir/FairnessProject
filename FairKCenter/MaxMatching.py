import pandas as pd
import sys
import networkx
from networkx.algorithms.mis import maximal_independent_set
sys.setrecursionlimit(50000)
class CR:
    def __init__(self, req_dic, file,ball_dic):
        self.dic_ball_centers = ball_dic
        self.dic_colors =req_dic
        self.df = pd.read_csv(file)
        self.dic_colors_in_ball ={}
        self.dic_new_set ={}
        self.dic_num_colors={}
        self.dic_color_center={}
        self.dic_new_ball ={}
        self.list_nodes1 =[]
        self.list_nodes2 =[]

        self.list_edges = []
        self.dic_new_center={}


    def add_nodes1(self):
        for i in self.dic_colors.keys():
            list_temp = [i + str(j) for j in range(int(self.dic_colors[i]))]
            self.list_nodes1 = self.list_nodes1+list_temp
    def add_nodes2(self):
        for i in self.dic_ball_centers:
            self.list_nodes2.append(i)

    def add_edegs(self):
        for i in self.list_nodes2:
            for c in self.dic_colors_in_ball[i]:
                nodes = [n for n in self.list_nodes1 if c == "".join([i for i in n if not i.isdigit()])]
                self.list_edges.extend((i,b) for b in nodes)

    def colors_in_balls(self):
        for i in self.dic_ball_centers:
            self.dic_colors_in_ball[i] = list(set([self.df.Colors[j] for j in self.dic_ball_centers[i]]))

    def ret_center(self,match):
        for pair in match:
            if type(pair[1]) == str:
                pair = (pair[1], pair[0])
            res =[i for i in self.dic_ball_centers[pair[1]] if self.df.Colors[i] == ''.join([i for i in pair[0] if not i.isdigit()])]
            self.dic_new_center[pair[1]] = res[0]

    def create_graph(self):
        G = networkx.Graph()
        G.add_edges_from(self.list_edges)
        P=networkx.max_weight_matching(G)
        #p = maximal_independent_set(G, seed=0)
        print(P)
        print("max_match ={}".format(len(P)))
        # colors_dic={}
        # for c in self.dic_colors:
        #     colors_dic[c] = len([i for i in P if c in str(i[0])])
        # print("colors_dic ={}".format(colors_dic))


        self.ret_center(P)
        # for i in range(0, 10):
        #     p=maximal_independent_set(G, seed=i)
        #     print(P)
        #     print(len(P))
        return self.dic_new_center







