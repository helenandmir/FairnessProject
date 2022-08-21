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
        self.list_nodes =[]
        self.list_edges = []
        self.dic_new_center={}

    def reduse_points(self):
       for i in self.dic_ball_centers:
           ball_copy = self.dic_ball_centers[i].copy()
           for j in ball_copy:
               if self.dic_colors[self.df.Colors[j]] == 0:
                  self.dic_ball_centers[i].remove(j)

    def add_points(self):
        self.dic_new_ball = {}
        non_zero_colors = [i for i in self.dic_colors if i!=0]
        for i in self.dic_ball_centers:
            self.dic_new_ball [i] =list([])
            color_list = self.df.Colors[self.dic_ball_centers[i]]
            color_list= list(set(dict(color_list).values()))
            for c in color_list:
                list_temp = [c+"#"+str(i)]*int(self.dic_colors[c])
                list_end = [list_temp[l] +"#" + str(l) for l in range(0, len(list_temp))]
                self.dic_new_ball[i].extend(list_end)

    def create_nodes(self):
        for i in self.dic_new_ball:
            self.list_nodes.extend(self.dic_new_ball[i])
    def create_edges(self):
        for i in self.dic_new_ball.keys():
            list_same_ball = list([j for j in self.dic_new_ball[i] if str(i) in j])
            self.list_edges.extend(list([(a, b) for idx, a in enumerate(list_same_ball) for b in list_same_ball[idx + 1:]]))
        non_zero_colors = [i for i in self.dic_colors if self.dic_colors[i] != 0]
        for c in non_zero_colors:
            list_same_color = list([j for j in self.list_nodes if c in j])
            self.list_edges.extend(list([(a, b) for idx, a in enumerate(list_same_color) for b in list_same_ball[idx + 1:]]))
    def return_center(self,list_point):
        for i in self.dic_ball_centers:
            str_s = "#" + str(i) + "#"
            for j in list_point:
                if str_s in j:
                    res = j.split("#")[0]
                    self.dic_new_center[i] = res
                    break
    def create_graph(self):
        G = networkx.Graph()
        G.add_edges_from(self.list_edges)
        p = maximal_independent_set(G, seed=0)
        print(p)
        # for i in range(0, 10):
        #     p=maximal_independent_set(G, seed=i)
        #     print(P)
        #     print(len(P))
        self.return_center(p)







