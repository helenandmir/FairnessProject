import sys
import time
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.pyplot
import random
from matplotlib import pyplot as plt, colors
from math import sin, cos, sqrt, atan2, radians
from scipy.spatial import KDTree
import math
import CreateGraph
import matplotlib.pyplot as plt
import MaxMatching
fileA = "../DataSet/Listings.csv"
fileB = "../DataSet/Listings_radius_100.csv"

class FairKCenter:
    def __init__(self,req_dic,color_list,k):
        self.req_dic = req_dic  # A Dictionary that holds the constraints, the desired number of representatives from each entry
        self.col_list1 = [i for i in self.req_dic if self.req_dic[i] != 0]
        self.col_list1.append("ID")
        self.dic_id_loc = {}  # A dictionary that holds the location for each point
        #self.dic_id_dis = {}  # A dictionary that holds the list of distance from each point to key point
        self.dic_id_NR = {}  # A dictionary that holds the neighborhood radius for each point
        self.dic_close_center ={}
        self.dic_dis_to_close_center ={}

        self.dic_center_ball ={}  # A dictionary that holds the ball of each center point
        self.ball ={}
        self.dic_center_id_NR = {}  # A dictionary that holds the neighborhood radius for each center point
        self.dic_center_loc={}
        self.col_list = ["ID", "X", "Y","Z","Longitude","Latitude","Colors"]
        #self.df = pd.read_csv("zomato.csv", usecols=self.col_list)
        self.df = pd.read_csv(fileA, usecols=self.col_list)
        self.df1 = pd.read_csv(fileB,usecols=self.col_list1)
        self.df2 = pd.read_csv(fileB,usecols=["ID","NR_TYPE_ONE"])


        self.K = k #number of centers
        self.num_color ={}
        for i in color_list:
            self.num_color[i]=0

    def initialization_NR_dic(self,all_point):
        start_time=time.time()

        non_zero_color=[i for i in self.req_dic if self.req_dic[i]!=0]

        for i in all_point:
            min = 10000
            for j in range(1,len(non_zero_color)+1):
                if self.df1.iat[i,j] <=min:
                    min=self.df1.iat[i,j]
            self.dic_id_NR[i] = max(self.df2.iat[i,1], min)
            self.dic_id_loc[i] = [self.df.X[i], self.df.Y[i], self.df.Z[i]]

        print('time to "initialization_NR_dic" end is {}'.format(time.time() - start_time))

    def fair_k(self, alpha):
        balance = len(self.df.ID) / self.K
        start_time = time.time()
        self.dic_id_NR = dict(sorted(self.dic_id_NR.items(), key=lambda item: item[1]))
        temp_dic_id_NR = self.dic_id_NR.copy()
        temp_dic_id_NR2 = self.dic_id_NR.copy()

        list_NR_keys = list(temp_dic_id_NR.keys())
        while len(self.dic_center_id_NR.keys()) != self.K and len(temp_dic_id_NR.keys()) != 0:
            # print(len(self.dic_center_id_NR.keys()))
            p = min(temp_dic_id_NR, key=temp_dic_id_NR.get)
            self.dic_center_id_NR[p] = self.dic_id_NR[p]
            self.dic_center_loc[p] = self.dic_id_loc[p]
            self.dic_center_ball[p] = list([])
            self.dic_dis_to_close_center[p] = 0
            temp_dic_id_NR.pop(p)
            for i in temp_dic_id_NR2:
                dis = distance.euclidean(self.dic_id_loc[i], self.dic_center_loc[p])
                if dis <= self.dic_id_NR[i] + self.dic_center_id_NR[p]:  # alpha*
                    if i in temp_dic_id_NR:
                        temp_dic_id_NR.pop(i)
            temp_dic_id_NR2 = temp_dic_id_NR.copy()
        print('time to "two_fair_k_center" end is {}'.format(time.time() - start_time))
        print("number of center is {}".format(len(self.dic_center_id_NR)))
        return len(self.dic_center_id_NR)
    def fair_k_2(self, alpha):
        balance = len(self.df.ID)/self.K
        start_time = time.time()
        self.dic_id_NR = dict(sorted(self.dic_id_NR.items(), key=lambda item: item[1]))
        temp_dic_id_NR = self.dic_id_NR.copy()

        while len(self.dic_center_id_NR.keys()) != self.K and len(temp_dic_id_NR.keys())!=0:
            #print(len(self.dic_center_id_NR.keys()))
            p = min(temp_dic_id_NR, key=temp_dic_id_NR.get)
            self.dic_center_id_NR[p] = self.dic_id_NR[p]
            self.dic_center_loc[p] = self.dic_id_loc[p]
            self.dic_center_ball[p] = list([])
            self.dic_dis_to_close_center[p] = 0
            temp_dic_id_NR.pop(p)

            ##########
            list_t = list(temp_dic_id_NR.keys())
            tuple_color = tuple(zip(list(self.df.X[list_t]), list(self.df.Y[list_t]),list(self.df.Z[list_t])))
            tree_c = KDTree(np.array(list(tuple_color)))
            n = len(list_t)
            dist_c, ind_c = tree_c.query([[self.df.X[p], self.df.Y[p],self.df.Z[p]]], n)
            for dis,i in zip(dist_c[0],ind_c[0]):
                if dis <= self.dic_id_NR[list_t[i]] +self.dic_id_NR[p]:
                    temp_dic_id_NR.pop(list_t[i])



            #########

        print('time to "two_fair_k_center" end is {}'.format(time.time() - start_time))
        print("number of center is {}".format(len(self.dic_center_id_NR)))
        return len(self.dic_center_id_NR)
    def two_fair_k_center(self, alpha):
        """
        The main algorithm
        :return: set of centers point
                  """
        start_time = time.time()
        ball ={}
        self.dic_id_NR = dict(sorted(self.dic_id_NR.items(), key=lambda item: item[1]))
        temp_dic_id_NR = self.dic_id_NR.copy()
        list_NR_keys = list(temp_dic_id_NR.keys())
        for p in list(self.df.ID):
            self.dic_dis_to_close_center[p] = sys.maxsize  # Initialize each point at an infinite distance from the center
        while len(list_NR_keys) != 0:
            p = list_NR_keys[0]
            signal = 0  # 0 if p can be center point and 1 otherwise
            for s in self.dic_center_id_NR.keys():
                dis = distance.euclidean(self.dic_id_loc[s], self.dic_id_loc[p])
                if dis <= self.dic_id_NR[s]+self.dic_id_NR[p]:  #1.399*

                    list_NR_keys.remove(p)
                    signal = 1

                    break
            if signal == 0:  # update P to be new center
                self.dic_center_id_NR[p] = self.dic_id_NR[p]
                self.dic_center_loc[p] = self.dic_id_loc[p]
                self.dic_center_ball[p] = list([])
                # #self.dic_center_ball[p].append(p)
                # self.ball[p] =list([])
                # self.ball[p].append(p)
                self.dic_dis_to_close_center[p] = 0
                self.dic_close_center[p] = p
                list_NR_keys.remove(p)
        print('time to "two_fair_k_center" end is {}'.format(time.time() - start_time))
        print("number of center is {}".format(len(self.dic_center_id_NR)))


        return len(self.dic_center_id_NR)



    def update_dis_from_center(self):

        start_time = time.time()
        for p in list(self.df.ID):
            # arr_coordinates = np.array(self.dic_id_loc)
            tree = KDTree(np.array(list(self.dic_center_loc.values())))
            dist, ind = tree.query([[self.df.X[p], self.df.Y[p], self.df.Z[p]]], 1)
            self.dic_dis_to_close_center[p] = dist[0]
            self.dic_close_center[p] = np.array(list(self.dic_center_loc.keys()))[ind[0]]
        print('time to "update_dis_from_center" end is {}'.format(time.time() - start_time))

    def results2(self):
        self.update_dis_from_center()
        start_time=time.time()
        # ##########
        # dic_alpha={}
        # for k,v in self.dic_id_NR.items():
        #     if self.dic_dis_to_close_center[k] == 0 and  self.dic_id_NR[k]==0:
        #         print("if")
        #         dic_alpha[k] = 1
        #     elif self.dic_id_NR[k] == 0:
        #         print("######################elif###########3")
        #         dic_alpha[k] = math.inf
        #     else:
        #         dic_alpha[k]=self.dic_dis_to_close_center[k]/ self.dic_id_NR[k]
        #
        # ##########
        dic_alpha = dict([(k,self.dic_dis_to_close_center[k]/ self.dic_id_NR[k]) for k, v in self.dic_id_NR.items()])
        max_key = max(dic_alpha, key=lambda x: dic_alpha[x])
        print("The point with maximum alpha is:{}".format(max_key))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_key],self.dic_id_NR[max_key]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_key],self.dic_dis_to_close_center[max_key]))
        print("number of center is {}".format(len(self.dic_center_id_NR)))
        print("#################")
        max_dis = max(self.dic_dis_to_close_center, key=self.dic_dis_to_close_center.get)
        print("The point with maximum distance from his nearest center is:{} in {}".format(max_dis, (
            self.df.Longitude[max_dis], self.df.Latitude[max_dis])))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_dis], self.dic_id_NR[max_dis]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_dis],
                                                                      self.dic_dis_to_close_center[max_dis]))

        print("#################")
        max_key = max(self.dic_id_NR, key=lambda x: self.dic_id_NR[x])
        print("The point with maximum NR is {} is NR is {}".format(max_key, self.dic_id_NR[max_key] ))

        print(self.dic_center_id_NR.keys())
        print('time to "results2" end is {}'.format(time.time() - start_time))


    def initialization_ball_center2(self):

        tuple_color = tuple(zip(list(self.df.X[self.df.ID]), list(self.df.Y[self.df.ID]), list(self.df.Z[self.df.ID])))
        tree_c = KDTree(np.array(list(tuple_color)))
        for c in self.dic_center_id_NR:
            dist_c, ind_c = tree_c.query(list(self.dic_center_loc[c]),len(list(self.df.ID)), distance_upper_bound=self.dic_center_id_NR[c]+1)
            dis = [i for i in dist_c if i != math.inf]
            ind = ind_c[0:len(dis)]
            self.dic_center_ball[c] = list(ind)


    def plot_point(self):
        x_c =list(self.df.Longitude[self.dic_center_id_NR.keys()])
        y_c = list(self.df.Latitude[self.dic_center_id_NR.keys()])
        c_c = list(self.df.Colors[self.dic_center_id_NR.keys()])
        size_c = [100]*len(x_c)
        x = list(self.df.Longitude)
        y = list(self.df.Latitude)
        c = list(self.df.Colors)
        s_p = [30]*len(x)

        plt.scatter(x, y, c=c, marker='.',s=s_p,cmap='viridis', edgecolors='gray')
        plt.scatter(x_c, y_c, c=c_c, marker='*',s= size_c,edgecolors='black')

        plt.show()
    def replace_center(self,new_center):
        temp_dic = self.dic_center_id_NR.copy()

        for i in temp_dic:
            if i in new_center.keys():
                self.dic_center_id_NR.pop(i)
                self.dic_center_id_NR[new_center[i]] = self.dic_id_NR[new_center[i]]
                self.dic_center_loc.pop(i)
                self.dic_center_loc[new_center[i]] = self.dic_id_loc[new_center[i]]

                self.dic_center_ball[new_center[i]] = self.dic_center_ball[i]
                self.dic_center_ball.pop(i)
            else:
                self.dic_center_id_NR.pop(i)
                self.dic_center_loc.pop(i)
                self.dic_center_ball.pop(i)



    def add_center(self,color_center):
        point_in_balls =[]
        for i in self.dic_center_ball:
            point_in_balls = point_in_balls + self.dic_center_ball[i]
        points_no_center = list(set(self.dic_id_NR.keys()) - set(point_in_balls))
        for i in self.dic_center_id_NR:
            col =self.df.Colors[i]
            if col in color_center:
                self.req_dic[col] -=1
        for col in self.req_dic:
            if self.req_dic[col] != 0:
                col_list = [i for i in points_no_center if self.df.Colors == col]
                res = {key: self.dic_center_id_NR[key] for key in col_list}
                new_center = min(res, key=res.get)
                self.dic_center_id_NR[new_center]=self.dic_id_NR[new_center]
                self.dic_center_loc[new_center]=self.dic_id_NR[new_center]
                self.req_dic[col] -=1
                self.dic_center_ball[new_center]=[]
                loc_c = self.dic_center_loc[new_center]
                NR_c = self.dic_center_loc[new_center]
                for p in points_no_center:
                    if distance.euclidean(loc_c, self.dic_id_loc[p]) - NR_c <=0.000000001:
                       self.dic_center_ball[new_center].append(p)
                       points_no_center.pop(p)
    def save_group(self):
        dic_colors = {}
        for c in self.req_dic:
            dic_colors[c] = list([])
        for i in self.dic_center_ball:
            for c in self.req_dic:
                dic_colors[c].append(0)
            for j in self.dic_center_ball[i]:
                dic_colors[self.df.Colors[j]]+=1


def main() :
    start_time = time.time()

    k = 100
    k_temp =k
    col_list = ["Colors"]
    df1 = pd.read_csv(fileA, usecols=col_list)
    type_list = list(df1.Colors)
    type_set = list(set(type_list))
    num_type = len(type_set)
    color_list = list(set(df1.Colors))#list(matplotlib.colors.cnames.keys())[0:num_type]
    req_dic = {}
    center_colors = {}
    dic_orig = {}
    #Creating relative requirements dictionary

    for c in color_list:
        req_dic[c]= np.ceil((list(df1.Colors).count(c)/len(df1.Colors))*k)
        if req_dic[c] < 3:
            req_dic[c]=0
        dic_orig[c]=(list(df1.Colors).count(c)/len(df1.Colors))*k
        center_colors[c]=0
    req_dic["purple"] =5
    # #Creating random requirements dictionary
    # while len(color_list) > 0:
    #     c = random.choice(color_list)
    #     range_num = min(list(df1.Colors).count(c), k_temp)
    #     num_random = random.randint(0, range_num)
    #     req_dic[c] = num_random
    #     k_temp -= num_random
    #     center_colors[c] = 0
    #     color_list.remove(c)



    print(sum(req_dic.values()))
    print("requirement dictionary:")
    print(req_dic)
    color_list = list(df1.Colors)#list(matplotlib.colors.cnames.keys())[0:num_type]

    fair = FairKCenter(req_dic,color_list,k)
    fair.initialization_NR_dic(list(fair.df.ID))

    fair.fair_k_2(0.01)
    #fair.two_fair_k_center(0.01)


    fair.initialization_ball_center2()
    C = MaxMatching.CR(fair.req_dic, fileA, fair.dic_center_ball)
    C.add_nodes1()
    C.add_nodes2()
    C.colors_in_balls()
    C.add_edegs()
    new_center = C.create_graph()


    print(new_center)
    fair.replace_center(new_center)
    print([fair.df.Colors[id] for id in fair.dic_center_id_NR.keys()])
    #fair.add_center(IS.dic_color_center.keys())
    fair.results2()
    print('time to end is {}'.format(time.time() - start_time))
    fair.plot_point()
    # low =1
    # high=2
    # while low <= high:
    #     mid = (low+high)/2
    #     print("alpha(mid) ={}".format(mid))
    #     fair = FairKCenter(req_dic,color_list,k)
    #     fair.initialization_NR_dic(list(fair.df.ID))
    #     if fair.two_fair_k_center(mid) < fair.K:
    #         high = mid
    #     elif fair.two_fair_k_center(mid) == fair.K:
    #         fair.initialization_ball_center2()
    #         C = CreateGraph.CR(fair.req_dic, "virginia_new.csv", fair.dic_center_ball)
    #         C.reduse_points()
    #         C.add_points()
    #         C.create_nodes()
    #         C.create_edges()
    #         C.create_graph()
    #         fair.results2()
    #
    #         fair.plot_point()
    #
    #         break
    #     else:
    #         low =mid


if __name__ == '__main__':
    main()

