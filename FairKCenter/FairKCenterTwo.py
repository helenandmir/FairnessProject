import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.pyplot
import random
from matplotlib import pyplot as plt, colors
from math import sin, cos, sqrt, atan2, radians


class FairKCenter:
    # global
    def __init__(self,req_dic,color_list,k):
        self.dic_id_loc = {}  # A dictionary that holds the location for each point
        #self.dic_id_dis = {}  # A dictionary that holds the list of distance from each point to key point
        self.dic_id_NR = {}  # A dictionary that holds the neighborhood radius for each point
        self.dic_close_center ={}
        self.dic_dis_to_close_center ={}

        self.dic_center_ball ={}  # A dictionary that holds the ball of each center point
        self.dic_center_id_NR = {}  # A dictionary that holds the neighborhood radius for each center point
        self.col_list = ["ID", "X", "Y","Z","Longitude","Latitude","Rating_color"]
        #self.df = pd.read_csv("zomato.csv", usecols=self.col_list)
        self.df = pd.read_csv("virginia_part.csv", usecols=self.col_list)
        self.df1 = pd.read_csv("save_part.csv")

        self.req_dic = req_dic #A Dictionary that holds the constraints, the desired number of representatives from each entry

        self.K = k #number of centers
        self.num_color ={}
        for i in color_list:
            self.num_color[i]=0



    def convert_euclidean(self, id1, id2):
        R = 6373.0
        lon1 = self.df.Longitude[id1]
        lat1 = self.df.Latitude[id1]
        lon2 = self.df.Longitude[id2]
        lat2 = self.df.Latitude[id2]
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (sin(dlat / 2)) ** 2 + cos(lat1) * cos(lat2) * (sin(dlon / 2)) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    def initialization_NR_dic(self,all_point):
        for id in all_point:
            list_color = self.df1.iloc[id, 2:]
            list_color = list_color[[i for i in self.req_dic.keys() if self.req_dic[i] != 0]]
            m=max(list_color.items(), key=lambda x: x[1])
            self.dic_id_NR[id] = m[1]
            self.dic_id_loc[id] = [self.df.X[id], self.df.Y[id],self.df.Z[id]]


    def two_fair_k_center(self):
        """
          The main algorithm
          :return: set of centers point
        """
        start_time=time.time()

        all_point = list(self.df.ID)
        self.initialization_NR_dic(all_point)
        self.dic_id_NR = dict(sorted(self.dic_id_NR.items(), key=lambda item: item[1]))


        temp_dic_id_NR =  self.dic_id_NR.copy()
        list_NR_keys = list(temp_dic_id_NR.keys())
        for p in all_point:
            self.dic_dis_to_close_center[p] = sys.maxsize
        while len(list_NR_keys) != 0:
            p = list_NR_keys[0]
            signal = 0
            for s in self.dic_center_id_NR.keys():
                #dis = self.convert_euclidean(s,p)
                dis =distance.euclidean(self.dic_id_loc[s], self.dic_id_loc[p])
                if dis <= self.dic_id_NR[s]+self.dic_id_NR[p]:#
                    if dis < self.dic_dis_to_close_center[p]:
                        self.dic_dis_to_close_center[p] = dis
                        self.dic_close_center[p] = s
                    list_NR_keys.remove(p)
                    signal = 1
                    if dis < self.dic_id_NR[s]:
                        self.dic_center_ball[s].append(p)

                    break
            if signal == 0:
                self.dic_center_id_NR[p] = self.dic_id_NR[p]
                self.num_color[self.df.Rating_color[p]] +=1
                self.dic_center_ball[p] =list([])
                self.dic_dis_to_close_center[p] = 0
                self.dic_close_center[p] = p
                list_NR_keys.remove(p)

        print('time to "two_fair_k_center" end is {}'.format(time.time() - start_time))
        print("number of center is {}".format(len(self.dic_center_id_NR)))
        return len(self.dic_center_id_NR)
    def initialization_dic_dis_to_close_center(self):
        start_time=time.time()
        for p in list(self.df.ID):
            dic_close_center = {}
            for c in self.dic_center_id_NR.keys():
                #dic_close_center[c] = self.convert_euclidean(c,p)
                dic_close_center[c] = distance.euclidean(self.dic_id_loc[c], self.dic_id_loc[p])
            self.dic_dis_to_close_center[p] = dic_close_center[min(dic_close_center, key=dic_close_center.get)]
            self.dic_close_center[p] = min(dic_close_center, key=dic_close_center.get)

        #print('time to "initialization_dic_dis_to_close_center" end is {}'.format(time.time() - start_time))

    def results(self):
        """
             Displays the results
       """
        start_time=time.time()

        self.initialization_dic_dis_to_close_center()
        dic_means = {} # minimize the sum of the squares of the travel distances
        dic_medians = {}# minimize the average travel distance
        dic_centers = {} # minimize the maximum travel distance among individuals in P
        dic_alpha = {}
        dic_nearest_centers={}
        # for each point found is nearest centers
        for id in self.dic_id_loc.keys():

            dic_means[id] = np.power(self.dic_dis_to_close_center[id], 2)
            dic_medians[id] = self.dic_dis_to_close_center[id]
            dic_alpha[id] = self.dic_dis_to_close_center[id] / self.dic_id_NR[id]
            #dic_nearest_centers[id] = min(list_temp, key=list_temp.get)

        means_result = np.sum(list(dic_means.values()))
        medians_result = np.sum(list(dic_medians.values())) / len(list(dic_medians.values()))
        centers_result = np.max(list(dic_medians.values()))
        print("->>> max alpha information <<<-")
        print('max alpha is {}'.format(max(dic_alpha.values())))
        point_with_max_alpha = max(dic_alpha, key=dic_alpha.get)
        cen =self.dic_close_center[point_with_max_alpha]
        print('the point with max alpha is: {} in place {} from center point {} in place {}'.format(point_with_max_alpha,self.dic_id_loc[point_with_max_alpha],cen,self.dic_id_loc[cen]))
        #d= self.convert_euclidean(point_with_max_alpha, cen)
        d = distance.euclidean(self.dic_id_loc[point_with_max_alpha], self.dic_id_loc[cen])
        print('the distance between the point with maximum alpha to his nearest center is {}'.format(d))
        print('the NR of  point with maximum alpha is {}'.format(self.dic_id_NR[point_with_max_alpha]))
        print("->>> other information(k-means,k-medians,k-centers) <<<-")
        print('max NR {} from point {}'.format(max(self.dic_id_NR.values()),max(self.dic_id_NR,key=self.dic_id_NR.get)))
        print('the k-means result is {}'.format(means_result))
        print('the k-medians result is {}'.format(medians_result))
        print('the k-centers result is {}'.format(centers_result))

        print('time to "result" end is {}'.format(time.time() - start_time))
        return max(dic_alpha, key=dic_alpha.get)

    def initialization_ball_center(self):
         all_point = list(self.df.ID)
         for s in self.dic_center_id_NR.keys():
             self.dic_center_ball[s] = list([])

             loc_s = self.dic_id_loc[s]
             NR_s = self.dic_center_id_NR[s]
             all_point_temp = all_point
             for p in all_point_temp:

                if distance.euclidean(loc_s, self.dic_id_loc[p])<= NR_s:#self.convert_euclidean(s, p)<= NR_s :
                    self.dic_center_ball[s].append(p)
                    #all_point.remove(p)
             #print("{} ball is {}".format(s,self.dic_center_ball[s]))
         sum =0
         for s in self.dic_center_id_NR.keys():
             sum = sum +len(list(self.dic_center_ball[s]))
         #print("sum ={}".format(sum))

    def swap_center(self):
        del_centers = []
        new_centers = []
        for s in self.dic_center_id_NR.keys():
            c = self.df.Rating_color[s]
            if self.num_color[c] > self.req_dic[c]:
                center_candidates = [i for i in self.dic_center_ball[s] if
                                     self.num_color[self.df.Rating_color[i]] < self.req_dic[self.df.Rating_color[i]]]

                # print("{} ball is {}".format(s,fair.dic_center_ball[s] ))

                del_centers.append(s)
                new_centers.append(center_candidates[0])
                self.num_color[c] = self.num_color[c] - 1
                self.num_color[self.df.Rating_color[center_candidates[0]]] = self.num_color[self.df.Rating_color[
                    center_candidates[0]]] + 1
        for s in del_centers:
            self.dic_center_id_NR.pop(s)
        for n in new_centers:
            self.dic_center_id_NR[n] = self.dic_id_NR[n]
        for p in self.df.ID:
            for n in self.dic_center_id_NR.keys():
                dis = distance.euclidean(self.dic_id_loc[p], self.dic_id_loc[n])
                if dis <= self.dic_dis_to_close_center[p]:
                    self.dic_dis_to_close_center[p] = dis
                    self.dic_close_center[p] = n

    def plot_point(self):
        x_c =list(self.df.Longitude[self.dic_center_id_NR.keys()])
        y_c = list(self.df.Latitude[self.dic_center_id_NR.keys()])
        c_c = list(self.df.Rating_color[self.dic_center_id_NR.keys()])
        size_c = [100]*len(x_c)
        x = list(self.df.Longitude)
        y = list(self.df.Latitude)
        c = list(self.df.Rating_color)
        s_p = [30]*len(x)

        plt.scatter(x, y, c=c, marker='.',s=s_p,cmap='viridis')
        plt.scatter(x_c, y_c, c=c_c, marker='*',s= size_c,edgecolors='black')
        plt.show()

def main() :
    k = 100
    k_temp = k
    col_list = ["Rating_color"]
    df1 = pd.read_csv("virginia_part.csv", usecols=col_list)
    type_list = list(df1.Rating_color)
    type_set = list(set(type_list))
    num_type = len(type_set)
    color_list = list(matplotlib.colors.cnames.keys())[0:num_type]
    req_dic = {}
    center_colors = {}
    dic_orig = {}


    #Creating relative requirements dictionary

    for c in color_list:
        req_dic[c]= np.ceil((list(df1.Rating_color).count(c)/len(df1.Rating_color))*k)
        if req_dic[c] < 2:
            req_dic[c]=0
        dic_orig[c]=(list(df1.Rating_color).count(c)/len(df1.Rating_color))*k
        center_colors[c]=0
    '''
    # Creating random requirements dictionary
    while len(color_list) > 0:
        c = random.choice(color_list)
        range_num = min(list(df1.Rating_color).count(c), k_temp)
        num_random = random.randint(0, range_num)
        req_dic[c] = num_random
        k_temp -= num_random
        center_colors[c] = 0

        color_list.remove(c)
 '''
    print("requirement dictionary:")
    print(req_dic)
    color_list = list(matplotlib.colors.cnames.keys())[0:num_type]

    fair = FairKCenter(req_dic,color_list,k)
    fair.two_fair_k_center()
    print("center points before algorithm 2")
    print(fair.dic_center_id_NR.keys())
    #fair.count_center_from_each_color()
    fair.swap_center()
    print(fair.dic_center_id_NR.keys())
    print(list(fair.df.Rating_color[fair.dic_center_id_NR.keys()]))
    print(len(fair.dic_center_id_NR.keys()))
    #fair.dic_center_id_NR = fair.dic_replace_center_NR.copy()
    fair.results()
    print(center_colors)
    print('time to end is {}'.format(time.time() - start_time))
    fair.plot_point()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('time to end is {}'.format(time.time() - start_time))

