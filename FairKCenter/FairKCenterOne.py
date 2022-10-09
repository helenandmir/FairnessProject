import math
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

fileA = "../DataSet/Point.csv"
fileB = "../DataSet/Point_radius_500.csv"
k = 494
class FairKCenter:
    # global

    def __init__(self, k):
        self.dic_id_loc = {}  # A dictionary that holds the location for each point
        self.dic_id_NR = {}  # A dictionary that holds the neighborhood radius for each point
        self.dic_close_center = {}
        self.dic_dis_to_close_center = {}

        self.dic_center_ball = {}  # A dictionary that holds the ball of each center point
        self.dic_center_id_NR = {}  # A dictionary that holds the neighborhood radius for each center point
        self.dic_center_loc = {}
        self.col_list = ["ID", "X", "Y", "Z", "Longitude", "Latitude", "Colors"]
        self.df = pd.read_csv(fileA, usecols=self.col_list)
        self.df1 = pd.read_csv(fileB)
        self.K = k  # number of centers

    def initialization_NR_dic(self):
        start_time = time.time()
        self.dic_id_NR = dict(self.df1.NR_TYPE_ONE)
        self.dic_id_loc = list(np.array(list(zip(self.df.X, self.df.Y,self.df.Z))))

        print('time to "initialization_NR_dic" end is {}'.format(time.time() - start_time))

    def fair_k(self, alpha):
        balance = len(self.df.ID)/self.K
        start_time = time.time()
        #self.dic_id_NR = dict(sorted(self.dic_id_NR.items(), key=lambda item: item[1]))
        temp_dic_id_NR = self.dic_id_NR.copy()
        temp_dic_id_NR2 = self.dic_id_NR.copy()

        list_NR_keys = list(temp_dic_id_NR.keys())
        while len(self.dic_center_id_NR.keys()) != self.K and len(temp_dic_id_NR.keys())!=0:
            #print(len(self.dic_center_id_NR.keys()))
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

        while len(temp_dic_id_NR.keys())!=0:
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
            if len(tuple_color) == 0:
                break
            tree_c = KDTree(np.array(list(tuple_color)))
            n = len(list_t)
            dist_c, ind_c = tree_c.query([[self.df.X[p], self.df.Y[p],self.df.Z[p]]], n)
            for dis,i in zip(dist_c[0],ind_c[0]):
                if dis <= alpha*self.dic_id_NR[list_t[i]] :#+self.dic_id_NR[p]

                    temp_dic_id_NR.pop(list_t[i])



            #########

        print('time to "two_fair_k_center" end is {}'.format(time.time() - start_time))
        #print("number of center is {}".format(len(self.dic_center_id_NR)))
        return len(self.dic_center_id_NR)

    def two_fair_k_center(self, alpha):
        """
         The main algorithm
        :return: set of centers point
                  """
        start_time = time.time()

        self.dic_id_NR = dict(sorted(self.dic_id_NR.items(), key=lambda item: item[1]))
        temp_dic_id_NR = self.dic_id_NR.copy()
        list_NR_keys = list(temp_dic_id_NR.keys())
        for p in list(self.df.ID):
            self.dic_dis_to_close_center[
                p] = sys.maxsize  # Initialize each point at an infinite distance from the center
        while len(list_NR_keys) != 0:
            p = list_NR_keys[0]
            signal = 0  # 0 if p can be center point and 1 otherwise
            for s in self.dic_center_id_NR.keys():
                dis = distance.euclidean(self.dic_id_loc[s], self.dic_id_loc[p])
                if dis <= self.dic_id_NR[s] + self.dic_id_NR[p]:  # alpha*

                    list_NR_keys.remove(p)
                    signal = 1

                    break
            if signal == 0:  # update P to be new center
                self.dic_center_id_NR[p] = self.dic_id_NR[p]
                self.dic_center_loc[p] = self.dic_id_loc[p]
                self.dic_center_ball[p] = list([])
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
            self.dic_dis_to_close_center[p] = dist[0][0]
            self.dic_close_center[p] = np.array(list(self.dic_center_loc.keys()))[ind[0][0]]
        print('time to "update_dis_from_center" end is {}'.format(time.time() - start_time))
    def uniform_calculation(self):
        num_center = len(self.dic_center_id_NR)
        uni = len(set(self.df.Colors))/num_center
        colors_list = list(set(self.df.Colors))
        result_arr ={}
        distribution_arr = {}
        result=0
        center_color_num = {}
        color_num={}
        for c in colors_list:
            color_num[c] = len([i for i in self.df.ID if self.df.Colors[i]==c])
            center_color_num[c] = len([i for i in self.dic_center_id_NR.keys() if self.df.Colors[i]==c])
        for c in colors_list:
            distribution_arr[c] = (center_color_num[c]/num_center)
            #result_arr[c] = abs(uni-(center_color_num[c]/num_center))
            result += abs(uni-(center_color_num[c]/num_center))
        print("The number of points of each color in the data:{}".format(color_num))
        print("The number of center of each color in the result:{}".format(center_color_num))
        #print("result_arr list ={}".format(result_arr))
        print("distribution_arr ={}".format(distribution_arr))
        print("uniform={}".format(uni))
        result = result/len(colors_list)
        return result

    def relative_calculation(self):
        result_arr ={}
        color_list = list(set(self.df.Colors))
        result =0
        dic_rel ={}
        req_result={}
        num_color = len(color_list)
        for c in color_list:
            num_c = len([i for i in self.df.ID if self.df.Colors[i]==c])
            dic_rel[c] = num_c/len(self.df.ID)
            req_result[c]=0
        for i in self.dic_center_id_NR:
            c = self.df.Colors[i]
            req_result[c] += 1
        for c in color_list:
            result_arr[c]=abs(dic_rel[c] - (req_result[c]/len(self.dic_center_id_NR)))
            result += abs(dic_rel[c] - (req_result[c]/len(self.dic_center_id_NR)))

        result=result / num_color
        print("result arr rel ={}".format(result_arr))
        return result
    def result_distance(self):
        k=500
        num_centers = len(self.dic_center_id_NR)
        colors_list = list(set(self.df.Colors))
        num_point = len(self.df.Colors)
        relative_dic={}
        uniform_dic={}
        result_dis_dic= {}
        uni = math.floor(k / len(colors_list))
        for c in colors_list:
            uniform_dic[c]=uni
            relative_dic[c] =math.floor((list(self.df.Colors).count(c) /len(self.df.Colors)) *k)
            result_dis_dic[c] = len([i for i in self.dic_center_id_NR.keys() if self.df.Colors[i] ==c])
        #calculat distance
        dis = math.sqrt(sum(((relative_dic.get(d,0)/num_centers) - (result_dis_dic.get(d,0)/num_centers))**2 for d in set(relative_dic) | set(result_dis_dic)))
        dis2 = math.sqrt(sum(((uniform_dic.get(d, 0) / num_centers) - (result_dis_dic.get(d, 0) / num_centers)) ** 2 for d in set(uniform_dic) | set(result_dis_dic)))
        print("relative_dic: {}".format(relative_dic))
        print("uniform_dic: {}".format(uniform_dic))
        print("result_dis_dic: {}".format(result_dis_dic))
        return dis,dis2

    def result_distance2(self):
        k=500
        num_centers = len(self.dic_center_id_NR)
        colors_list = list(set(self.df.Colors))
        rand_list=["antiquewhite","blueviolet","bisque","burlywood","black","cadetblue","blue","azure","aqua","blanchedalmond"]
        rand_dic_cal ={}
        rand_dic={}
        result_dis_dic={}
        for c in rand_list:
            rand_dic_cal[c] =list(self.df.Colors).count(c)
        for c in colors_list:
            result_dis_dic[c] = len([i for i in self.dic_center_id_NR.keys() if self.df.Colors[i] == c])
            if c in rand_list:
                rand_dic[c] = math.floor((rand_dic_cal[c]/sum(rand_dic_cal.values()))*k)
            else:
                rand_dic[c]=0

        dis = math.sqrt(sum(((rand_dic.get(d,0)/num_centers) - (result_dis_dic.get(d,0)/num_centers))**2 for d in set(rand_dic) | set(result_dis_dic)))

        return dis
    def results2(self):
        self.update_dis_from_center()
        start_time = time.time()

        dic_alpha = dict([(k, self.dic_dis_to_close_center[k] / self.dic_id_NR[k]) for k, v in self.dic_id_NR.items()])
        max_key = max(dic_alpha, key=lambda x: dic_alpha[x])
        print("The point with maximum alpha is:{} in {}".format(max_key, (
        self.df.Longitude[max_key], self.df.Latitude[max_key])))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_key], self.dic_id_NR[max_key]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_key],
                                                                      self.dic_dis_to_close_center[max_key]))
        print("#################")
        max_dis = max(self.dic_dis_to_close_center, key=self.dic_dis_to_close_center.get)
        print("The point with maximum distance from his nearest center is:{} in {}".format(max_dis, (
        self.df.Longitude[max_dis], self.df.Latitude[max_dis])))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_dis], self.dic_id_NR[max_dis]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_dis],
                                                                      self.dic_dis_to_close_center[max_dis]))
        print("#################")
        max_key = max(self.dic_id_NR, key=lambda x: self.dic_id_NR[x])
        print("The point with maximum NR is {} is NR is {}".format(max_key, self.dic_id_NR[max_key]))

        print("#################")
        print(self.dic_center_id_NR.keys())
        dis,dis2 = self.result_distance()
        print("the distance between the result to relative set is {}".format(dis))
        print("the distance between the result to uniform set is {}".format(dis2))
        ##########
        dis3 = self.result_distance2()
        print("the distance between the result to rand set is {}".format(dis3))
        ##########

        # rel = self.relative_calculation()
        # print("rel = {}".format(rel))
        print('time to "results2" end is {}'.format(time.time() - start_time))

    def save_group(self):
        df1 = pd.read_csv("virginia_new.csv", usecols=["Cuisines_Colors"])
        type_list = list(df1.Cuisines_Colors)
        req_dic = list(set(type_list))
        dic_colors = {}
        for c in req_dic:
            dic_colors[c] = list([])
        count = 0
        for i in self.dic_center_ball:
            for c in req_dic:
                dic_colors[c].append(0)
            for j in self.dic_center_ball[i]:
                dic_colors[self.df.Cuisines_Colors[j]][count] += 1
            count += 1
        print("debug")
        self.df4 = pd.read_csv("group.csv")
        for c in dic_colors:
            self.df4[str(c)] = dic_colors[c]
            self.df4.to_csv("group.csv")
            self.df4.head()

        print("debug")

    def plot_point(self):
        x_c = list(self.df.Longitude[self.dic_center_id_NR.keys()])
        y_c = list(self.df.Latitude[self.dic_center_id_NR.keys()])
        c_c = list(self.df.Colors[self.dic_center_id_NR.keys()])
        size_c = [100] * len(x_c)
        x = list(self.df.Longitude)
        y = list(self.df.Latitude)
        c = list(self.df.Colors)
        s_p = [50] * len(x)

        plt.scatter(x, y, c=c, marker='.', s=s_p, cmap='viridis', edgecolors='gray')
        plt.scatter(x_c, y_c, c=c_c, marker='*', s=size_c, edgecolors='black')
        plt.show()


def main():
    # fair = FairKCenter(k)
    # fair.initialization_NR_dic()
    # res = fair.fair_k_2(1.3261226846527259)
    # print(res)
    # fair.results2()
    # fair.plot_point()


        # print(fair.dic_center_id_NR.keys())
        # print(len(fair.dic_center_id_NR.keys()))

    start_time = time.time()
    low =1
    high=2
    while low <= high:
        mid = (low+high)/2
        print("alpha(mid) ={}".format(mid))
        fair = FairKCenter(k)
        fair.initialization_NR_dic()
        res= fair.fair_k_2(mid)
        print(res)
        if res < fair.K:
            high = mid
        elif res == fair.K:
            print("mid={}".format(mid))
            fair.results2()
            print('time to end is {}'.format(time.time() - start_time))

            # print(fair.dic_center_id_NR.keys())
            # print(len(fair.dic_center_id_NR.keys()))
            fair.plot_point()
            break
        else:
            low =mid


if __name__ == '__main__':
    main()
