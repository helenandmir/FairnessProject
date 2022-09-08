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
k=100
class FairKCenter:
    def __init__(self,req_dic,color_list,k):
        self.req_dic = req_dic  # A Dictionary that holds the constraints, the desired number of representatives from each entry
        self.update_req_dic = dict([(k,self.req_dic[k]) for k, v in self.req_dic.items() ])
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
        self.dic_group ={}
        self.dic_group_dis={}
        #self.df = pd.read_csv("zomato.csv", usecols=self.col_list)
        self.df = pd.read_csv(fileA, usecols=self.col_list)
        self.df1 = pd.read_csv(fileB,usecols=self.col_list1)
        self.df2 = pd.read_csv(fileB,usecols=["ID","NR_TYPE_ONE"])


        self.K = k #number of centers
        self.num_color ={}
        for i in color_list:
            self.num_color[i]=0

    def initialization_NR_dic(self,all_point,num_min):
        start_time=time.time()

        non_zero_color=[i for i in self.req_dic if self.req_dic[i]!=0]

        for i in all_point:

            arr = []
            for j in range(1,len(non_zero_color)+1):
                arr.append(self.df1.iat[i,j])
            for l in range(1,num_min):
                min1 = min(arr)
                arr.remove(min1)
            max_min = min(arr)

            self.dic_id_NR[i] = max(self.df2.iat[i,1], max_min)
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
            if len(tuple_color)==0:
                break
            tree_c = KDTree(np.array(list(tuple_color)))
            n = len(list_t)
            dist_c, ind_c = tree_c.query([[self.df.X[p], self.df.Y[p],self.df.Z[p]]], n)
            if n == 1:
                print("@@@@@@")
                break
            for dis, i in zip(dist_c[0], ind_c[0]):
                if dis <= self.dic_id_NR[list_t[i]] + self.dic_id_NR[p]:
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


    def uniform_calculation(self):
        num_center = len(self.dic_center_id_NR)
        uni = len(set(self.df.Colors)) / num_center
        colors_list = list(set(self.df.Colors))
        result_arr = {}
        distribution_arr = {}
        result = 0
        center_color_num = {}
        color_num = {}
        for c in colors_list:
            color_num[c] = len([i for i in self.df.ID if self.df.Colors[i] == c])
            center_color_num[c] = len([i for i in self.dic_center_id_NR.keys() if self.df.Colors[i] == c])
        for c in colors_list:
            distribution_arr[c] = (center_color_num[c] / num_center)
            # result_arr[c] = abs(uni-(center_color_num[c]/num_center))
            result += abs(uni - (center_color_num[c] / num_center))
        print("The number of points of each color in the data:{}".format(color_num))
        print("The number of center of each color in the result:{}".format(center_color_num))
        # print("result_arr list ={}".format(result_arr))
        print("distribution_arr ={}".format(distribution_arr))
        print("uniform={}".format(uni))
        result = result / len(colors_list)
        return result
    def result_distance(self):
        num_centers = len(self.dic_center_id_NR)
        colors_list = list(set(self.df.Colors))
        num_point = len(self.df.Colors)
        relative_dic={}
        result_dis_dic= {}
        for c in colors_list:
            # result_dis_dic[c] = len([i for i in self.dic_center_id_NR.keys() if self.df.Colors[i] ==c])
            #math.floor(num_centers/len(colors_list))
            relative_dic[c] = math.floor((list(self.df.Colors).count(c) / num_point)*num_centers )
            result_dis_dic[c] = len([i for i in self.dic_center_id_NR.keys() if self.df.Colors[i] ==c])
        #calculat distance
        dis = math.sqrt(sum(((relative_dic.get(d,0)/num_centers) - (result_dis_dic.get(d,0)/num_centers))**2 for d in set(relative_dic) | set(result_dis_dic)))
        print("relative_dic: {}".format(relative_dic))
        print("result_dis_dic: {}".format(result_dis_dic))
        return dis

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
        print("The point with maximum NR is {} is NR is {}".format(max_key, self.dic_id_NR[max_key]))
        print("Its alpha is {}, Its radius is {}.".format(dic_alpha[max_key], self.dic_id_NR[max_key]))
        print("Its distance from the nearest center {} is {}.".format(self.dic_close_center[max_key],
                                                                      self.dic_dis_to_close_center[max_key]))

        print("#################")
        print(self.dic_center_id_NR.keys())
        dis = self.result_distance()
        print("the distance between the result to relative set is {}".format(dis))

        print('time to "results2" end is {}'.format(time.time() - start_time))


    def initialization_ball_center2(self):

        tuple_color = tuple(zip(list(self.df.X[self.df.ID]), list(self.df.Y[self.df.ID]), list(self.df.Z[self.df.ID])))
        tree_c = KDTree(np.array(list(tuple_color)))
        for c in self.dic_center_id_NR:
            dist_c, ind_c = tree_c.query(list(self.dic_center_loc[c]),len(list(self.df.ID)), distance_upper_bound=self.dic_center_id_NR[c])#distance_upper_bound=self.dic_center_id_NR[c]+1
            dis = [i for i in dist_c if i != math.inf]
            ind = [self.df.ID[i] for i in ind_c[0:len(dis)] ]
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
                #print("before = {}".format(len(self.dic_center_id_NR)))
                #print(" i ={} new_center[i] ={} ".format(i,new_center[i]))
                self.dic_center_id_NR.pop(i)
                self.dic_center_id_NR[new_center[i]] = self.dic_id_NR[new_center[i]]
                self.dic_center_loc.pop(i)
                self.dic_center_loc[new_center[i]] = self.dic_id_loc[new_center[i]]

                self.dic_center_ball[new_center[i]] = self.dic_center_ball[i]
                self.dic_center_ball.pop(i)
                #print("after = {}".format(len(self.dic_id_NR)))
            else:
                self.dic_center_id_NR.pop(i)
                self.dic_center_loc.pop(i)
                self.dic_center_ball.pop(i)

    def update_requirments(self):
        print("before")
        for i in self.update_req_dic :
            self.update_req_dic[i] = self.update_req_dic[i]  -len([j for j in self.dic_center_id_NR.keys() if self.df.Colors[j]==i])
        print("end")
    def initialization_grups(self):
        # create ball
        list_t = list(self.dic_center_id_NR.keys())
        tuple_color = tuple(zip(list(self.df.X[list_t]), list(self.df.Y[list_t]), list(self.df.Z[list_t])))
        tree_c = KDTree(np.array(list(tuple_color)))
        for i in self.dic_center_id_NR.keys():
            self.dic_group[i] =[]

        for i in self.df.ID:
            dist_c, ind_c = tree_c.query([[self.df.X[i], self.df.Y[i], self.df.Z[i]]], 1)
            self.dic_group[list_t[ind_c[0]]].append(i)
            #self.dic_group_dis[list_t[ind_c[0]]].append(list_t[dist_c[0]])
    def add_center(self):
        new_k=sum(self.req_dic.values())
        temp_dic_id_NR = self.dic_center_id_NR.copy()
        while len(self.dic_center_id_NR)!=new_k:
            center = max(temp_dic_id_NR, key=temp_dic_id_NR.get)
            all_point=dict([(i,distance.euclidean(self.dic_id_loc[i], self.dic_center_loc[center]) )for i in self.dic_group[center] if self.update_req_dic[self.df.Colors[i]]!=0 and i not in self.dic_center_id_NR.keys()])
            new_center = max(all_point, key=all_point.get)
            self.dic_center_id_NR[new_center]=self.dic_id_NR[new_center]
            self.dic_center_loc[new_center]=self.dic_id_loc[new_center]
            color = self.df.Colors[new_center]
            self.update_req_dic[color] -=1
            temp_dic_id_NR.pop(center)
        print("end")



def main() :
    start_time = time.time()


    col_list = ["Colors"]
    df1 = pd.read_csv(fileA, usecols=col_list)
    type_list = list(df1.Colors)
    type_set = list(set(type_list))
    num_type = len(type_set)
    color_list = list(set(df1.Colors))#list(matplotlib.colors.cnames.keys())[0:num_type]
    req_dic = {}
    center_colors = {}
    dic_orig = {}


    # Creating relative requirements dictionary

    for c in color_list:
        req_dic[c] = math.floor((list(df1.Colors).count(c) / len(df1.Colors)) * k)

    # #Creating random requirements dictionary
    # while len(color_list) > 0:
    #     c = random.choice(color_list)
    #     range_num = min(list(df1.Colors).count(c), k_temp)
    #     num_random = random.randint(0, range_num)
    #     req_dic[c] = num_random
    #     k_temp -= num_random
    #     center_colors[c] = 0
    #     color_list.remove(c)

    # #Creating uniform requirements dictionary
    #
    # uni=math.floor(k/len(color_list))
    # print(uni)
    # for c in color_list:
    #     req_dic[c] = uni


    print(sum(req_dic.values()))
    print("requirement dictionary:")
    print(req_dic)
    color_list = list(df1.Colors)#list(matplotlib.colors.cnames.keys())[0:num_type]
    for i in range(1,len(color_list)):
        print("->>i={}<<-".format(i))
        fair = FairKCenter(req_dic,color_list,k)
        fair.initialization_NR_dic(list(fair.df.ID),i)
        fair.fair_k_2(0.01)
        fair.initialization_ball_center2()
        C = MaxMatching.CR(fair.req_dic, fileA, fair.dic_center_ball)
        C.add_nodes1()
        C.add_nodes2()
        C.colors_in_balls()
        C.add_edegs()
        new_center = C.create_graph()
        num = len(fair.dic_center_id_NR)
        print("len new_center = {}".format(len(new_center)))
        fair.replace_center(new_center)

        print("len center after replace ={}".format(len(fair.dic_center_id_NR)))
        if len(fair.dic_center_id_NR)==num:
            print([fair.df.Colors[id] for id in fair.dic_center_id_NR.keys()])
            fair.update_requirments()
            fair.initialization_grups()
            fair.add_center()
            fair.results2()
            print('time to end is {}'.format(time.time() - start_time))
            fair.plot_point()
            break



    #fair.two_fair_k_center(0.01)






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

