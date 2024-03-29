from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt



class MaxSum:
    def __init__(self,req_dic,file1,file2):
        self.req_dic = req_dic
        self.df1  = pd.read_csv(file1)

        self.df2  = pd.read_csv(file2)

        self.list_colors =[]
        self.original_matrix = np.array(self.df2)
        self.Matrix =list([])
        for i in self.req_dic:
             while self.req_dic[i]!=0:
                      self.Matrix.append(list(self.df2[i]))
                      self.req_dic[i]-=1
                      self.list_colors.append(i)
        self.Matrix =np.array((self.Matrix)).transpose()
        self.Matrix_org = self.Matrix.copy()
        self.result_Hungarian =[]
        self.result_org =[]


    def convert(self):
        max_val = self.Matrix.max()
        #self.Matrix[self.Matrix==0]=-max_val
        self.Matrix = [[max_val - e for e in row] for row in self.Matrix]
        self.Matrix=[[100 if e == 1 else e for e in row] for row in self.Matrix]

    def play_hungarian_algo(self):
        self.Matrix = np.array(self.Matrix)
        row_ind, col_ind = linear_sum_assignment(self.Matrix)
        list_color = [self.list_colors[x] for x in col_ind]
        list_id = [self.df2.ID[x] for x in row_ind]
        dic_plot = dict(zip(list_id,list_color))
        print("indexes of centers = {}".format(dic_plot))
        #print("Hungarian_results = {}".format(self.Matrix_org[row_ind, col_ind].sum()))
        list_res =[i for i in self.Matrix_org[row_ind, col_ind]]
        self.result_Hungarian = list(list_res)
        #print("list_res ={}".format(list_res))
        return list_res,dic_plot



    def plot_point(self):

        labels = [i for i in range(len(self.result_org))]
        men_means = [20, 34, 30, 35, 27]
        women_means = [25, 32, 34, 20, 25]
        result_max_min=[0.130879346, 0.104278075, 0.075425791, 0.078729282, 0.073969509, 0.048368954, 0.061776062,
                       0.057906459, 0.145214521, 0.116700201, 0.086486486, 0.114851485, 0.092592593, 0.100293064,
                       0.079365079, 0.117117117, 0.107344633, 0.105084746, 0.101863354]
        x = np.arange(len(self.result_org))  # the label locations
        width = 0.3  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x + 0.00, self.result_org, width, label='source', color ='r')
        rects2 = ax.bar(x + 0.3, self.result_Hungarian, width, label='Hungarian',color ='b')
        rects2 = ax.bar(x + 0.6 , result_max_min, width, label='MaxMin', color='g')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_title('similarity')
        ax.set_xticks(x, labels)
        ax.legend()
        fig.tight_layout()
        plt.show()

