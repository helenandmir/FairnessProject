from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

file1 ="../DataSet/Banks.csv"
file2 ="../DataSet/Banks_group_100_3.csv"

class MaxSum:
    def __init__(self,req_dic):
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
        self.Matrix = [[max_val - e for e in row] for row in self.Matrix]

    def play_hungarian_algo(self):
        self.Matrix = np.array(self.Matrix)
        row_ind, col_ind = linear_sum_assignment(self.Matrix)
        list_color = [self.list_colors[x] for x in col_ind]
        list_id = [self.df2.ID[x] for x in row_ind]
        dic_plot = dict(zip(list_id,list_color))
        print("indexes of centers = {}".format(dic_plot))
        print("Hungarian_results = {}".format(self.Matrix_org[row_ind, col_ind].sum()))
        list_res =[i for i in self.Matrix_org[row_ind, col_ind]]
        self.result_Hungarian = list(list_res)
        print("list_res ={}".format(list_res))

    def source_result(self):
        list_center = list(self.df2.ID)
        list_colors = list(pd.read_csv(file2, nrows=0).columns.tolist())
        sum_result =0
        for i in list_center:
            row = list_center.index(i)
            col =list_colors.index(self.df1.Colors[i])
            sum_result =sum_result+self.Matrix_org[row,col]
            self.result_org.append(self.Matrix_org[row,col])
            Max_min_res=[0.021428571, 0.003389831, 0.019033675, 0.0, 0.017610063, 0.017156863, 0.010279001, 0.014888337, 0.014414414, 0.01369863, 0.012658228, 0.012631579, 0.004878049, 0.012173913, 0.009708738, 0.011538462, 0.010344828, 0.001436782, 0.002008032, 0.00968523, 0.004807692, 0.004761905, 0.009208103, 0.003484321, 0.004347826, 0.008474576, 0.004166667, 0.0, 0.007281553, 0.0, 0.0, 0.007029877, 0.0, 0.006756757, 0.004, 0.001298701, 0.006302521, 0.0, 0.0, 0.005399568, 0.00525394, 0.0, 0.005089059, 0.005059022, 0.001677852, 0.004962779, 0.0, 0.0, 0.0, 0.003773585, 0.0, 0.001828154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print("MaxMin result is {}".format(sum(Max_min_res)))
        print("source_result is {}".format(sum_result))

    def plot_point(self):

        labels = [i for i in range(len(self.result_org))]
        men_means = [20, 34, 30, 35, 27]
        women_means = [25, 32, 34, 20, 25]
        result_max_min=[0.021428571, 0.003389831, 0.019033675, 0.0, 0.017610063, 0.017156863, 0.010279001, 0.014888337, 0.014414414, 0.01369863, 0.012658228, 0.012631579, 0.004878049, 0.012173913, 0.009708738, 0.011538462, 0.010344828, 0.001436782, 0.002008032, 0.00968523, 0.004807692, 0.004761905, 0.009208103, 0.003484321, 0.004347826, 0.008474576, 0.004166667, 0.0, 0.007281553, 0.0, 0.0, 0.007029877, 0.0, 0.006756757, 0.004, 0.001298701, 0.006302521, 0.0, 0.0, 0.005399568, 0.00525394, 0.0, 0.005089059, 0.005059022, 0.001677852, 0.004962779, 0.0, 0.0, 0.0, 0.003773585, 0.0, 0.001828154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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

def main() :
    # requairment
    df2 = pd.read_csv(file2)
    df1 =pd.read_csv(file1)
    color_list = pd.read_csv(file2, nrows=0).columns.tolist()
    color_list.remove("ID")
    req_dic ={}
    dic_center_and_colors ={}
    for i in color_list:
        req_dic[i] =0
    for i in df2.ID:
        dic_center_and_colors[i]=df1.Colors[i]
        c = df1.Colors[i]
        req_dic[c] = req_dic[c]+1
    print("req_dic {}".format(req_dic))
    print(dic_center_and_colors)
    H = MaxSum(req_dic)
    H.convert()
    H.play_hungarian_algo()
    H.source_result()
    H.plot_point()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('time to end is {}'.format(time.time() - start_time))
