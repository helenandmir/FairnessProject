from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
import time


class MaxSum:
    def __init__(self,req_dic):
        self.req_dic = req_dic
        self.df_group  = pd.read_csv("../DataTemp/group.csv")
        self.original_matrix = np.array( self.df_group)
        self.Matrix =list([])
        for i in self.req_dic:
             while self.req_dic[i]!=0:
                      self.Matrix.append(list(self.df_group[i]))
                      self.req_dic[i]-=1
        self.Matrix =np.array((self.Matrix)).transpose()
        self.row_zero ={}
        self.col_zero ={}
        self.num_row = len(self.Matrix)
        self.num_col = len(self.Matrix[0])
        self.Matrix_org = self.Matrix.copy()
        print(self.Matrix_org)

    def initialization_matrix(self):

        self.Matrix = self.Matrix.reshape(-1)
        sort_arr = np.sort(self.Matrix)
        count = 0
        while (sort_arr.size != 0):
            max_num = max(sort_arr)
            index = np.where(self.Matrix == max_num)
            self.Matrix[index] = count
            count += 1
            sort_arr = np.delete(sort_arr, np.where(sort_arr == max_num))
        self.Matrix = self.Matrix.reshape(self.num_row, self.num_col)

    def convert(self):
        max_val = self.Matrix.max()
        self.Matrix= [[max_val-e for e in row] for row in self.Matrix]
    def play_hungarian_algo(self):
        self.Matrix = np.array(self.Matrix)
        row_ind, col_ind = linear_sum_assignment(self.Matrix)
        print("indexes of centers = {}".format(col_ind))
        print("MaxSum = {}".format(self.Matrix_org[row_ind, col_ind].sum()))

def main() :
    #req_dic={"green":0,"orange":1,"blue":2,"red":0}
    req_dic={'cornflowerblue': 0, 'darkgoldenrod': 0, 'blueviolet': 0, 'azure': 0, 'burlywood': 0, 'cyan': 0, 'cornsilk': 0,
     'darkred': 0, 'darkslateblue': 0, 'darksalmon': 0, 'darkcyan': 0, 'coral': 0, 'brown': 0, 'darkturquoise': 0,
     'deeppink': 0, 'chartreuse': 0, 'blue': 0, 'darkgray': 0, 'darkmagenta': 0, 'darkseagreen': 0, 'darkgrey': 0,
     'blanchedalmond': 0, 'aquamarine': 0, 'darkkhaki': 0, 'deepskyblue': 0, 'beige': 0, 'darkorange': 0, 'aqua': 40,
     'aliceblue': 10, 'darkgreen': 0, 'crimson': 0, 'dimgray': 0, 'bisque': 0, 'darkblue': 0, 'black': 12,
     'darkorchid': 0, 'antiquewhite': 0, 'darkslategray': 0, 'chocolate': 0, 'darkviolet': 0, 'cadetblue': 0,
     'darkslategrey': 0, 'darkolivegreen': 0}
    H = MaxSum(req_dic)
    #H.initialization_matrix()
    H.convert()
    H.play_hungarian_algo()
    #H.get_result()
if __name__ == '__main__':
    start_time = time.time()
    main()
    print('time to end is {}'.format(time.time() - start_time))

