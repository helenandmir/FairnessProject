from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


class MaxMin:
    def __init__(self,req_dic,dic_center_and_colors,file1,file2):
        self.req_dic = req_dic
        self.df2 = pd.read_csv(file2)
        self.df1 = pd.read_csv(file1)
        self.list_center =list(dic_center_and_colors.keys())
        self.list_colors =[]
        self.original_matrix = np.array(self.df2)
        self.Matrix =list([])
        for i in self.req_dic:
             while self.req_dic[i]!=0:
                      self.Matrix.append(list(self.df2[i]))
                      self.req_dic[i]-=1
                      self.list_colors.append(i)
        self.Matrix =np.array((self.Matrix)).transpose()
        self.Matrix2 = self.Matrix.copy()
        self.num_row = len(self.Matrix)
        self.num_col = len(self.Matrix)
        self.row_zero = {}
        self.col_zero = {}

    def sort_indexs_matrix(self):
        # sort the matrix cells
        self.Matrix = self.Matrix.reshape(-1)
        self.Matrix[self.Matrix == 0] = -1
        sort_arr = np.sort(self.Matrix)
        count = 0
        while (sort_arr.size != 0):
            max_num = max(sort_arr)
            index = np.where(self.Matrix == max_num)
            self.Matrix[index] = count
            count += 1
            sort_arr = np.delete(sort_arr, np.where(sort_arr == max_num))
        self.Matrix = self.Matrix.reshape(self.num_row, self.num_col)

    def main_algo(self):
        self.updste_row_zero()
        while (self.end_check() == False):
            self.Matrix = self.Matrix.reshape(-1)
            min_val = min(self.Matrix[np.nonzero(self.Matrix)])
            self.Matrix = np.array([i - min_val if i > 0 else i for i in self.Matrix])
            self.updste_row_zero()
            self.updste_col_zero()


    def updste_col_zero(self):
        self.Matrix = self.Matrix.reshape(self.num_row, self.num_col)
        for i in range(self.num_col):
           if np.count_nonzero(self.Matrix[:,i] == 0) != 0 :
              self.col_zero[i] = np.count_nonzero(self.Matrix[:,i] == 0)

    def updste_row_zero(self):
        self.Matrix = self.Matrix.reshape(self.num_row, self.num_col)
        for i in range(self.num_row):
           if np.count_nonzero(self.Matrix[i, :] == 0) != 0 :
              self.row_zero[i] = np.count_nonzero(self.Matrix[i, :] == 0)


    def end_check(self):
        # check if there is "zero's cover"
        if len(self.row_zero) == len(self.Matrix) and len(self.col_zero) == len(self.Matrix):
            if (self.coverage_test()==True):
                return True
        return False

    def get_max_row(self,arr):
        max_r = -1
        max_ind = -1
        for i in range(0,len(arr[:,0])):
            num_z = len([j for j in arr[i,:] if j == 0])
            if num_z > max_r:
                max_r = num_z
                max_ind = i
        if max_r > 0:
           return (max_ind,max_r)
        else:
            return (-1,-1)

    def get_max_col(self, arr):
        max_c = -1
        max_ind = -1
        for i in range(0, len(arr[0,:])):
            num_z = len([j for j in arr[:,i] if j == 0])
            if num_z > max_c:
                max_c = num_z
                max_ind = i
        if max_c > 0:
            return (max_ind,max_c)
        else:
            return (-1,-1)

    def coverage_test(self):
        matrix_copy = self.Matrix.copy()
        count =0
        while(len(matrix_copy)!=0):
              count+=1
              max_r,num_max_r = self.get_max_row(matrix_copy)
              max_c,num_max_c = self.get_max_col(matrix_copy)
              if max_r ==-1 and max_c==-1:
                  return False

              if num_max_r > num_max_c:
                  matrix_copy = np.delete(matrix_copy, max_r, 0)
              else:
                  if len(matrix_copy[0,:]) ==1:
                      matrix_copy =[]
                  else:
                      matrix_copy = np.delete(matrix_copy, max_c, 1)
        if count == self.num_col:
            return True
        return False

    def get_result(self):
        dic_result = {}
        list_col = list([])
        while (len(self.row_zero) != 0):
            # row with the fewest zeros
            min_row = min(self.row_zero, key=self.row_zero.get)
            col = [i for i in range(self.num_col) if (self.Matrix[min_row, i] == 0 and i not in list_col)]
            min_col = 1000
            c = -1
            # column with the fewest zeros (corresponding to the zeros in the row)
            for i in col:
                if self.col_zero[i] < min_col:
                    min_col = self.col_zero[i]
                    c = i
            dic_result[min_row] = c
            list_col.append(c)
            self.row_zero.pop(min_row)
        #print(dic_result)
        return dic_result
    def get_min_source(self,dic_center_and_colors):
        list_colors = list(self.req_dic.keys())
        list_sours_res =[]
        list_centers =list(dic_center_and_colors.keys())
        for i in dic_center_and_colors.keys():
            list_sours_res.append(self.original_matrix[list_centers.index(i),list_colors.index(dic_center_and_colors[i])+1])
        # print("list similarity of source is {}".format(list_sours_res))
        # print("Min similarity in source is {}".format(min(list_sours_res)))
        return list_sours_res
    def get_result2(self, dic_result):
        min_r = 2
        result_list = [self.Matrix2[i, dic_result[i]] for i in dic_result.keys()]

        dic_center_color={}
        for i in dic_result.keys():
            dic_center_color[self.list_center[i]]=self.list_colors[dic_result[i]]
        for i in dic_result.keys():
            if self.Matrix2[i, dic_result[i]] < min_r:
                min_r = self.Matrix2[i, dic_result[i]]
        # print("->>>>>>***<<<<<<-")
        # print("MaxMin is {}".format(min_r))
        # print("list similarity  is {}".format(result_list))
        # print("balls and colors match :{}".format(dic_center_color))
        return result_list
