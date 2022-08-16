import numpy as np
import time
import pandas as pd

class Max_Min:
    # global
    def __init__(self,req_dic):
        self.req_dic = req_dic
        self.df_group = pd.read_csv("group.csv")
        self.original_matrix = np.array(self.df_group[req_dic.keys()])
        self.Matrix =list([])
        self.Matrix2 =list([])
        self.row_zero ={}
        self.col_zero ={}
        self.num_row =0
        self.num_col = 0

    def matrix_adaptation_requirements(self):
        self.Matrix = list([])
        for i in self.req_dic:
            while self.req_dic[i] != 0:
                self.Matrix.append(list(self.df_group[i]))
                self.req_dic[i] -= 1
        self.Matrix = np.array((self.Matrix)).transpose()
        self.Matrix= self.Matrix/ self.Matrix.sum(axis=1)[:, None]
    def sort_indexs_matrix(self):
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
    def initialization(self):
        self.matrix_adaptation_requirements()
        self.num_row = len(self.Matrix)
        self.num_col = len(self.Matrix[0])
        self.Matrix2 = self.Matrix.copy()
        self.sort_indexs_matrix()

        print(self.Matrix)

    def main_algo(self):
        self.updste_row_zero()
        while(self.end_check() == False):
            self.Matrix = self.Matrix.reshape(-1)
            min_val = min(self.Matrix[np.nonzero(self.Matrix)])
            self.Matrix =  np.array([i-min_val if i >0 else i for i in self.Matrix])
            self.updste_row_zero()
            self.updste_col_zero()

            print(self.Matrix)


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
        dic_result ={}
        list_col = list([])
        while(len(self.row_zero)!=0):
            min_row = min(self.row_zero, key=self.row_zero.get)
            col = [i for i in range(self.num_col) if (self.Matrix[min_row, i] == 0 and i not in list_col)]
            min_col = 1000
            c = -1
            for i in col:
                if self.col_zero[i] < min_col:
                    min_col = self.col_zero[i]
                    c=i
            dic_result[min_row] = c
            list_col.append(c)
            self.row_zero.pop(min_row)
        print(dic_result)
        print(self.get_result2(dic_result))
        print(self.Matrix2)

    def get_result2(self,dic_result):
        min_r =2
        for i in dic_result.keys():
            if self.Matrix2[i,dic_result[i]] < min_r:
                min_r = self.Matrix2[i,dic_result[i]]
        return min_r


def main() :
    #req_dic = {"green": 1, "orange": 0, "blue": 2, "red": 1}
    #req_dic = {"blue":1,"red":1,"pink":1,"yellow":1,"green":1,"orange":1}
    #req_dic = {"blue":1, "red":1, "yellow":1, "orange":1}
    # req_dic={'cornflowerblue': 0, 'darkgoldenrod': 1, 'blueviolet': 0, 'azure': 0, 'burlywood': 0, 'cyan': 0, 'cornsilk': 10,
    #  'darkred': 0, 'darkslateblue': 1, 'darksalmon': 0, 'darkcyan': 0, 'coral': 0, 'brown': 0, 'darkturquoise': 0,
    #  'deeppink': 0, 'chartreuse': 0, 'blue': 10, 'darkgray': 5, 'darkmagenta': 0, 'darkseagreen': 5, 'darkgrey': 0,
    #  'blanchedalmond': 0, 'aquamarine': 5, 'darkkhaki': 0, 'deepskyblue': 0, 'beige': 0, 'darkorange': 0, 'aqua': 10,
    #  'aliceblue': 0, 'darkgreen': 5, 'crimson': 0, 'dimgray': 0, 'bisque': 0, 'darkblue': 0, 'black': 10,
    #  'darkorchid': 0, 'antiquewhite': 0, 'darkslategray': 0, 'chocolate': 0, 'darkviolet': 0, 'cadetblue': 0,
    #  'darkslategrey': 0, 'darkolivegreen': 0}
    req_dic = {'cornflowerblue': 0, 'darkgoldenrod': 0, 'blueviolet': 0, 'azure': 0, 'burlywood': 0, 'cyan': 0,
               'cornsilk': 0,
               'darkred': 0, 'darkslateblue': 0, 'darksalmon': 0, 'darkcyan': 0, 'coral': 0, 'brown': 0,
               'darkturquoise': 0,
               'deeppink': 0, 'chartreuse': 0, 'blue': 0, 'darkgray': 0, 'darkmagenta': 0, 'darkseagreen': 0,
               'darkgrey': 0,
               'blanchedalmond': 0, 'aquamarine': 0, 'darkkhaki': 0, 'deepskyblue': 0, 'beige': 0, 'darkorange': 0,
               'aqua': 40,
               'aliceblue': 10, 'darkgreen': 0, 'crimson': 0, 'dimgray': 0, 'bisque': 0, 'darkblue': 0, 'black': 12,
               'darkorchid': 0, 'antiquewhite': 0, 'darkslategray': 0, 'chocolate': 0, 'darkviolet': 0, 'cadetblue': 0,
               'darkslategrey': 0, 'darkolivegreen': 0}
    # for i in req_dic.keys():
    #     req_dic[i] = 1
    # req_dic['aqua'] =11
    # req_dic['blanchedalmond'] = 10
    H = Max_Min(req_dic)
    H.initialization()
    H.main_algo()
    H.get_result()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('time to end is {}'.format(time.time() - start_time))