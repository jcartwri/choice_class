## На вход приходят s_vector в виде tensor
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import math
import tensorflow as tf

matr1 = pd.read_excel('/home/kitos/Документы/python_work/vector_model/matr1.xlsx')
matr2 = pd.read_excel('/home/kitos/Документы/python_work/vector_model/matr2.xlsx')
matr3 = pd.read_excel('/home/kitos/Документы/python_work/vector_model/matr3.xlsx')
matr4 = pd.read_excel('/home/kitos/Документы/python_work/vector_model/matr3.xlsx')

vector1 = pd.read_excel('~/Документы/python_work/vector_model/vector1.xlsx')
vector2 = pd.read_excel('~/Документы/python_work/vector_model/vector2.xlsx')
vector3 = pd.read_excel('~/Документы/python_work/vector_model/vector3.xlsx')

s_vector1 = pd.Series(vector1.res)
s_vector2 = pd.Series(vector2.res)
s_vector3 = pd.Series(vector3.res)

s_vector1.name = None
s_vector2.name = None
s_vector3.name = None


s_vector1 = tf.constant(s_vector1.as_matrix(), dtype=tf.float32, shape=[1, len(s_vector1)])
s_vector2 = tf.constant(s_vector2.as_matrix(), dtype=tf.float32, shape=[1, len(s_vector2)])
s_vector3 = tf.constant(s_vector3.as_matrix(), dtype=tf.float32, shape=[1, len(s_vector3)])

class Combination:
    def __init__(self, matrx_input):
        self.len_matr = len(matrx_input)
        for i in range(len(matrx_input)):
            del matrx_input[i]["type"]
            if i == 0:
                self.list_columns = list(matrx_input[0])
            matrx_input[i].columns = range(matrx_input[i].shape[1])
            matrx_input[i] /= matrx_input[i].sum(axis = 1).mean()
            l = len(matrx_input[i])
            matrx_input[i] = tf.constant(matrx_input[i].as_matrix(), dtype = tf.float32, shape=[l,l])
        self.matrx = matrx_input
        
    def ft_min_precision(self, mas):
        for i in range(len(mas)):
            if mas[i] == None:
                continue
            value = self.matrx[i][mas[i]][mas[i]]
            for j in range(len(mas)):
                if i == j or mas[j] == None:
                    continue
                if mas[i] == mas[j]:
                    value *= self.matrx[j][mas[i]][[mas[i]]]
                else:
                    value *= (1 - self.matrx[j][mas[i]][mas[i]])
            if value > self.v_value:
                self.v_value = value
                self.v_index = mas[i]

    def decision_prec(self, mas1, begin, end, x, y, value):
        if mas1[begin] is None or begin == x:
            if begin != end:
                self.decision_prec(mas1, begin + 1, end, x, y, value)
            elif value > self.v_value:
                    self.v_value = value
                    self.v_index = mas1[x][y]
            return ()
        if begin == end:
            for i in range(len(mas1[end])):
                if mas1[end][i] == mas1[x][y]:
                    val = self.matrx[begin][mas1[x][y]][mas1[x][y]]
                else:
                    val = 1.0 - self.matrx[begin][mas1[x][y]][mas1[x][y]]
                if value * val > self.v_value:
                    self.v_value = value * val
                    self.v_index = mas1[x][y]
            return ()
        for j in range(len(mas1[begin])):
            if mas1[begin][j] == mas1[x][y]:
                val = self.matrx[begin][mas1[x][y]][mas1[x][y]]
            else:
                val = 1.0 - self.matrx[begin][mas1[x][y]][mas1[x][y]]
            self.decision_prec(mas1, begin + 1, end, x, y, value * val)

    def ft_full_brute_force(self, mas1, begin, end):
        if mas1[begin] is None:
            if begin == end:
                return()
            else:
                self.ft_full_brute_force(mas1, begin + 1, end)
            return ()
        if begin == end:
            for i in range(len(mas1[end])):
                ind = mas1[begin][i]
                self.decision_prec(mas1, 0, end, begin, i, self.matrx[begin][ind][ind])
            return ()
        for j in range(len(mas1[begin])):
            ind = mas1[begin][j]
            self.decision_prec(mas1, 0, end, begin, j, self.matrx[begin][ind][ind])
        self.ft_full_brute_force(mas1, begin + 1, end)

    def algorithm(self, vector_input):
        mas = []
        std1 = []
        std2 = []
        if sum(x is not None for x in vector_input) == 0 or len(vector_input) != self.len_matr:
            print("Error")
            return -1
        for i in range(len(vector_input)):
            if (vector_input[i] is None):
                mas.append(i)
                std1.append(None)
                std2.append(None)
            else:
                vector_input[i] = vector_input[i] / 100.0
                std = list(tf.reduce_sum(tf.math.subtract(self.matrx[i], tf.linalg.matrix_transpose(vector_input[i]), name=0)**2, axis=1)**0.5)
                s_min = min(std)
                m = [i for i in range(len(std)) if (-0.05 < std[i] - s_min < 0.05)]
                std1.append(m)
                std2.append(std.index(s_min))
        self.v_index = -1
        self.v_value = 0
        if sum(x is not None for x in std2) < 6:
            self.ft_full_brute_force(std1, 0, std1.index(std1[-1]))
        else:
            self.ft_min_precision(std2)
        return 1

C = Combination([matr1, matr2, matr3])
if C.algorithm([s_vector1, s_vector2, s_vector3]) != -1:
    print(C.list_columns[C.v_index])
print(len(C.list_columns))
