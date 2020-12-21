
#from sklearn.model_selection import train_test_split
import io
import re, string
import numpy as np
import nltk
import pandas as pd
import random
import pprint, time

f = np.genfromtxt('C:/Users/jkk_k/Documents/Mestrado/NLP/Tarefa 5/texte-tuple_1995_2.txt', delimiter = ',', dtype = str)
g = np.genfromtxt('C:/Users/jkk_k/Documents/Mestrado/NLP/Tarefa 5/texte-tuple_1994_2.txt', delimiter = ',', dtype = str)

data_train = []
for line in f:
	data_train.append((line[0], line[1]))

data_test = []
for lines in g:
	data_test.append((lines[0], lines[1]))

#data_train,data_test =train_test_split(data_train,train_size=0.80,test_size=0.20)

#print(data_train)
#print(data_test)

#classe das palavras
classes_ = {classe for palavra,classe in data_train}
print(classes_)
#total palavras
vocab = {palavra for palavra,classe in data_train}

#Emission Probability
def EP(palavra, classe, train_tuple = data_train):
    classes_list = []
    #conta numero de vezes que a classe ocorreu no dataset de treinamento
    for par in train_tuple:
    	if par[1]==classe:
    		classes_list.append(par)
    count_classe = len(classes_list)
    #conta numero de vezes que a palavra ocorreu com a classe dada
    palavra_classe = []
    for par in classes_list:
    	if par[0]==palavra:
    		palavra_classe.append(par[0])
    count_palavra_classe = len(palavra_classe) 
     
    return (count_palavra_classe, count_classe)

#Transition Probability
def TP(t2, t1, train_tuple = data_train):
    classes_ = []
    for par in train_tuple:
    	classes_.append(par[1])
    count_t1 = []
    for t in classes_:
    	if t==t1:
    		count_t1.append(t)
    count_t1 = len(count_t1)
    count_t2_t1 = 0
    for index in range(len(classes_)-1):
        if classes_[index]==t1 and classes_[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

#probabilidade de classe j acontecer dps de i
tags_matrix = np.zeros((len(classes_), len(classes_)), dtype='float32')
for i, t1 in enumerate(list(classes_)):
    for j, t2 in enumerate(list(classes_)): 
        tags_matrix[i, j] = TP(t2, t1)[0]/TP(t2, t1)[1] 
classes_df = pd.DataFrame(tags_matrix, columns = list(classes_), index=list(classes_)) #melhor leitura

def Viterbi(palavras_, train_tuple = data_train):
    estado = []
    T = list(set([par[1] for par in train_tuple]))
     
    for key, palavra in enumerate(palavras_):        
        p = [] 
        for classe in T:
            if key == 0:
                T_P = classes_df.loc['PU', classe]
            else:
                T_P = classes_df.loc[estado[-1], classe]                 
            
            E_P = EP(palavras_[key], classe)[0]/EP(palavras_[key], classe)[1]
            estado_prob = E_P * T_P    
            p.append(estado_prob)
        #max     
        pmax = max(p)        
        estado_max = T[p.index(pmax)] 
        estado.append(estado_max)
    return list(zip(palavras_, estado))
 
#cria lista só com as palavras
test_palavras = []
for palavra, classe in data_test:
	test_palavras.append(palavra)

pred = Viterbi(test_palavras)
print(pred)
 
#acuracia
igual = [] 
for i, j in zip(pred, data_test):
	if i == j:
		igual.append(i)
 
acuracia = (len(igual)/len(pred))*100
print('Acuracia: ',acuracia)