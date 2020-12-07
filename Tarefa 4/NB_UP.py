#http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/ - dataset origem
import numpy as np
import re
import string
import math
import csv
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import nltk
from sklearn.model_selection import KFold

with open('spam.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    X = []
    Y = []
    for row in readCSV:
    	X.append(row[1])
    	if row[0] == 'ham':
    		Y.append(0)
    	else:
    		Y.append(1)

class NB_Model(object):	

	def tokenize(self, text):
		pontuacao = str.maketrans("", "", string.punctuation)
		text = text.translate(pontuacao).lower()
		text = re.sub(r'\d+','',text)
		stop_words = set(nltk.corpus.stopwords.words('english'))
		text = list(set(text.split(' ')) - stop_words)
		stemmer= PorterStemmer()
		for i,word in enumerate(text):
   			text[i] = stemmer.stem(word)		
		return text
 
	def contagem_word(self, words):
		word_counts = {}
		for word in words:
			word_counts[word] = word_counts.get(word, 0.0) + 1.0
		return word_counts

	#https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/210-naive-bayes.pdf
	#uso de log pq estava dando 0 como resultado
	def fit(self, X, Y):
	    self.num_mensagens = {}
	    self.log_Priori = {}
	    self.word_counts = {}
	    self.vocab = set()
	    self.num_mensagens['spam'] = 0
	    self.num_mensagens['ham'] = 0
	    for label in Y:
	    	if label == 1:
	    		self.num_mensagens['spam'] += 1
	    	else:
	    		self.num_mensagens['ham'] += 1
	    n = len(X)	    		
	    self.log_Priori['spam'] = math.log(self.num_mensagens['spam'] / n)	    
	    self.log_Priori['ham'] = math.log(self.num_mensagens['ham'] / n)
	    self.word_counts['spam'] = {}
	    self.word_counts['ham'] = {}	 
	    for x, y in zip(X, Y):
	        c = 'spam' if y == 1 else 'ham'
	        counts = self.contagem_word(self.tokenize(x))	        
	        for word, count in counts.items():
	            if word not in self.vocab:
	                self.vocab.add(word)
	            if word not in self.word_counts[c]:
	                self.word_counts[c][word] = 0.0
	 
	            self.word_counts[c][word] += count
	    #print(self.vocab)
	    

	def predict(self, X):
	    resultado = []
	    for x in X:
	        counts = self.contagem_word(self.tokenize(x))	        
	        spam_score = 0
	        ham_score = 0
	        for word, abrobrinha in counts.items():
	            if word not in self.vocab: continue  
	            log_spam = math.log( (self.word_counts['spam'].get(word, 0.0) + 1) / (self.num_mensagens['spam'] + len(self.vocab)) )
	            log_ham = math.log( (self.word_counts['ham'].get(word, 0.0) + 1) / (self.num_mensagens['ham'] + len(self.vocab)) )
	            spam_score += log_spam
	            ham_score += log_ham	 
	        spam_score += self.log_Priori['spam']
	        ham_score += self.log_Priori['ham']
	 
	        if spam_score > ham_score:
	            resultado.append(1)
	        else:
	            resultado.append(0)

	    return resultado

train_X , test_X , train_y, test_y = train_test_split(X,Y, test_size = 0.25)

NB = NB_Model()
NB.fit(train_X, train_y)
eu = NB.fit(train_X, train_y)
 
pred = NB.predict(test_X)
true = test_y
 
acuracia = 0.0
for i in range(len(pred)):
	if pred[i] == true[i]:
		acuracia += 1
acuracia = acuracia / float(len(pred))
print(acuracia)

#Cross Validation

def cross_val(indices, X, Y, k = 5):
    kf = KFold(n_splits=k, shuffle=False)
    scores = []
    for train_index, test_index in kf.split(X):
    	print(train_index, test_index)  

    	train = []
    	y_train = []
    	test = []
    	y_test = []
    	
    	for item in train_index:    		
    		train.append(X[item])
    		y_train.append(Y[item])

    	for item in test_index:
    		test.append(X[item])
    		y_test.append(Y[item])

    	NB = NB_Model()
    	NB.fit(train, y_train)
    	pred = NB.predict(test)
    	true = y_test
		 
    	acuracia = 0.0
    	for i in range(len(pred)):
    		if pred[i] == true[i]:
    			acuracia += 1
    	acuracia = acuracia / float(len(pred))
    	scores.append(acuracia)

    return scores

indices = list(range(len(X)))
score = cross_val(indices, X , Y, k=10)
print('Acuracias Cross Validation: ', score)
print('Media Acuracia(%):', np.array(score).mean()*100)
print('STD Acuracia: ', np.array(score).std())
