
from nltk import word_tokenize, sent_tokenize 
from nltk.lm.preprocessing import padded_everygram_pipeline
import io
from nltk.lm import MLE
import re, string
import numpy as np


f = io.open('dados.txt',"r",encoding="cp1252",errors='ignore')
text = f.read()

tokenized_text = [list(map(str.lower, word_tokenize(re.sub(rf"[{string.punctuation}]", "", sent)))) #separa o texto em sentença e delas 'tokeniza', deixando minusculas e tirando pontuação
                  for sent in sent_tokenize(text)]

print(tokenized_text[0])

train_data, padded_sents = padded_everygram_pipeline(3, tokenized_text)

model = MLE(3)
model.fit(train_data, padded_sents)

print(model.vocab)
print(model.counts)

print(model.score('cardoso', 'fernando henrique'.split()) )

print(model.score('revela', 'datafolha'.split()) )

def interpol(word,context):
	print(context)
	p = 0.65*(model.score(word,str(context).split()))+0.175*(model.score(word,context[1].split()))+0.175*(model.score(word)) 
	return p

print(interpol('cardoso', 'fernando henrique'.split()))


#for senten in tokenized_text:
#	print(senten)

print(tokenized_text[0][10-1])


f_94 = io.open('dados_94.txt',"r",encoding="cp1252",errors='ignore')
text_94 = f_94.read()

tokenized_text_94 = [list(map(str.lower, word_tokenize(re.sub(rf"[{string.punctuation}]", "", sent)))) 
                  for sent in sent_tokenize(text_94)]

print(tokenized_text_94[0])
p_geral = []
perp = []

for senten in tokenized_text_94:
	i=0
	p_sen = []
	for i in range(len(senten)):
		if i ==0:
			p_sen.append(model.score(senten[i]))
			print(i)
		elif i ==1:
			p_sen.append(model.score(senten[i], senten[i-1].split()))
			print(i)
		else:
			p_sen.append(interpol(senten[i],senten[i-2:i]))
			print(i)
	print(p_sen)
	p_geral.append(np.prod(p_sen, dtype = float))
	if np.prod(p_sen, dtype = float) == 0:
		perp.append(0)
	elif len(senten) == 0:
		perp.append(0)
	else:
		perp.append((1/np.prod(p_sen, dtype = float))**(1/len(senten)))		

#print(p_geral)
print(len(tokenized_text_94))
print(len(p_geral))
#print(perp)
#print(np.mean(perp, dtype = float))
print(model.vocab)
print(model.counts)

np.savetxt("GFG.csv",perp, delimiter =", ", fmt ='% s') 
