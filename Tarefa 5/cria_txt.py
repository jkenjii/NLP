import re 
import io
from nltk.lm import MLE
import re, string
import numpy as np
from nltk import word_tokenize

f = open('C:/Users/jkk_k/Documents/Mestrado/NLP/Tarefa 3/CHAVEFolha/cg.Folha.1995',"r",encoding="cp1252",errors='ignore')
text = f.readlines(100000)


text_data = []
for line in text:
	#palavra = re.findall('\[(.*)\]', line)
	#classe = re.findall("^.*/([^ ]+).*$','\\1", line)

	tokens = word_tokenize(re.sub(rf"[{string.punctuation}]", "", line))
	
	print(tokens)
	if len(tokens) > 1:
		if tokens[2].isupper() is True:
			print(tokens[1], tokens[2])
			text_data.append((tokens[1], tokens[2]))

		elif tokens[3].isupper() is True:
			print(tokens[1], tokens[3])
			text_data.append((tokens[1], tokens[3]))

		elif tokens[4].isupper() is True:
			print(tokens[1], tokens[4])
			text_data.append((tokens[1], tokens[4]))
	print(line)

print(text_data)


#txt = open('texte-tuple_1994_2.txt','a')
#txt.write
#for item in text_data:
#	print(item)
	#txt.write(f'{item[0]},{item[1]}\n')



#txt.close()