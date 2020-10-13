import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", None, "display.max_columns", None)

words = re.findall(r'\w+', open('Iracema.txt').read().lower())
histWords = Counter(words).most_common()
print(histWords)
qntWords = len(histWords)
print('Quantidade de Palavras Distintas: ',qntWords)

df = pd.DataFrame(histWords[0:50], columns=['Word', 'Frequencia'])
print(df)
df.plot(kind='bar', x='Word')
plt.show()

#Evitar a, e, o, que, de, para (artigos, conectivos, palavras de mesmo tamanho...)
#Cada vez que aumentar o numero de caracteres do sufixos é colocado uma condição a mais para evitar
def PrefixosMaior1(size, histWords):
	size
	pref = []
	for i in range(len(histWords)):
		if ((((len(histWords[i][0])!=1 and len(histWords[i][0])!=2) and len(histWords[i][0])!=3) and len(histWords[i][0])!=4) and len(histWords[i][0])!=5) :
			char = histWords[i][0][:size]
			pref.append((char,histWords[i][1]))		
	return pref

df_pref = pd.DataFrame(PrefixosMaior1(5,histWords), columns=['Prefixo', 'Frequencia'])
print(df_pref)
soma_dos_iguais = df_pref.groupby('Prefixo').sum()[['Frequencia']]
soma_des = soma_dos_iguais.sort_values('Frequencia', ascending=False)
print(soma_des)

soma_des[:50].plot(kind='bar')
plt.show()
