
import io
from bs4 import BeautifulSoup
import os

path = os.listdir('C:/Users/jkk_k/Documents/Mestrado/NLP/Tarefa 3/CHAVEFolha/folha95')
print(path)

txt = open('dados_95.txt','a')
txt.write

for item in path:
	f = open(f'C:/Users/jkk_k/Documents/Mestrado/NLP/Tarefa 3/CHAVEFolha/folha95/{item}',"r",encoding="cp1252",errors='ignore')
	
	print(item)
	fmais = f.read()
	#print(fmais)
	soup = BeautifulSoup(fmais,"html.parser")
	tag = soup.findAll('doc')
#print(tag)
	for ab in soup.findAll('doc'):
		txt.write(ab.findChild('text').text)
#print(opa.get_text())
#print(f.read())
txt.close()