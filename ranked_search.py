from html_text_parser import list_from_file
import numpy as np
import re
import nltk


docs_3dnews = list_from_file("data/3dnews.ru.list")
docs_overclockers = list_from_file("data/overclockers.ru.list")
docs_opennet = list_from_file("data/opennet.ru.list")
docs = docs_3dnews + docs_overclockers + docs_opennet
docs = np.array(docs)

sno = nltk.stem.SnowballStemmer('russian')

index_invert = {}
for i, doc in enumerate(docs):

    text = " ".join(str(x) for x in doc[0])
    text += " ".join(str(x) for x in doc[1])

    space_symb = re.compile(u'[\s]')
    whitelist_symb = re.compile(u'[^a-zA-Zа-яА-Яё\- ]')
    text = space_symb.sub(' ', text)
    text = whitelist_symb.sub(' ', text)
    
    text = text.lower().replace(u'ё', u'е')
    tokens = text.split(" ")

    for t in tokens:
    	token = sno.stem(t)
    	if token not in index_invert:
    		index_invert[token] = []
    	index_invert[token].append(i)

for k in index_invert:
	index_invert[k] = list(set(index_invert[k]))

for k in index_invert:
	print(f"{k}:{index_invert[k]}")
# print(len(index_invert))