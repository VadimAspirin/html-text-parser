from html_text_parser import list_from_file
import numpy as np
import re

data_3dnews = list_from_file("data/3dnews.ru.list")
data_overclockers = list_from_file("data/overclockers.ru.list")
data_opennet = list_from_file("data/opennet.ru.list")
data = data_3dnews + data_overclockers + data_opennet
data = np.array(data)
data = data[:,1]
data = [' '.join(d) for d in data]


def canonize(source):
    stop_symbols = '.,!?:;-\n\r()'

    stop_words = (u'это', u'как', u'так',
                  u'и', u'в', u'над',
                  u'к', u'до', u'не',
                  u'на', u'но', u'за',
                  u'то', u'с', u'ли',
                  u'а', u'во', u'от',
                  u'со', u'для', u'о',
                  u'же', u'ну', u'вы',
                  u'бы', u'что', u'кто',
                  u'он', u'она', u'по')

    white_list_re = u"^[A-Za-zА-Яа-яё]*$"

    return [x for x in [y.strip(stop_symbols) for y in source.lower().split()] if len(x) > 1 and re.match(white_list_re, x) and (x not in stop_words)]

def genshingle(source):
    shingleLen = 10 #длина шингла
    out = []
    for i in range(len(source)-(shingleLen-1)):
    out.append (' '.join( [x for x in source[i:i+shingleLen]] ).encode('utf-8'))

    return out


data = [canonize(d) for d in data]
print(data[100])


# print(re.match(u"^[A-Za-zА-Яа-яё]*$", "правила"))