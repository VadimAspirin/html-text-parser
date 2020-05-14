from html_text_parser import list_from_file
import numpy as np
import re
import binascii



def shingle(data):

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
        shingleLen = 10
        out = []
        for i in range(len(source)-(shingleLen-1)):
            out.append(binascii.crc32(' '.join( [x for x in source[i:i+shingleLen]] ).encode('utf-8')))

        return out


    def compaire(source1, source2):
        same = 0
        for i in range(len(source1)):
            if source1[i] in source2:
                same = same + 1

        return same*2/float(len(source1) + len(source2))*100


    data_canonize = [canonize(d) for d in data]
    data_shingle = [genshingle(d) for d in data_canonize]

    for i in range(len(data_shingle)):
        maximum = 0.0
        for j in range(i+1, len(data_shingle)):
            score = compaire(data_shingle[i], data_shingle[j])
            if score > maximum:
                maximum = score

        print(f"{i}:", maximum)



if __name__== "__main__":
    data_3dnews = list_from_file("data/3dnews.ru.list")
    data_overclockers = list_from_file("data/overclockers.ru.list")
    data_opennet = list_from_file("data/opennet.ru.list")
    data = data_3dnews + data_overclockers + data_opennet
    data = np.array(data)
    data = data[:,1]
    data = [u' '.join(d) for d in data]

    shingle(data)