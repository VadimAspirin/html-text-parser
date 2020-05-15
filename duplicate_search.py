from html_text_parser import list_from_file
import numpy as np
import re
import binascii
import os
from multiprocessing import Process, Pool
import time
import itertools



def _compaire(source1, source2):
    # same = 0
    # for i in range(len(source1)):
    #     if source1[i] in source2:
    #         same = same + 1

    s1 = list(set(source1))
    s2 = list(set(source2))
    s = s1 + s2
    unique_count = len(set(s))
    count = len(s)
    same = count - unique_count

    return same*2/float(len(source1) + len(source2))


def _canonize(source):

    stop_symbols = '.,!?:;-\n\r()'
    white_list_re = u"^[A-Za-zА-Яа-яё]*$"
    word_min_len = 1
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

    return [x for x in [y.strip(stop_symbols) for y in source.lower().split()] if len(x) > word_min_len and re.match(white_list_re, x) and (x not in stop_words)]


def _genshingle(source, shingleLen=10, div=49):
    out = []
    for i in range(len(source)-(shingleLen-1)):
        result = binascii.crc32(' '.join( [x for x in source[i:i+shingleLen]] ).encode('utf-8'))
        if not result % div:
            continue
        out.append(result)

    return out


def __shingle_process(params):
    data = params[0]
    i = params[1]

    max_score = 0.0
    near_paper = None
    for j in range(i+1, len(data)):
        score = _compaire(data[i], data[j])
        if score > max_score:
            max_score = score
            near_paper = j

    return (i, near_paper, max_score)


def shingle(data, proc_count=4, quiet=True, max_count_near_papers=5):

    if not quiet:
        print("data_processing ...")
        t_start = time.time()
    data_canonize = [_canonize(d) for d in data]
    data_shingle = [_genshingle(d) for d in data_canonize]
    if not quiet:
        print("data_processing: time: ", time.time() - t_start)

    if not quiet:
        print("duplicate_search ...")
        t_start = time.time()

    _data = [[i, j] for i, j in zip([data_shingle]*len(data_shingle), range(len(data_shingle)))]
    # _data = _data[:500]

    results = []
    with Pool(processes=proc_count) as pool:
        for params in pool.imap_unordered(__shingle_process, _data):
            if params[2] != 0:
                results.append(params)
            if not quiet:
                print(f"{params[0]}: {params[2]}")

    if not quiet:
        print("duplicate_search: time: ", time.time() - t_start)

    results = np.array(results)
    results = results[results[:,2].argsort()]
    item_num = results[:,0][::-1].astype(int).tolist()
    near_item_num = results[:,1][::-1].astype(int).tolist()
    score_per_item = results[:,2][::-1].astype(float).tolist()


    if not quiet:
        it_end = len(item_num) if len(item_num) < max_count_near_papers else max_count_near_papers
        for i in range(it_end):
            print(f"[near_papers: {i}] [score: {score_per_item[i]}]")
            print("=========================================================")
            print(f"[paper: {item_num[i]}]")
            print(data[item_num[i]])
            print("---------------------------------------------------------")
            print(f"[paper: {near_item_num[i]}]")
            print(data[near_item_num[i]])
            print("=========================================================")

    return (item_num, near_item_num, score_per_item)
    



if __name__== "__main__":
    data_3dnews = list_from_file("data/3dnews.ru.list")
    data_overclockers = list_from_file("data/overclockers.ru.list")
    data_opennet = list_from_file("data/opennet.ru.list")
    data = data_3dnews + data_overclockers + data_opennet
    data = np.array(data)
    data = data[:,1]
    data = [u' '.join(d) for d in data]

    shingle(data, proc_count=16, quiet=False, max_count_near_papers=150)