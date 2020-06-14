from html_text_parser import list_from_file
import numpy as np
import re
import Stemmer


def str_to_norm_words(in_str):
    stemmer = Stemmer.Stemmer('russian')

    space_symb = re.compile(u'[\s]')
    whitelist_symb = re.compile(u'[^a-zA-Zа-яА-Яё\- ]')
    text = space_symb.sub(' ', in_str)
    text = whitelist_symb.sub(' ', text)
    
    text = text.lower().replace(u'ё', u'е')
    tokens = text.split(" ")

    return stemmer.stemWords(tokens)


def docs_normolize(docs):
    docs_norm = []
    for doc in docs:

        text = " ".join(str(x) for x in doc[0])
        text += " ".join(str(x) for x in doc[1])

        words_norm = str_to_norm_words(text)
        docs_norm.append(words_norm)

    return docs_norm


def docs_word_count(docs_norm, tokens, index_invert):
    result = np.zeros((len(docs_norm), len(tokens)))
    for token_i, token in enumerate(tokens):
        for doc_i in index_invert[token]:
            result[doc_i][token_i] = docs_norm[doc_i].count(token)
    return result


def tf(docs_norm, word_count_per_doc):
    doc_count = len(docs_norm)
    result = np.zeros(word_count_per_doc.shape)
    for i in range(doc_count):
        d = word_count_per_doc[i].astype(float)
        idx = d != 0
        result[i][idx] = d[idx] / len(docs_norm[i])
    return result


def idf(index_invert, tokens):
    return np.log(len(docs_norm) / np.array([len(index_invert[token]) for token in tokens if token in index_invert]))


def docs_to_index_invert(docs_norm):
    index_invert = {}
    for i, doc in enumerate(docs_norm):

        for word in doc:
            if word not in index_invert:
                index_invert[word] = []
            index_invert[word].append(i)

    return {i: list(set(index_invert[i])) for i in index_invert}


def boolean_retrieval(request_str, index_invert, boolean_operation='or'):
    assert boolean_operation in ['or', 'and']
    words = str_to_norm_words(request_str)
    doc_ids = []
    for word in words:
        if word not in index_invert:
            continue
        doc_ids.append(index_invert[word])
    if boolean_operation == 'or':
        results = [j for i in doc_ids for j in i]
        results = list(set(results))
    if boolean_operation == 'and':
        results = [j for i in doc_ids for j in i]
        results_unique = list(set(results))
        results = [i for i in results_unique if results.count(i) == len(doc_ids)]
    return results


def vsm_ranging(request_str, docs_ids, tokens, word_count_per_doc):
    words = str_to_norm_words(request_str)
    words = [word for word in words if word in tokens]
    word_id_request = np.unique(np.array([np.where(tokens == word) for word in words]).reshape(-1))

    word_count_docs = word_count_per_doc[docs_ids][:,word_id_request]
    words_count_request = np.array([words.count(tokens[i]) for i in word_id_request])
    
    d = (word_count_docs * words_count_request).sum(1)
    norm = (np.sqrt((word_count_docs * word_count_docs).sum(1)) * np.sqrt((words_count_request * words_count_request).sum()))
    cos_metric = d / norm

    sort_buf = np.array([docs_ids, cos_metric]).T
    sort_buf = sort_buf[sort_buf[:,1].argsort()]

    return np.flip(sort_buf[:,0]).astype(int), np.flip(sort_buf[:,1])


def bm25_ranging(request_str, docs_ids, tokens, word_count_per_doc, docs_tf, docs_idf, k1=2.0, b=0.75):
    words = str_to_norm_words(request_str)
    words = [word for word in words if word in tokens]
    word_id_request = np.unique(np.array([np.where(tokens == word) for word in words]).reshape(-1))

    idf_words = docs_idf[word_id_request]
    tf_words = docs_tf[docs_ids][:,word_id_request]
    docs_len_target = word_count_per_doc[docs_ids][:,word_id_request].sum(1).reshape(len(docs_ids), 1)
    docs_len_mean = docs_len_target.mean()
    score = ((tf_words * (k1 + 1) * idf_words) / (tf_words + k1 * (1 - b + b * (docs_len_target / docs_len_mean)))).sum(1)

    sort_buf = np.array([docs_ids, score]).T
    sort_buf = sort_buf[sort_buf[:,1].argsort()]

    return np.flip(sort_buf[:,0]).astype(int), np.flip(sort_buf[:,1])


def map_relevant_metrics(target, k):
    assert vsm_relevant.dtype == 'bool'
    N = len(target)
    mean_ap = 0
    for i in range(N):
        counter = 0
        ap = 0
        for j in range(0, k):
            if target[i][j]:
                counter += 1
                ap += counter / (j + 1)
        mean_ap += ap / k
    return mean_ap / N


def ndcg_relevant_metrics(target, k):
    assert vsm_relevant.dtype == 'bool'
    N = len(target)
    mean_ndcg = 0
    for i in range(N):
        dcg = 0
        idcg = 0
        for j in range(0, k):
            dcg += target[i][j] / np.log2((j + 1) + 1)
            idcg += 1 / np.log2((j + 1) + 1)
        dcg /= k
        idcg /= k
        mean_ndcg += dcg/idcg
    return mean_ndcg / N



if __name__== "__main__":
    
    docs_3dnews = list_from_file("data/3dnews.ru.list")
    docs_overclockers = list_from_file("data/overclockers.ru.list")
    docs_opennet = list_from_file("data/opennet.ru.list")
    docs = docs_3dnews + docs_overclockers + docs_opennet

    docs_norm = docs_normolize(docs)
    index_invert = docs_to_index_invert(docs_norm)
    tokens = np.array(list(index_invert.keys()))
    word_count_per_doc = docs_word_count(docs_norm, tokens, index_invert)

    docs_tf = tf(docs_norm, word_count_per_doc)
    docs_idf = idf(index_invert, tokens)
    
    s = [
         u'Новые умные часы', 
         u'Сборка компьютера из комплектующих', 
         u'Сравнение видеокарт',
         u'Какую файловую систему выбрать',
         u'Смартфоны samsung',
        ]

    max_result_count = 30
    for request in s:
        print("Request:", request)

        docs_ids = boolean_retrieval(request, index_invert, boolean_operation='and')

        print("Ranging: VSM")
        res, score = vsm_ranging(request, docs_ids, tokens, word_count_per_doc)
        if len(res) > max_result_count:
            res = res[:max_result_count]
            score = score[:max_result_count]
        print("Score:", list(score))
        for i in range(len(res)):
            print(i, docs[res[i]])

        print()

        print("Ranging: BM25")
        res, score = bm25_ranging(request, docs_ids, tokens, word_count_per_doc, docs_tf, docs_idf)
        if len(res) > max_result_count:
            res = res[:max_result_count]
            score = score[:max_result_count]
        print("Score:", list(score))
        for i in range(len(res)):
            print(i, docs[res[i]])

        print()


    vsm_relevant = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
                             [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]]).astype(bool)
    bm25_relevant = np.array([[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                              [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]]).astype(bool)
    
    map1 = map_relevant_metrics(vsm_relevant, 1)
    map5 = map_relevant_metrics(vsm_relevant, 5)
    map10 = map_relevant_metrics(vsm_relevant, 10)
    map15 = map_relevant_metrics(vsm_relevant, 15)
    print("VSM:")
    print(f"map@1: {map1}\nmap@5: {map5}\nmap@10: {map10}\nmap@15: {map15}")

    print()

    map1 = map_relevant_metrics(bm25_relevant, 1)
    map5 = map_relevant_metrics(bm25_relevant, 5)
    map10 = map_relevant_metrics(bm25_relevant, 10)
    map15 = map_relevant_metrics(bm25_relevant, 15)
    print("BM25:")
    print(f"map@1: {map1}\nmap@5: {map5}\nmap@10: {map10}\nmap@15: {map15}")

    print()

    ndcg1 = ndcg_relevant_metrics(vsm_relevant, 1)
    ndcg5 = ndcg_relevant_metrics(vsm_relevant, 5)
    ndcg10 = ndcg_relevant_metrics(vsm_relevant, 10)
    ndcg15 = ndcg_relevant_metrics(vsm_relevant, 15)
    print("VSM:")
    print(f"ndcg@1: {ndcg1}\nndcg@5: {ndcg5}\nndcg@10: {ndcg10}\nndcg@15: {ndcg15}")

    print()

    ndcg1 = ndcg_relevant_metrics(bm25_relevant, 1)
    ndcg5 = ndcg_relevant_metrics(bm25_relevant, 5)
    ndcg10 = ndcg_relevant_metrics(bm25_relevant, 10)
    ndcg15 = ndcg_relevant_metrics(bm25_relevant, 15)
    print("BM25:")
    print(f"ndcg@1: {ndcg1}\nndcg@5: {ndcg5}\nndcg@10: {ndcg10}\nndcg@15: {ndcg15}")