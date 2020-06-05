from html_text_parser import list_from_file
import numpy as np
import re


def ngrams_dict_generate(tokens, n_gram=3):
    _n_grams = {}
    for token in tokens:

        if not len(token) > (n_gram-1):
            continue
        for i in range(len(token)-(n_gram-1)):
            _n_gram = ""
            for n in range(n_gram):
                _n_gram += token[i+n]

            if _n_gram not in _n_grams:
                _n_grams[_n_gram] = []
            _n_grams[_n_gram].append(token)

    for k in _n_grams:
        _n_grams[k] = list(set(_n_grams[k]))
        
    return _n_grams


def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1
 
    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
 
    return d[lenstr1-1,lenstr2-1]


def spell_checker_damerau_levenshtein(_dict, word, max_out=None, max_len_diff=3, n_gram=3):
    assert len(word) > (n_gram-1)
    _n_grams = []
    for i in range(len(word)-(n_gram-1)):
        res = ""
        for n in range(n_gram):
            res += word[i+n]
        _n_grams.append(res)

    word_candidates = []
    for i in _n_grams:
        if i not in _dict:
            continue
        word_candidates += _dict[i]
    word_candidates = list(set(word_candidates))

    for i in range(len(word_candidates)):
        if word_candidates[i][:1] != word[:1]:
            word_candidates[i] = ""
        if abs(len(word_candidates[i]) - len(word)) > max_len_diff:
            word_candidates[i] = ""

    word_candidates = np.array([i for i in word_candidates if i])
    word_candidates_score = np.array([damerau_levenshtein_distance(i, word) for i in word_candidates])
    word_candidates_idx = np.arange(len(word_candidates))

    buf = np.array([word_candidates_idx, word_candidates_score]).T
    buf = buf[buf[:,1].argsort()]
    word_candidates_idx_sort = buf[:,0].T
    word_candidates_score_sort = buf[:,1].T
    word_candidates = word_candidates[word_candidates_idx_sort]

    if max_out and len(word_candidates) > max_out:
        word_candidates = word_candidates[:max_out]

    return word_candidates.tolist()



if __name__== "__main__":
    data_3dnews = list_from_file("data/3dnews.ru.list")
    data_overclockers = list_from_file("data/overclockers.ru.list")
    data_opennet = list_from_file("data/opennet.ru.list")
    data = data_3dnews + data_overclockers + data_opennet
    data = np.array(data)

    tokens = []
    for text in data:

        _text = " ".join(str(x) for x in text[0])
        _text += " ".join(str(x) for x in text[1])

        reg = re.compile(u'[^a-zA-Zа-яА-Яё ]')
        _text = reg.sub('', _text)
        _text = _text.lower().replace(u'ё', u'е')
        _tokens = _text.split(" ")

        tokens += _tokens

    tokens = list(set(tokens))
    three_grams_dict = ngrams_dict_generate(tokens, n_gram=3)


    print(spell_checker_damerau_levenshtein(three_grams_dict, u"новй", max_out=10))
    print(spell_checker_damerau_levenshtein(three_grams_dict, u"послетний", max_out=10))
    print(spell_checker_damerau_levenshtein(three_grams_dict, u"поестка", max_out=10))
