import requests
from bs4 import BeautifulSoup
import pickle


def list_to_file(item_list, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(item_list, fp)


def list_from_file(item_list, file_name):
    with open ('outfile', 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


def parser(url, tag, attributes, first_page, last_page, quiet=True):
    results = []
    for i in range(first_page, last_page+1):

        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        response = requests.get(f'{url}{i}', headers=headers)
        text = response.text.encode().decode('utf-8')
        soup = BeautifulSoup(text, 'html.parser')


        for script in soup(["script", "style"]):
            script.decompose()
        div = soup.find(tag, attributes)


        if div is None:
            print(f"{i}: FAIL")
            continue

        item = [text for text in div.stripped_strings if len(text.split(" ")) > 1]

        if not quiet:
            print(item)

        results.append(item)
        print(f"{i}: OK")

    return results




if __name__== "__main__":

    result = parser("https://3dnews.ru/", "div", {"class": "js-mediator-article"}, 4, 1010395, False)
    list_to_file(result, "data/3dnews.ru.list")

    result = parser("https://overclockers.ru/softnews/show/", "div", {"itemprop": "articleBody"}, 15000, 103198, False)
    list_to_file(result, "data/overclockers.ru.list")

    result = parser("https://www.opennet.ru/opennews/art.shtml?num=", "td", {"class": "chtext"}, 1000, 52892, False)
    list_to_file(result, "data/opennet.ru.list")