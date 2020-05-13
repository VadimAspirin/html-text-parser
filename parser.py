import requests
from bs4 import BeautifulSoup
import pickle


def list_to_file(item_list, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(item_list, fp)


def list_from_file(file_name):
    with open (file_name, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


def parser(url, tag, attributes,):

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.Timeout:
        return None

    text = response.text.encode().decode('utf-8')
    soup = BeautifulSoup(text, 'html.parser')

    for script in soup(["script", "style"]):
        script.decompose()
    div = soup.find(tag, attributes)

    if div is None:
        return None

    item = [text for text in div.stripped_strings if len(text.split(" ")) > 1]

    return item


def simple_url_parser(base_url, head_tag, head_attributes, body_tag, body_attributes, first_page_num, last_page_num, quiet=True):

    results = []
    for i in range(first_page_num, last_page_num+1):

        url = f"{base_url}{i}"
        body = parser(url, body_tag, body_attributes)
        head = parser(url, head_tag, head_attributes)

        if body is None or head is None:
            print(f"{i}: FAIL")
            continue

        if not quiet:
            print([head, body])

        results.append([head, body])
        print(f"{i}: OK")

    return results



if __name__== "__main__":

    # result = simple_url_parser(base_url="https://3dnews.ru/",
    #                            head_tag="div",
    #                            head_attributes={"class": "entry-header"},
    #                            body_tag="div",
    #                            body_attributes={"class": "js-mediator-article"},
    #                            first_page_num=990000,
    #                            last_page_num=1010395,
    #                            quiet=False)
    # list_to_file(result, "data/3dnews.ru.list")

    # result = simple_url_parser(base_url="https://overclockers.ru/softnews/show/",
    #                            head_tag="h1",
    #                            head_attributes={"itemprop": "headline"},
    #                            body_tag="div",
    #                            body_attributes={"itemprop": "articleBody"},
    #                            first_page_num=85000,
    #                            last_page_num=103198,
    #                            quiet=False)
    # list_to_file(result, "data/overclockers.ru.list")

    result = simple_url_parser(base_url="https://www.opennet.ru/opennews/art.shtml?num=",
                               head_tag="span",
                               head_attributes={"id": "r_title"},
                               body_tag="td",
                               body_attributes={"class": "chtext"},
                               first_page_num=35000,
                               last_page_num=52892,
                               quiet=False)
    list_to_file(result, "data/opennet.ru.list")