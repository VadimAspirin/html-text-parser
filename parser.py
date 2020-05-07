import requests
from bs4 import BeautifulSoup


def parser(url, attribute_name, attribute_value, first_page, last_page, quiet=True):
	for i in range(first_page, last_page+1):

		headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
		response = requests.get(f'{url}{i}', headers=headers)
		soup = BeautifulSoup(response.text, 'html.parser')
		div = soup.find('div', {attribute_name : attribute_value})


		if div is None:
			print(f"{i}: FAIL")
			continue

		item = [text for text in div.stripped_strings]

		if not quiet:
			print(item)
		print(f"{i}: OK")


# 4 1010395
# parser("https://3dnews.ru/", "class" "js-mediator-article", 120, 130, False)




parser("https://overclockers.ru/softnews/show/", "itemprop", "articleBody", 103198, 103198, False)


# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
# response = requests.get("https://overclockers.ru/softnews/show/103198", headers=headers)