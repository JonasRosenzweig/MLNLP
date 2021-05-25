# from bs4 import BeautifulSoup
# from fake_useragent import UserAgent
# import requests
#
# ua = UserAgent()
#
# def lovely_soup(url):
#     r = requests.get(url, headers={'User-Agent': ua.chrome})
#     return BeautifulSoup(r.text, 'lxml')
#
# soup = lovely_soup("https://www.reddit.com/r/wallstreetbets")
# home = soup.find(class_="_1OVBBWLtHoSPfGCRaPzpTf _3nSp9cdBpqL13CqjdMr2L_")
# tings = home.find_all('h3', {'class': '_eYtD2XCVieq6emjKBH3m'})
# for ting in tings:
#     print(ting.text)
# ************************************************************************************
# SELENIUM - Chrome driver method 2.0

# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import time
# from bs4 import BeautifulSoup
#
# driver = webdriver.Chrome(executable_path="C:\\Users\\HE400\\Documents\\chromedriver\\chromedriver.exe")
# url = "https://www.reddit.com/r/wallstreetbets"
# driver.maximize_window()
# driver.get(url)
#
# time.sleep(5)
# elem = driver.find_element_by_tag_name("body")
# no_of_pagedowns = 20
# while no_of_pagedowns:
#     elem.send_keys(Keys.PAGE_DOWN)
#     time.sleep(0.2)
#     no_of_pagedowns -= 1
# time.sleep(2)
# content = driver.page_source.encode('utf-8').strip()
# soup = BeautifulSoup(content, "html.parser")
# titles = soup.find_all('h3', {'class': '_eYtD2XCVieq6emjKBH3m'})
#
# for title in titles:
#     print(str(title))
#
# driver.quit()

# ************************************************************************************
# ************************************************************************************
# trying twitter - Chrome driver method 2.0

# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import time
# from bs4 import BeautifulSoup
#
# driver = webdriver.Chrome(executable_path="C:\\Users\\HE400\\Documents\\chromedriver\\chromedriver.exe")
# url = "https://twitter.com/search?q=Texas&src=typed_query"
# driver.maximize_window()
# driver.get(url)
#
# time.sleep(5)
# elem = driver.find_element_by_tag_name("body")
# no_of_pagedowns = 2
# while no_of_pagedowns:
#     elem.send_keys(Keys.PAGE_DOWN)
#     time.sleep(0.2)
#     no_of_pagedowns -= 1
# time.sleep(2)
# content = driver.page_source.encode('utf-8').strip()
# soup = BeautifulSoup(content, "html.parser")
# titles = soup.find_all('span', {'class': 'css-901oao'})
#
# for title in titles:
#     print(str(title))
#
# driver.quit()

# ************************************************************************************
# ************************************************************************************
# REDDIT COMMENT FROM A THREAD - WAllstreetbets

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup
import html
import re

import pandas as pd
import numpy as np

driver = webdriver.Chrome(executable_path="C:\\Users\\HE400\\Documents\\chromedriver\\chromedriver.exe")
url = "https://www.reddit.com/r/wallstreetbets/comments/lp3lhd/i_have_finally_dug_myself_out_of_a_very/"
driver.maximize_window()
driver.get(url)

ticks = ['GME', 'BB', 'AMC']
commentsR = []
titleR = []

time.sleep(5)
elem = driver.find_element_by_tag_name("body")

# no_of_pagedowns = 20

# while no_of_pagedowns:
#     elem.send_keys(Keys.PAGE_DOWN)
#     time.sleep(0.2)
#     no_of_pagedowns -= 1
# time.sleep(2)
content = driver.page_source.encode('utf-8').strip()
soup = BeautifulSoup(content, "html.parser")
titles = soup.find_all('h3', {'class': '_eYtD2XCVieq6emjKBH3m'})
comments = soup.find_all('p', {'class': '_1qeIAgB0cPwnLhDF9XSiJM'})

# Converts from bs4 to list of Strings
titleConvert = []
commentsConvert = []
commentClean = []
tickTracker = []
for t in titles:
    titleConvert.append(str(t))

for c in comments:
    commentsConvert.append(str(c))

# Cleans the html tags from the strings:
for c in commentsConvert:
    p = re.compile(r'<.*?>')
    txtComment = p.sub('', c)
    commentClean.append(txtComment)
    # print(txtComment)

# for t in titleClean:
#     p = re.compile(r'<.*?>')
#     txtTitle = p.sub('', t)
#     print(txtTitle)

# Checks for ticks in titles:

# for t in txtTitle:
#     if t.split() in ticks:
#         titleR.append(t)
#     print(t, "testing!")
#
# print(titleR, "amazing!")

# Checks for ticks in comments
index = 0

for cc in commentClean:
    while index < len(commentClean):
        wordsInSentence = commentClean[index].split()
        print(wordsInSentence)
        wordsInSentenceIndex = 0
        while wordsInSentenceIndex < len(wordsInSentence):
            if wordsInSentence[wordsInSentenceIndex] in ticks:
                commentsR.append(commentClean[index])
                tickTracker.append(wordsInSentence[wordsInSentenceIndex])
            wordsInSentenceIndex += 1
        index += 1

print(commentsR, 'Array of comments with ticks in them!')

# for cc in commentClean:
#     print(cc, "what am I?")
#     gg = commentClean[cc].split()
#     print(gg, "lmao! get trolled")

# for c in cc:
#     if c in ticks:
#         commentsR.append(c)
#     print(c, "testing")

# print(titleR, "amazing!")
# print(commentsR, "rr!")
# print(str(title))

# for comment in comments:
#    print(str(comment))

# import necessary libraries


driver.quit()

tableData = {'Comments': commentsR,
             'Ticks': tickTracker}
# I could make new array with headers and refer to the cleantxt comment in the table like the example above.
df = pd.DataFrame(tableData)
df.to_csv('filename.csv', index=False, header=['Comments', 'Ticks'])  # Here index=False will remove unnecessary indexing/numbering in your csv.
