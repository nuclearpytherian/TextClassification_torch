# 네이버 영화 평점 크롤러

import requests
from bs4 import BeautifulSoup
import random
import time
from itertools import chain
import pandas as pd
import os
import argparse
from tqdm import tqdm


def naver_movie_comment_crawler(BASE_URL, TABLE_, TITLE_, SCORE_, COMMENT_, PAGE):
    url = BASE_URL.format(PAGE)
    res = requests.get(url)
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'lxml')
        tds = soup.select(TABLE_)
        list_ = []
        for td in tds:
            title = td.select_one(TITLE_).text.strip()
            if title == '':
                title = td.select_one(TITLE_).next_sibling.strip()

            score = td.select_one(SCORE_).text.strip()
            if score == '':
                score = td.select_one(SCORE_).next_sibling.strip()

            comment = td.select_one(COMMENT_).text.strip()
            if comment == '':
                comment = td.select_one(COMMENT_).next_sibling.strip()

            if title != '' and score != '' and comment != '':
                list_.append((title, int(score), comment))

        return list_

    else:
        print("Webpage not available.")



def page_crawling(start_page, end_page):

    BASE_URL = 'https://movie.naver.com/movie/point/af/list.naver?&page={}'
    TABLE_ = 'td.title'
    TITLE_ = 'a.movie.color_b'
    SCORE_ = 'em'
    COMMENT_ = 'br'

    comment_list = []
    for page in tqdm(range(start_page, end_page)):
        crw = naver_movie_comment_crawler(BASE_URL, TABLE_, TITLE_, SCORE_, COMMENT_, page)
        comment_list.append(crw)
        interval = round(random.uniform(0.2, 1.2), 2)
        time.sleep(interval)

    df = pd.DataFrame(list(chain(*comment_list)), columns=['title', 'score', 'comment'])
    df = df.dropna()
    return df


def select_data_df_and_toCSV(df, postive_score_cut, negative_score_cut):
    df['label'] = list(map(lambda x: 1 if int(x) >= postive_score_cut else (0 if int(x) <= negative_score_cut else 2), df['score'].values))
    df = df[df['label'] != 2]
    df.to_csv('data/movie_naver.csv', encoding='utf-8-sig', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_page', '-s',type=int, default=1)
    parser.add_argument('--end_page', '-e',type=int, default=10)
    args = parser.parse_args()

    df = page_crawling(args.start_page, args.end_page)

    if not os.path.isdir('data'):
        os.mkdir('data')
    select_data_df_and_toCSV(df,
                             postive_score_cut=7,
                             negative_score_cut=3)

    print(f"{len(df)} lines are parsed.")
    print("--FINISHED--")


    ### 참조 : https://ysyblog.tistory.com/59
    # BASE_URL = 'https://movie.naver.com/movie/point/af/list.naver?&page={}'
    # comment_list = []
    # for page in tqdm(range(args.start_page, args.end_page)):
    #     url = BASE_URL.format(page)
    #     res = requests.get(url)
    #     if res.status_code == 200:
    #         soup = BeautifulSoup(res.text, 'lxml')
    #         tds = soup.select('table.list_netizen > tbody > tr > td.title')
    #         for td in tds:
    #             movie_title = td.select_one('a.movie').text.strip()
    #             # link = td.select_one('a.movie').get('href')
    #             # link = parse.urljoin(BASE_URL, link)
    #             score = td.select_one('div.list_netizen_score > em').text.strip()
    #             comment = td.select_one('br').next_sibling.strip()
    #             comment_list.append((movie_title, score, comment))
    #         interval = round(random.uniform(0.2, 1.2), 2)
    #         time.sleep(interval)
    #
    # if not os.path.isdir('data'):
    #     os.mkdir('data')
    #
    # df = pd.DataFrame(comment_list, columns=['title', 'score', 'comment'])
    # postive_score_cut = 7
    # negative_score_cut = 3
    # df['label'] = list(map(lambda x: 1 if int(x) >= postive_score_cut else (0 if int(x) <= negative_score_cut else 2), df['score'].values))
    # df = df[df['label'] != 2]
    # df.to_csv('data/movie_naver.csv', encoding='utf-8-sig', index=False)
    # print(f"{len(df)} lines are parsed.")
    # print("--FINISHED--")

