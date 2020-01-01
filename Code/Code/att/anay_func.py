from __future__ import print_function
import json
import os
import pickle
import time
import praw
import pandas as pd
import re
import google_test # google_sen
import wordcloud_etc # give_stemmed_dict, count_senti
from pandas.io.json import json_normalize

jsonfile_path = "/Users/ranliu/Desktop/Class-related/CS6474/PJ/aftermid/user_bladerskb/data_cmv_1101.json"

df = pd.DataFrame()
user_str_list = []
with open(jsonfile_path) as jsonfile:

    lines = json.load(jsonfile)
    for items in range(len(lines["data"]["children"])):

        if "body" in lines["data"]["children"][items]["data"]:

            body = lines["data"]["children"][items]["data"]["body"]
            user_str_list.append(body)

def cleanText(text):
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r' ', text)
    text = re.sub(r'\s+', r' ', text)
    text = text.lower()

    text = text.replace(">","")
    text = text.replace("*", "")
    text = text.replace("_", "")
    return text

user_data = cleanText(" ".join(user_str_list))

print(user_data)

#with open("user_data_1101.pkl", "wb") as f:
#    pickle.dump(user_data, f)
#### save data

def main_senti(text):
    sentiment_result = google_test.google_sen(text)
#    new_list = wordcloud_etc.give_stemmed_dict([sentiment_result])
    return wordcloud_etc.count_senti(text, sentiment_result, False)

user_result = main_senti(user_data)

print(user_result)

with open("user_data_1101_sentimentresult.pkl", "wb") as f:
    pickle.dump(user_result, f)

