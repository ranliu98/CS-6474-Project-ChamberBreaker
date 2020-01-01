import praw
import pandas as pd
import datetime as dt
import pprint
import time
from collections import Counter
from tqdm import tqdm

reddit = praw.Reddit(client_id='', \
                     client_secret='', \
                     user_agent='', \
                     username='', \
                     password='')

path = "/home/rliu361/class_sc/SelfDrivingCars_1116_hot1000.csv"
sdc_df = pd.read_csv(path)
user_dict = dict(Counter(sdc_df["author"].tolist()))
#print(user_dict)

def dict_slice(adict, start):
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[start:]:
        dict_slice[k] = adict[k]
    return dict_slice

def get_comments(user):
    # name of the dict is submission but this is actually comments. you can verify this by seeing there is only body
    submission_dict = {"author":[], \
                    "created":[], \
                    "created_utc":[], \
                    "permalink": [], \
                    "body": [], \
                    "subreddit":[]}
    for ids in tqdm(reddit.redditor(user).comments.new(limit=None)):
        comment = reddit.comment(id=ids)

        submission_dict["author"].append(comment.author)
        submission_dict["created"].append(comment.created)
        submission_dict["created_utc"].append(comment.created_utc)
        submission_dict["permalink"].append(comment.permalink)
        submission_dict["body"].append(comment.body)
        submission_dict["subreddit"].append(comment.subreddit)

        time.sleep(1)
    comments_data = pd.DataFrame(submission_dict)
    #print(comments_data.shape, comments_data.head())
    #comments_data.to_csv("comments_data_bladerskb_1116.csv")
    #comments_data.to_pickle("comments_data_bladerskb_1116.plk")
    return comments_data

def get_submission(user):
    # code for submission data
    submission_dict = {"author":[], \
                    "created":[], \
                    "created_utc":[], \
                    "permalink": [], \
                    "selftext": [], \
                    "subreddit":[], \
                    "title":[]}

    for ids in tqdm(reddit.redditor(user).submissions.new(limit=None)):
        submission = reddit.submission(id=ids)
        submission_dict["author"].append(submission.author)
        submission_dict["created"].append(submission.created)
        submission_dict["created_utc"].append(submission.created_utc)
        submission_dict["permalink"].append(submission.permalink)
        submission_dict["selftext"].append(submission.selftext)
        submission_dict["subreddit"].append(submission.subreddit)
        submission_dict["title"].append(submission.title)
        time.sleep(1)

    submission_data = pd.DataFrame(submission_dict)
    #print(submission_data.head())
    #submission_data.to_csv("submission_data_bladerskb_1116.csv")
    #submission_data.to_pickle("submission_data_bladerskb_1116.plk")
    return submission_data

def deal_with_user(df_sub, df_com):
    #df_sub = pd.read_csv(path_sub)
    df_sub["body"] = df_sub["title"]+". "+ df_sub["selftext"]
    df_sub = df_sub.drop(['title', 'selftext'], axis=1)
    #df_com = pd.read_csv(path_com)

    df_sub = df_sub.append(df_com)
    df_sub = df_sub.dropna()
    df_sub = df_sub.sort_values(by=["created_utc"],ascending=False)

    print(df_sub.shape)

    '''
    sep = round(df_sub.shape[0]/8)

    df_recent_1 = df_sub[:sep]
    df_recent_2 = df_sub[sep: 2*sep]
    df_recent_3 = df_sub[2* sep: 3 * sep]
    df_recent_4 = df_sub[3* sep: 4 * sep]
    df_after_1 = df_sub[4 * sep:5* sep]
    df_after_2 = df_sub[5 * sep: 6*sep]
    df_after_3 = df_sub[6* sep: 7 * sep]
    df_after_4 = df_sub[7* sep:]
    #df_after = df_sub[round(df_sub.shape[0]/2):]
    return [df_recent_1, df_recent_2, df_recent_3, df_recent_4, df_after_1, df_after_2, df_after_3, df_after_4]
    ''' # previous return
    return df_sub

user_dict = dict_slice(user_dict,234)

for user in tqdm(user_dict):
    df_com = get_comments(user)
    df_sub = get_submission(user)
    df_user = deal_with_user(df_sub, df_com)
    df_user.to_csv("userdata_%s_1119.csv" %user)
    df_user.to_pickle("userdata_%s_1119.plk" %user)

#print(submission.title) # to make it non-lazy
#pprint.pprint(vars(submission))
