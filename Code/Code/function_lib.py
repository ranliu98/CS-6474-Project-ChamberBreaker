import praw
import pandas as pd
import datetime
import pprint
import time
from collections import Counter
from tqdm import tqdm
import argparse
import math
import sys
from google_API_use import main_senti
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

reddit = praw.Reddit(client_id="", \
                     client_secret="", \
                     user_agent='', \
                     username='', \
                     password='')

def get_comments(user,ana_limit = 25):
    # name of the dict is submission but this is actually comments. you can verify this by seeing there is only body
    submission_dict = {"author":[], \
                    "created":[], \
                    "created_utc":[], \
                    "permalink": [], \
                    "body": [], \
                    "subreddit":[]}
    for ids in tqdm(reddit.redditor(user).comments.new(limit=ana_limit)):
        comment = reddit.comment(id=ids)

        submission_dict["author"].append(comment.author)
        submission_dict["created"].append(comment.created)
        submission_dict["created_utc"].append(comment.created_utc)
        submission_dict["permalink"].append(comment.permalink)
        submission_dict["body"].append(comment.body)
        submission_dict["subreddit"].append(comment.subreddit)

    comments_data = pd.DataFrame(submission_dict)
    #print(comments_data.shape, comments_data.head())
    #comments_data.to_csv("comments_data_bladerskb_1116.csv")
    #comments_data.to_pickle("comments_data_bladerskb_1116.plk")
    return comments_data

def get_submission(user,ana_limit = 25):
    # code for submission data
    submission_dict = {"author":[], \
                    "created":[], \
                    "created_utc":[], \
                    "permalink": [], \
                    "selftext": [], \
                    "subreddit":[], \
                    "title":[]}

    for ids in tqdm(reddit.redditor(user).submissions.new(limit=ana_limit)):
        submission = reddit.submission(id=ids)
        submission_dict["author"].append(submission.author)
        submission_dict["created"].append(submission.created)
        submission_dict["created_utc"].append(submission.created_utc)
        submission_dict["permalink"].append(submission.permalink)
        submission_dict["selftext"].append(submission.selftext)
        submission_dict["subreddit"].append(submission.subreddit)
        submission_dict["title"].append(submission.title)

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

def main_user_data():

    path = "Dataset/SelfDrivingCars_1116_hot1000.csv"
    sdc_df = pd.read_csv(path)
    user_dict = dict(Counter(sdc_df["author"].tolist()))

    for user in tqdm(user_dict):
        df_com = get_comments(user)
        df_sub = get_submission(user)
        df_user = deal_with_user(df_sub, df_com)
        df_user.to_csv("userdata_%s_1119.csv" %user)
        df_user.to_pickle("userdata_%s_1119.plk" %user)

def entropy(alist):
    # a list of raw numbers [1,2,3,4,5]
    sum_list = sum(alist)
    alist = [x/sum_list for x in alist] # get possibility instead of raw value
    alist = [x**x for x in alist]
    alist = [-math.log(x) for x in alist]
    return sum(alist)

class User_properties():
    def __init__(self,df,time_period,start_selection):
        '''
        :param df: input dataframe to calculate properties
        :param time_period: time period to seperate dataframe values
        :param start_selection: time to begin. Actually the time when the data collection ends.
        '''
        self.df = df
        author = df["author"].tolist()
        self.name = author[0]
        self.time_period = time_period
        self.start_selection = start_selection
        self.sep_mon_df = pd.DataFrame()

    def user_at_least(self, created_utc = "created_utc", number_of_time_period = 1):
        '''
        :param created_utc: the utc created time inside the original dataframe
        :param number_of_time_period: number of
        :return: a judgement, if the df return there is him, if it does not return, forget it. Default = 1
                 [keep user with at least start_selection-number*time_period]
        '''
        df = self.df
        start_selection = self.start_selection
        time_period = self.time_period
        # pasta this line to get only limited number of users. Filter out the users who do not have post at start_selection-time_period time.
        if (df[created_utc].tolist())[-1] <= start_selection-number_of_time_period*time_period:
            return True
        else:
            return False

    def user_total_time(self, created_utc = "created_utc"):
        '''
        :param created_utc:
        :return: the number of how many "3_months" passed by one user
        '''
        df = self.df
        time_period = self.time_period
        df_time_list = df[created_utc].tolist()
        return round(math.ceil((df_time_list[0]-df_time_list[-1])/time_period))

    def user_sub(self):
        '''
        :return: Counter object of user's sub. Use dict() to see results.
        '''
        df = self.df
        return (Counter(df["subreddit"].tolist()).most_common())

    def user_sub_sep(self, created_utc = "created_utc"):
        '''
        # divide their data into different time period. Each one will return a list,
        :param created_utc:
        :return: a list containing several dicts with 'sub': 'freq' mentioned, empty means no acts.
        '''
        df = self.df
        time_period = self.time_period
        start_selection = self.start_selection

        alist = []
        df_time_list = df[created_utc].tolist()
        total_sep = round(math.ceil((df_time_list[0]-df_time_list[-1])/time_period))
        for pe in range(total_sep-1):
            period_df = df[df[created_utc]>start_selection-(pe+1)*time_period]
            period_df = period_df[period_df[created_utc]<=start_selection-pe*time_period]# (at_least_selection, at_least_selection+time_period]
            period_sub = dict(Counter(period_df["subreddit"].tolist()))
            alist.append(period_sub)
        #print(len(alist)) : how many time period happened before
        return alist

    def user_sub_sep_get_text(self, created_utc = "created_utc"):
        '''
        :param created_utc:
        :return: return a list of text.
        '''
        df = self.df
        time_period = self.time_period
        start_selection = self.start_selection
        sep_mon_df = self.sep_mon_df

        alist = []
        df_time_list = df[created_utc].tolist()
        total_sep = round(math.ceil((df_time_list[0]-df_time_list[-1])/time_period))
        print("total_sep :",total_sep)

        if total_sep != 1:
            for pe in range(total_sep-1):
                period_df = df[df[created_utc]>start_selection-(pe+1)*time_period]
                period_df = period_df[period_df[created_utc]<=start_selection-pe*time_period]# (at_least_selection, at_least_selection+time_period]

                period_text_list = period_df["body"].tolist()
                text = " ".join(period_text_list)
                if sys.getsizeof(text) > 1000000:
                    print("--- too large a text file!--- Please do it seperately!!! ---")
                    text = ""
                alist.append(text)
        else:
            df_text_list = df["body"].tolist()
            text = " ".join(df_text_list)
            if sys.getsizeof(text) > 1000000:
                print("--- too large a text file!--- Please do it seperately!!! ---")
                text = ""
            alist.append(text)

        author = df["author"].tolist()
        sep_mon_df["user_name"] = [author[0]]*len(alist)
        sep_mon_df["time_period"] = list(range(len(alist)))
        sep_mon_df["text"] = alist

        try:
            sep_mon_df["senti_result"] = [main_senti(atext) for atext in alist]
        except:
            print("this user df has a problem")
        self.sep_mon_df = sep_mon_df
        return alist

    def user_sub_entropy(self):
        '''
        :return: user's entropy calculated according to its subreddits frequency.
        '''
        df = self.df
        user_sub_dict = dict(Counter(df["subreddit"].tolist()).most_common())

        return entropy(user_sub_dict.values())

def concat_result(alist):
    # input: a list of all results of dictionary form
    New_dict = {}
    for dicts in alist:
        for keys in dicts.keys():
            if keys not in New_dict.keys():
                New_dict.update({keys: dicts[keys]})
            else:
                New_dict.update({keys: [New_dict[keys][0]+ dicts[keys][0], New_dict[keys][1]+ dicts[keys][1]]})
    return New_dict

def concat_result_with_sentiment(alist):
    # input: a list of all results of dictionary form

    def new_value(a_value):
        # input: dicts[keys] which is [senti, freq]
        result = {'pos':[],'neutr':[],'neg':[]} # posi neutr neg
        assert type(a_value[0]) == float

        if a_value[0] > 0:
            result['pos'].extend(a_value)
        elif a_value[0] == 0:
            result['neutr'].extend(a_value)
        elif a_value[0] < 0:
            result['neg'].extend(a_value)
        return result

    New_dict = {}
    for dicts in tqdm(alist):
        for keys in dicts.keys():
            if keys not in New_dict.keys():
                temp = new_value(dicts[keys])
                New_dict.update({keys: {'pos':[],'neutr':[],'neg':[]}})
                New_dict[keys]['pos'].append(temp['pos'])
                New_dict[keys]['neutr'].append(temp['neutr'])
                New_dict[keys]['neg'].append(temp['neg'])
                #New_dict.update({keys: new_value(dicts[keys])})
            else:
                temp = new_value(dicts[keys])
                New_dict[keys]['pos'].append(temp['pos'])
                New_dict[keys]['neutr'].append(temp['neutr'])
                New_dict[keys]['neg'].append(temp['neg'])
            #with open("see_test.txt",'a') as file:
            #    file.write("here:\n{}\n".format(New_dict))

    return New_dict

def user_recommendation():
    sub_path = "SelfDrivingCars_top210_withresult_full.csv"
    user_path = "test_one_user.csv"

    user_df = pd.read_csv(user_path)
    user_df["senti_result"] = user_df["senti_result"].apply(eval)
    All_user = concat_result(user_df["senti_result"].tolist())

    print(All_user)
    print("---"*30)

    sub_df = pd.read_csv(sub_path)
    sub_df["mother_senti_result"] = sub_df["mother_senti_result"].apply(eval)
    sub_df["child_senti_result"] = sub_df["child_senti_result"].apply(eval)
    Mother_result = concat_result(sub_df["mother_senti_result"].tolist())
    #sub_df["child_senti_result"] = sub_df["child_senti_result"].apply(concat_result)
    Child_result = concat_result(sub_df["child_senti_result"].tolist())

    All_sub = concat_result([Mother_result,Child_result])

    print(All_sub)

    recommend = {}

    for keys in All_sub.keys():
        if keys not in All_user:
            recommend.update({keys:All_sub[keys][1]})

    print("here is recommendation")
    print(recommend)
    d = list(zip(recommend.values(), recommend.keys()))
    d = sorted(d, reverse=True)
    print(d)

def UrlRoutine(permalink):
    """ Returns the unique submission id from the match thread url """
    submission_id = reddit.submission(url="https://www.reddit.com"+permalink)
    return str(submission_id)

def post_fresh_recommendation(All_user,sub_path):
    sub_df = pd.read_csv(sub_path)
    sub_df["mother_senti_result"] = sub_df["mother_senti_result"].apply(eval)
    sub_df["child_senti_result"] = sub_df["child_senti_result"].apply(eval)
    sub_df["post_senti_result"] = sub_df.apply(lambda row: concat_result([row['mother_senti_result'], row['child_senti_result']]), axis=1)

    recommend = {}
    def post_recommend(All_sub):
        fresh_points = 0
        fresh_list = {}
        total_points = 0
        # All_sub here is actually one post dict result
        for keys in All_sub.keys():
            if keys not in All_user:
                fresh_points = fresh_points+ All_sub[keys][1]
                fresh_list.update({keys:All_sub[keys][1]})
        for keys in All_sub.keys():
            total_points = total_points+All_sub[keys][1]

        d = list(zip(fresh_list.values(), fresh_list.keys()))
        d = sorted(d, reverse=True)

        fresh_topic_list = [items[1] for items in d]

        if len(fresh_list)>5:
            fresh_topic_list = fresh_topic_list[:5]

        return fresh_points/total_points, fresh_topic_list

    sub_df["post_freshness"], sub_df["fresh_topic_to_user"] = zip(*sub_df["post_senti_result"].apply(post_recommend))
    sub_df = sub_df.sort_values(by='post_freshness', ascending=False)
    sub_df = sub_df[:10]
    sub_df["id"] = sub_df["permalink"].apply(UrlRoutine)
    #print(sub_df["id"].tolist())

    post_result = zip(sub_df["post_freshness"].tolist(), sub_df["fresh_topic_to_user"].tolist(), sub_df["id"].tolist())

    return post_result

def main():
    # recommendation final
    parser = argparse.ArgumentParser()
    parser.add_argument("--username",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the user. A string without '' and must be correct.")
    parser.add_argument("--sub_reddit_path",
                        default="apple_top320_withresult_full.csv",
                        type=str,
                        help="sub reddit datapath inside your own laptop.")
    parser.add_argument("--ana_limit",
                        default=50,
                        type=int,
                        help="The amount of posts you want to use.")
    args = parser.parse_args()

    user = args.username
    sub_path = args.sub_reddit_path
    ana_limit = args.ana_limit


    time_period_default = datetime.datetime.timestamp(
        datetime.datetime.strptime('2019-11-30 00:00:00', "%Y-%m-%d %H:%M:%S")) - datetime.datetime.timestamp(
        datetime.datetime.strptime('2019-08-31 00:00:00', "%Y-%m-%d %H:%M:%S"))  # "3 months"
    start_selection_default = datetime.datetime.timestamp(datetime.datetime.strptime('2019-11-30 00:00:00',
                                                                             "%Y-%m-%d %H:%M:%S"))  # set up most recent time as '2019-11-31 0:0:0'

    df_com = get_comments(user)
    df_sub = get_submission(user)
    df_user = deal_with_user(df_sub, df_com)
    df_user = df_user[:ana_limit]

    print("df_user:\n",df_user.head())

    user = User_properties(df_user, time_period = time_period_default, start_selection = start_selection_default)
    user.user_sub_sep_get_text()
    user_df = user.sep_mon_df
    All_user = concat_result(user_df["senti_result"].tolist())

    print("All_user \n {}".format(All_user))

    sub_df = pd.read_csv(sub_path)
    sub_df["mother_senti_result"] = sub_df["mother_senti_result"].apply(eval)
    sub_df["child_senti_result"] = sub_df["child_senti_result"].apply(eval)
    Mother_result = concat_result(sub_df["mother_senti_result"].tolist())
    Child_result = concat_result(sub_df["child_senti_result"].tolist())

    All_sub = concat_result([Mother_result,Child_result])

    for keys in All_sub.keys():
        if len(keys) <= 2:
            All_sub.update({keys:[0,0]})

    recommend = {}

    for keys in All_sub.keys():
        if keys not in All_user:
            recommend.update({keys: All_sub[keys][1]})

    print("here is recommendation")
    d = list(zip(recommend.values(), recommend.keys()))
    d = sorted(d, reverse=True)
    print(d[:100])

    post_result = post_fresh_recommendation(All_user, sub_path)
    return [i for i in post_result]

def main_post():
    # recommendation post final
    parser = argparse.ArgumentParser()
    parser.add_argument("--username",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the user. A string without '' and must be correct.")
    parser.add_argument("--sub_reddit_path",
                        default="/Users/ranliu/Desktop/Class-related/CS6474/PJ/aftermid/SDC_Dataset/SelfDrivingCars_top210_withresult_full.csv",
                        type=str,
                        help="sub reddit datapath inside your own laptop.")
    parser.add_argument("--ana_limit",
                        default=25,
                        type=int,
                        help="The amount of posts you want to use.")
    args = parser.parse_args()

    user = args.username
    sub_path = args.sub_reddit_path
    ana_limit = args.ana_limit

    time_period_default = datetime.datetime.timestamp(
        datetime.datetime.strptime('2019-11-30 00:00:00', "%Y-%m-%d %H:%M:%S")) - datetime.datetime.timestamp(
        datetime.datetime.strptime('2019-08-31 00:00:00', "%Y-%m-%d %H:%M:%S"))  # "3 months"
    start_selection_default = datetime.datetime.timestamp(datetime.datetime.strptime('2019-11-30 00:00:00',
                                                                             "%Y-%m-%d %H:%M:%S"))  # set up most recent time as '2019-11-31 0:0:0'

    df_com = get_comments(user)
    df_sub = get_submission(user)
    df_user = deal_with_user(df_sub, df_com)
    df_user = df_user[:ana_limit]

    user = User_properties(df_user, time_period = time_period_default, start_selection = start_selection_default)
    user.user_sub_sep_get_text()
    user_df = user.sep_mon_df
    All_user = concat_result(user_df["senti_result"].tolist())

    post_result = post_fresh_recommendation(All_user, sub_path)
    return [i for i in post_result]

def main_sentiment():
    sub_path = "/Users/ranliu/Desktop/Class-related/CS6474/PJ/aftermid/apple_dataset/apple_top320_withresult_full.csv"

    sub_df = pd.read_csv(sub_path)
    sub_df["mother_senti_result"] = sub_df["mother_senti_result"].apply(eval)
    sub_df["child_senti_result"] = sub_df["child_senti_result"].apply(eval)

    ##### get recommendation: majority #####
    Mother_result_1 = concat_result(sub_df["mother_senti_result"].tolist())
    Child_result_1 = concat_result(sub_df["child_senti_result"].tolist())
    All_sub_1 = concat_result([Mother_result_1,Child_result_1])
    for keys in All_sub_1.keys():
        if len(keys) <= 2:
            All_sub_1.update({keys:[0,0]})
    recommend = {}
    for keys in All_sub_1.keys():
        recommend.update({keys: All_sub_1[keys][1]})
    d = list(zip(recommend.values(), recommend.keys()))
    d = sorted(d, reverse=True)
    with open("recommendation_apple_top100.txt",'w') as file:
        file.write("{}".format(d[:100]))

    ##### get sentiment: find majority #####
    Mother_result = concat_result_with_sentiment(sub_df["mother_senti_result"].tolist())
    Child_result = concat_result_with_sentiment(sub_df["child_senti_result"].tolist())
    def concat_mother_child(alist):
        # input: a list of 2 dict

        New_dict = {}
        for dicts in tqdm(alist):
            for keys in dicts.keys():
                if keys not in New_dict.keys():
                    New_dict.update({keys: dicts[keys]})
                else:
                    temp = dicts[keys]
                    for things in temp['pos']:
                        New_dict[keys]['pos'].append(things)
                    for things in temp['neutr']:
                        New_dict[keys]['neutr'].append(things)
                    for things in temp['neg']:
                        New_dict[keys]['neg'].append(things)
                # with open("see_test.txt",'a') as file:
                #    file.write("here:\n{}\n".format(New_dict))

        return New_dict
    All_sub = concat_mother_child([Mother_result,Child_result])
    with open("All_sub_1201_apple.txt", 'w') as file:
        file.write("{}".format(All_sub))

    with open("All_sub_1201_apple_MOTHER.txt", 'w') as file:
        file.write("{}".format(Mother_result))

def main_sentiment_deal():
    recommendation_top100 = [(6151, 'one'), (3319, 'all'), (3228, 'Apple'), (2654, 'app'), (2129, 'phone'), (1800, 'thing'), (1328, 'use'), (1286, 'way'), (1271, 'ear'), (1149, 'pro'), (1027, 'work'), (918, 'people'), (747, 'some'), (737, 'Pro'), (693, 'AirPods'), (685, 'Mac'), (671, 'any'), (663, 'sound'), (632, 'issue'), (608, 'ones'), (576, 'battery'), (563, 'screen'), (551, 'something'), (530, 'apple'), (522, 'device'), (514, 'iPhone'), (483, 'update'), (470, 'anyone'), (443, 'iPad'), (402, 'time'), (376, 'lot'), (368, 'phones'), (360, 'App'), (355, 'more'), (354, 'noise'), (347, 'case'), (346, 'problem'), (346, 'ears'), (345, 'quality'), (345, 'apps'), (340, 'music'), (340, 'keyboard'), (336, 'camera'), (335, 'things'), (333, 'anything'), (331, 'price'), (323, 'product'), (312, 'store'), (306, 'point'), (303, 'video'), (300, 'devices'), (300, 'MacBook'), (293, 'watch'), (282, 'headphones'), (277, 'man'), (271, 'top'), (270, 'Thanks'), (264, 'key'), (260, 'support'), (253, 'shit'), (252, 'most'), (249, 'someone'), (247, 'option'), (247, 'list'), (247, 'back'), (244, 'head'), (243, 'laptop'), (242, 'times'), (242, 'money'), (242, 'air'), (240, 'bit'), (238, 'end'), (236, 'everything'), (234, 'feature'), (232, 'reason'), (231, 'upgrade'), (231, 'much'), (231, 'issues'), (228, 'version'), (228, 'design'), (227, 'mode'), (224, 'review'), (222, 'pair'), (221, 'charge'), (219, 'help'), (218, 'same'), (218, 'model'), (215, 'idea'), (211, 'products'), (209, 'Anyone'), (207, 'China'), (205, 'need'), (204, 'Music'), (203, 'audio'), (202, 'service'), (201, 'airpods'), (196, 'user'), (196, 'release'), (195, 'size'), (195, 'nothing')]
    recommendation_top100 = recommendation_top100[:80]
    recommendation_top100 = [items[1] for items in recommendation_top100]
    print("recommendation_top100\n{}\n".format(recommendation_top100))
    with open("All_sub_1201_apple.txt","r") as file:
        senti_result = file.read()
        senti_result = eval(senti_result)

    top100_senti = {}
    top100_senti_precentage = {}
    top100_senti_pos_over_neg = {}
    top100_senti_pos_over_neg_list = []
    for items in recommendation_top100:

        try:
            def count_freq(senti):
                freq_number = 0
                for lists in senti_result[items][senti]:
                    if len(lists) == 2:
                        freq_number = freq_number + lists[1]
                return freq_number

            pos = count_freq('pos')
            neutr = count_freq('neutr')
            neg = count_freq('neg')
            items_senti = {'pos': pos, 'neutr': neutr, 'neg': neg}
            items_senti_precentage = {'pos': pos/(pos+neutr+neg), 'neutr': neutr/(pos+neutr+neg), 'neg': neg/(pos+neutr+neg)}
            items_senti_pos_over_neg = pos/neg

            top100_senti.update({items:items_senti})
            top100_senti_precentage.update({items:items_senti_precentage})
            top100_senti_pos_over_neg.update({items:items_senti_pos_over_neg})
            top100_senti_pos_over_neg_list.append(items_senti_pos_over_neg)

        except:
            print("Key_error")

    print("{}\n\n{}\n\n{}".format(top100_senti, top100_senti_precentage, top100_senti_pos_over_neg))

    top100_senti_pos_over_neg_list = [math.log(i) for i in top100_senti_pos_over_neg_list]

    x = range(len(top100_senti_pos_over_neg_list))
    y = top100_senti_pos_over_neg_list
    y_1 = len(top100_senti_pos_over_neg_list)*[0]


    fig = plt.figure(figsize=(16, 9.6))
    ax = fig.add_subplot(111)
    #plt.plot(x, y_1, 'r')

    grid_x = [i-5 for i in list(range(200))]

    kkk_scatter = [i for i in zip(x,y)]
    kkk_scatter_pos = [i for i in kkk_scatter if i[1] > 0.4]
    kkk_scatter_neg = [i for i in kkk_scatter if i[1] < -0.4]
    kkk_scatter_neutr = [i for i in kkk_scatter if i[1] <= 0.4 and i[1] >= -0.4]

    plt.scatter([i[0] for i in kkk_scatter_pos], [i[1] for i in kkk_scatter_pos], alpha=0.8, c='g')
    plt.scatter([i[0] for i in kkk_scatter_neg], [i[1] for i in kkk_scatter_neg], alpha=0.8, c='r')
    plt.scatter([i[0] for i in kkk_scatter_neutr], [i[1] for i in kkk_scatter_neutr], alpha=0.4, c='k')

    pos_y_list = [-1.9459101490553132, -1.6094379124341003,-1.0986122886681098,
                  -0.6931471805599453, -0.4054651081081644, 0.0, 0.4054651081081644, 0.6931471805599453,
                  1.0986122886681098,1.6094379124341003,1.791759469228055]
    plt.yticks(np.array(pos_y_list), ["{}%".format(round(100* math.exp(y))) for y in pos_y_list])
    plt.ylim([-2.1, 1.7])
    plt.xlim([-1, 83])

    for y_g in pos_y_list:
        if y_g>0.3:
            grid_y = [y_g]*len(grid_x)
            plt.plot(grid_x, grid_y, 'g', alpha=0.8, linewidth=1, linestyle="-")
        elif y_g<-0.3:
            grid_y = [y_g]*len(grid_x)
            plt.plot(grid_x, grid_y, 'r', alpha=0.8, linewidth=1, linestyle="-")
        else:
            grid_y = [y_g]*len(grid_x)
            plt.plot(grid_x, grid_y, 'k', alpha=0.8, linewidth=1, linestyle="-")

    for i, txt in enumerate(y):
        plt.annotate(recommendation_top100[i], xy=(x[i], y[i]))
    #ax.set_yscale('symlog')
    #plt.grid(True)
    #plt.show()
    plt.xlabel("Topics: ranked by frequency")
    plt.ylabel("Sentiment diversity: positive freq/negative freq")
    plt.savefig("figure.png",dpi=400)

main_sentiment_deal()

#print([math.log(y) for y in [1.5]])
