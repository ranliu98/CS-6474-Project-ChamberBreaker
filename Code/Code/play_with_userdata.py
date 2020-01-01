import praw
import pandas as pd
import datetime as dt
import pprint
import time
import numpy as np
from collections import Counter
from tqdm import tqdm
import fnmatch
import os
import datetime
import math
import sys
import matplotlib.pyplot as plt
from google_API_use import main_senti


time_period = datetime.datetime.timestamp(datetime.datetime.strptime('2019-11-30 00:00:00', "%Y-%m-%d %H:%M:%S")) - datetime.datetime.timestamp(datetime.datetime.strptime('2019-08-31 00:00:00', "%Y-%m-%d %H:%M:%S")) #"3 months"
#time_period = datetime.datetime.timestamp(datetime.datetime.strptime('2019-11-30 00:00:00', "%Y-%m-%d %H:%M:%S")) - datetime.datetime.timestamp(datetime.datetime.strptime('2019-10-30 00:00:00', "%Y-%m-%d %H:%M:%S")) #"3 months"
start_selection = datetime.datetime.timestamp(datetime.datetime.strptime('2019-11-30 00:00:00', "%Y-%m-%d %H:%M:%S")) # set up most recent time as '2019-11-31 0:0:0'
all_sub_Counter=Counter()

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
        #print(alist)
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
        for pe in range(total_sep-1):
            period_df = df[df[created_utc]>start_selection-(pe+1)*time_period]
            period_df = period_df[period_df[created_utc]<=start_selection-pe*time_period]# (at_least_selection, at_least_selection+time_period]

            period_text_list = period_df["body"].tolist()
            text = " ".join(period_text_list)
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
        return alist

    def user_sub_entropy(self):
        '''
        :return: user's entropy calculated according to its subreddits frequency.
        '''
        df = self.df
        user_sub_dict = dict(Counter(df["subreddit"].tolist()).most_common())

        return entropy(user_sub_dict.values())

def all_sub(df, all_sub_Counter):
    all_sub_Counter = all_sub_Counter + Counter(df["subreddit"].tolist())
    print(dict(all_sub_Counter.most_common()))
    return all_sub_Counter

def sepe_user_sub_entropy(alist):
    # return a list of numbers
    entropy_num = []
    for items in alist:
        entropy_num.append(entropy(items.values()))

    return entropy_num

def main():

    x_list = []
    y_list = []
    for file in os.listdir("/Users/ranliu/Desktop/Class-related/CS6474/PJ/aftermid/User_data/userdata_first233"):
        if fnmatch.fnmatch(file, '*.csv'):
            # each df is one user's data.
            df = pd.read_csv(os.path.join('/Users/ranliu/Desktop/Class-related/CS6474/PJ/aftermid/User_data/userdata_first233', file))

            try:
                df["created_utc"] = df["created_utc"].apply(lambda x: round(float(x)))
                df["created_datetime"] = df["created_utc"].apply(lambda x: datetime.datetime.fromtimestamp(x)) # create a new line with readable datetime

                user = User_properties(df,time_period,start_selection)

                if user.user_at_least(number_of_time_period = 1):
                    #print(user.user_total_time(), user.user_sub_entropy())

                    sub_x_list = [items + 1 for items in list(range(user.user_total_time() - 1))]
                    sub_y_list = sepe_user_sub_entropy(user.user_sub_sep())
                    new_x = []
                    new_y = []

                    for i, items in enumerate(sub_y_list):
                        if items != -1:
                            new_y.append(items)
                            new_x.append(sub_x_list[i])

                    x_list.extend(new_x)
                    y_list.extend(new_y)

                    if len(user.user_sub_sep()) >= 8:
                        ...
            except:
                print("Error")
    #plt.plot([2.5]*150,'r')
    zip_result = zip(x_list,y_list)
    zip_result = sorted(zip_result)

    def get_pic_ave(count):
        number_place = []
        for sets in zip_result:
            if sets[0] == count:
                number_place.append(sets[1])
        here_result = 0.001
        try:
            here_result = sum(number_place)/len(number_place)
        except:
            print("No this is weird")
        return here_result

    ave_x_list = [i+1 for i in list(range(50))]
    ave_y_list = []
    for count in ave_x_list:
        ave_y_list.append(get_pic_ave(count))

    plt.scatter(x_list, y_list, alpha=0.4, edgecolors='face',label='user_data scatter plot')
    z1 = np.polyfit(ave_x_list, ave_y_list, 2)  # 用3次多项式拟合
    p1 = np.poly1d(z1)
    yvals = p1(ave_x_list)
    plot2 = plt.plot(ave_x_list, yvals, 'k', alpha=0.7, label='2rd polyfit values')
    #plt.plot(ave_x_list, ave_y_list, 'r')
    plt.legend(fontsize = 'large',loc='upper right',bbox_to_anchor=(0.9, 0.9))
    plt.grid(True,linestyle='-',alpha=0.5)
    plt.xlabel('Time: in 3 months')
    plt.ylabel("User Entropy: indicator of users' diversity")
    plt.xlim(-0.5,53.5)
    plt.show()
                    #all_sub_Counter = all_sub(df, all_sub_Counter) # flexible can be used elsewhere

def get_user_result():
    for file in os.listdir("/Users/ranliu/Desktop/Class-related/CS6474/PJ/aftermid/User_data/userdata_first233"):
        if fnmatch.fnmatch(file, '*.csv'):
            # each df is one user's data.
            df = pd.read_csv(os.path.join('/Users/ranliu/Desktop/Class-related/CS6474/PJ/aftermid/User_data/userdata_first233', file))

            try:
                user = User_properties(df, time_period, start_selection)

                if user.user_at_least(number_of_time_period):
                    # print(user.user_total_time(), user.user_sub_entropy())
                    user.user_sub_sep_get_text()
                    user.sep_mon_df.to_csv("user_result_{}.csv".format(user.name))

                    if len(user.user_sub_sep()) >= 8:
                        ...
            except:
                print("Error")

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

def concat_result_with_sep_freq(alist):
    # input: a list of all results of dictionary form
    New_dict = {}
    for dicts in alist:
        for keys in dicts.keys():
            if keys not in New_dict.keys():
                New_dict.update({keys: [dicts[keys][0],1]})
            else:
                New_dict.update({keys: [New_dict[keys][0]+ dicts[keys][0], New_dict[keys][1]+ 1]})
    return New_dict

def user_recommendation():
    sub_path = "/Users/ranliu/Desktop/Class-related/CS6474/PJ/aftermid/SelfDrivingCars_top210_withresult_test.csv"
    user_path = "/Users/ranliu/Desktop/Class-related/CS6474/PJ/aftermid/test_one_user.csv"

    user_df = pd.read_csv(user_path)
    user_df["senti_result"] = user_df["senti_result"].apply(eval)
    All_user = concat_result(user_df["senti_result"].tolist())

    print(All_user)
    print("---"*30)

    sub_df = pd.read_csv(sub_path)
    sub_df["mother_senti_result"] = sub_df["mother_senti_result"].apply(eval)
    sub_df["child_senti_result"] = sub_df["child_senti_result"].apply(eval)
    Mother_result = concat_result(sub_df["mother_senti_result"].tolist())
    sub_df["child_senti_result"] = sub_df["child_senti_result"].apply(concat_result)
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

def plot_senti(path):

    df = pd.read_csv(path)
    df["senti_result"] = df["senti_result"].apply(eval)
    All_user = concat_result_with_sep_freq(df["senti_result"].tolist())

    for keys in All_user:
        All_user.update({keys:[All_user[keys][1],All_user[keys][0]]})
    d = list(zip(All_user.values(), All_user.keys()))
    d = sorted(d, reverse=True)
    d_list = [items[1] for items in d[:50]]
    #d_list = ['car', "cars", 'self-driving', 'Tesla']


    senti_dict_list = df["senti_result"].tolist()
    def draw_pic(d_list, senti_dict_list):
        for topics in d_list:
            x_list = []
            y_list = []
            for i, items in enumerate(senti_dict_list):
                if topics in items and items[topics][0] != 0:
                    x_list.append(i)
                    y_list.append(items[topics][0])

            plt.plot(x_list, y_list, alpha=0.8)
        plt.show()

    draw_pic(d_list, senti_dict_list)

    print(d_list)

#for number in range(20):
#    a_list = [1]*len(range(number+1))
#    print(entropy(a_list))

#main()