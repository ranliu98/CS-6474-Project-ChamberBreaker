import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import Counter

'''
path = "/Users/ranliu/Desktop/Class-related/CS6474/PJ/mid/CMV_mother.csv"
df = pd.read_csv(path)

print(df.shape)
df = df.dropna(subset=['title'])
print("without empty title:",df.shape)

title_c = df["title"].tolist()
title_text = " ".join(title_c)
title_text = title_text.lower()
'''
#print("***")
#print("title_text" , title_text)
#print("***")

def countfreq(text):

    stopword = stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer() # this may be problematic
    snowball_stemmer = SnowballStemmer('english')

    word_tokens = nltk.word_tokenize(text)

    # print("step1", )
    word_tokens = [word for word in word_tokens if word not in stopword] #removing_stopwords
    word_tokens = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens] #lemmatized_word
    word_tokens = [snowball_stemmer.stem(word) for word in word_tokens] #stemmed_word
    freq = FreqDist(word_tokens)
    return freq

def keep_text(df, text):
    if text in str(df):
        return df
    else: return None

def df_title_word(df, word):
    df["title"] = df["title"].apply(str.lower)
    df["title"] = df["title"].apply(lambda x: keep_text(x, word))
    df = df.dropna(subset=['title'])
    return df

def draw_wc(title_text):
    wordcloud = WordCloud(max_font_size=40).generate(title_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()



def remove(text):
    #remove cmv and punc
    text = str(text)
    text = text.replace("cmv", "")
    text = text.replace("change my view", "")

    word_tokens = nltk.word_tokenize(text)
    stop_words = stopwords.words('english')
    new_text = [word for word in word_tokens if word not in stop_words]
    return "".join(new_text)

def task1():
    new_title_text = remove(title_text)
    print(new_title_text)
    new_title_text = TextBlob(new_title_text)

    noun_list = new_title_text.noun_phrases

    freq = countfreq("".join(noun_list))
    print("type(freq)", type(freq))

    print(freq.most_common(100))

#print("freq" , freq)




    df_gun = df_title_word(df, "gun")
    print("df_gun", df_gun.shape)

    for items in df_gun["selftext"].tolist():
        print("---")
        print(items)

    draw_wc("".join(noun_list))


#sentiment_list = [{'combustion vehicles': 0.0, 'transportation': 0.4000000059604645, 'least': -0.30000001192092896, 'way': 0.20000000298023224, 'lot': 0.0, 'pollution': -0.4000000059604645, 'fuel': 0.0, 'vehicles': -0.6000000238418579, 'everybody': 0.4000000059604645, 'alternative': 0.8999999761581421, 'problem': -0.699999988079071, 'humans': 0.0, 'vehicle': 0.699999988079071, 'solution': 0.30000001192092896, '5G communications': 0.0, 'another': 0.0, 'idea': 0.20000000298023224, 'system': 0.5, 'types': 0.0, 'detectors': 0.20000000298023224, 'processing power': 0.800000011920929, 'situations': -0.8999999761581421, 'dashboard display': 0.10000000149011612, 'app': 0.10000000149011612, 'pilot': 0.0, 'comms-abilities': 0.30000001192092896, 'ways': 0.10000000149011612, 'myriad': -0.20000000298023224, 'law': 0.0, 'health': -0.800000011920929, 'points': -0.4000000059604645, 'distribution': -0.20000000298023224, 'price': -0.5, 'obstacles': -0.699999988079071, 'problems': -0.8999999761581421, 'mass': -0.30000001192092896, 'second-hand market': -0.6000000238418579, 'usage': -0.5, 'maintenance': -0.30000001192092896, 'thanks': 0.4000000059604645, 'things': -0.5, 'traffic': 0.0, 'safety': -0.10000000149011612, 'impact': 0.0, 'traffic deaths': -0.4000000059604645, 'CMV': 0.10000000149011612}, {'delta(s)': 0.0, 'comments': 0.10000000149011612, '/u/gab_rod': 0.10000000149011612, 'OP': 0.10000000149011612, 'post': 0.0, 'deltas': 0.4000000059604645, 'conversation': 0.0, 'DeltaLog': 0.10000000149011612, 'users': 0.10000000149011612, 'Deltaboards': 0.0, 'Delta System Explained': 0.0, 'https://www.reddit.com/r/changemyview/wiki/deltasystem': 0.0, 'reversal': 0.10000000149011612, 'view': 0.20000000298023224, 'change': 0.0}, {'communication systems': 0.10000000149011612, 'vehicles': 0.0, 'lot': 0.10000000149011612, 'Latency': -0.699999988079071, 'research': 0.0, 'collisions': 0.10000000149011612, 'information': 0.0, 'danger': -0.8999999761581421, 'communication': 0.10000000149011612, 'system': 0.0, 'reaction': 0.10000000149011612, 'confidence': 0.8999999761581421, 'packet loss': -0.8999999761581421, 'problem': -0.6000000238418579, 'vehicle': -0.30000001192092896, 'sensor data': 0.0, 'degree': 0.6000000238418579, 'wireless communications': -0.10000000149011612, 'updates': -0.10000000149011612, 'latency': 0.10000000149011612, 'cloud computations': 0.0, 'decision': 0.10000000149011612, 'cloud': 0.0}, {'angle': 0.0, 'Self driving': -0.10000000149011612, 'car': -0.8999999761581421, 'network': 0.0, 'fail point': -0.800000011920929, 'point': 0.0, 'networks': 0.0, 'self driving cars': 0.0, 'liability': -0.6000000238418579, 'software': 0.0, 'refinement': 0.20000000298023224, 'automation talk': 0.10000000149011612, 'Starts': 0.0, 'process': 0.4000000059604645, 'https://youtu.be/Ucp0TTmvqOE': 0.0, 'Tesla': 0.0}, {'hurdle': 0.0, 'position': -0.20000000298023224, 'adoption': -0.10000000149011612, 'driving cars': 0.20000000298023224}, {'obstacles': -0.699999988079071, 'deer': -0.6000000238418579, 'road': -0.20000000298023224, 'street': -0.10000000149011612, 'kid': -0.4000000059604645, 'pedestrian': -0.4000000059604645, 'cyclist': 0.0, '5G': -0.30000001192092896, 'ball': 0.0, 'cones': -0.699999988079071, 'lines': -0.6000000238418579, 'construction zone': -0.800000011920929}, {'people': -0.20000000298023224, 'alternative': 0.0, 'millions': -0.30000001192092896, 'technology': -0.10000000149011612, 'People': -0.30000001192092896, 'lives': -0.20000000298023224, '_today_': -0.20000000298023224, 'future': -0.10000000149011612, 'roadways': 0.4000000059604645, 'environment': -0.8999999761581421, 'Fancier cars': -0.4000000059604645, 'solution': -0.10000000149011612, 'Americans': 0.0, 'earth': -0.800000011920929, 'midst': -0.6000000238418579, 'family': -0.5, 'friends': -0.4000000059604645, 'resources': -0.800000011920929}, \
#                  {'issues': 0.10000000149011612, 'vehicles': 0.10000000149011612, 'approach': 0.0, 'all': -0.800000011920929, 'problems': -0.800000011920929, 'people': 0.0, 'transport': 0.30000001192092896, 'ownership': 0.6000000238418579, 'cars': 0.20000000298023224, 'cycling': 0.0, 'city infrastructure': 0.10000000149011612, 'mind': 0.0, 'improvement': 0.0, 'B': 0.10000000149011612, 'timetable': -0.10000000149011612, 'walking': 0.0, 'effort': -0.30000001192092896, 'qualities': 0.20000000298023224, 'ambulances': 0.10000000149011612, 'fire engines': 0.0, 'electric': 0.0, 'https://betseybuckheit.com/wp-content/uploads/2013/09/Munster1991BusBicycleCar.jpg': 0.10000000149011612, 'sweat': 0.0, 'traffic': 0.0, 'way': 0.20000000298023224, 'cities': 0.0, 'car ownership': 0.10000000149011612, 'parking': 0.0, 'expenses': 0.20000000298023224, 'transportation': 0.20000000298023224, 'assist bikes': 0.20000000298023224, 'changes': -0.20000000298023224}, {'Cars': -0.10000000149011612, 'grid system': 0.0, 'vehicles': 0.20000000298023224, 'market': 0.30000001192092896, 'systems': 0.10000000149011612, 'signaling system': 0.0, 'vision': 0.0, 'cars': 0.20000000298023224, 'People': 0.0, 'radar': 0.4000000059604645, 'reason': 0.10000000149011612, 'safety reasons': 0.0, 'limit': 0.0, 'cyber security concerns': -0.6000000238418579, 'approach': 0.699999988079071, 'anything': 0.4000000059604645, 'benefit': 0.0, 'attack surface': 0.0, 'department': 0.0, 'progress': 0.0, 'development': 0.30000001192092896, 'network access': 0.0, 'TBH': 0.0, 'intent': 0.20000000298023224, 'intersections': 0.0, 'lane': 0.10000000149011612, 'changes': -0.20000000298023224, 'driving systems': 0.0, 'air': -0.10000000149011612, 'components': -0.30000001192092896, 'security risks': -0.30000001192092896}, {'cars': 0.10000000149011612, 'techonologies': -0.699999988079071, 'hype': -0.10000000149011612, 'best': 0.800000011920929, 'some': -0.4000000059604645, 'assumptions': -0.30000001192092896, 'blanket': -0.10000000149011612, 'cell phone companies': 0.0, 'technology': 0.0, '5G': -0.4000000059604645, 'everyone': 0.10000000149011612, 'vehicles': 0.0, 'Coal plants': 0.0, 'energy production': 0.0, 'Adoption': 0.0, 'speeds': 0.0, 'majority': 0.0, 'future': 0.0, 'incentives': -0.4000000059604645, 'price': 0.0, 'road': 0.10000000149011612, 'collisions': -0.800000011920929, 'car communications': -0.5, 'world': 0.0, 'fleet': 0.0, 'transportation': 0.10000000149011612, 'US': 0.0, 'gas': 0.0, 'Safety': -0.30000001192092896, 'equilibrium': 0.0, 'people': 0.0, 'demand': -0.20000000298023224, 'selling point': 0.30000001192092896, 'safety features': 0.6000000238418579, 'G. Lab': -0.10000000149011612, 'crumple zones': 0.0, 'pendulum swing': 0.0, 'lives': 0.20000000298023224, 'market': 0.0, 'gas vehicles': 0.0, 'prices': -0.5, 'maintenance cost': 0.0}, {'5G': 0.0, '4G our': -0.20000000298023224, 'cars': -0.10000000149011612, '6G': 0.0, '7G': 0.0, '8G': 0.0, 'busses': 0.0, 'carts': -0.10000000149011612, 'horses': -0.20000000298023224, 'buzzword technologies': 0.0, 'sentence': 0.0, 'breakthrough idea': 0.10000000149011612, 'WWIII': 0.0}]

def give_stemmed_dict(sentiment_list):
    # dict was stemmed
    snowball_stemmer = SnowballStemmer('english')
    new_list = []
    for items in sentiment_list:
        new_items = {}
        for keys in items:
            new_keys = snowball_stemmer.stem(keys)
            new_items.update({new_keys:items[keys]})
        new_list.append(new_items)
    return new_list
#new_list = give_stemmed_dict(sentiment_list) # looks ok

def count_senti(text, sentiment_list, stem):
    # if stem== True, sentiment_list need to be stemmed before

    def get_good_sentiment_dict(sentiment_list):
        def addlist(sentiment_list):
            # return one simple dictionary of sentiments
            result = {}
            for dic in sentiment_list:

                for key in dic:
                    if key not in result:
                        result[key] = dic[key]
                    else:
                        result[key] += dic[key]
            return result
        '''
        def addlist(sentiment_list):
            # return one simple dictionary of sentiments
            X_new = Counter()
            for items in sentiment_list:
                X = Counter(items)
                X_new = X_new + X
            z = dict(X_new)
            return z
        '''
        add_sentiment_dict = addlist(sentiment_list) # assume this is good

        def count_key(sentiment_list):
            X = []
            for items in sentiment_list:
                X.extend(list(items.keys()))
            return dict(Counter(X))
        count_dict = count_key(sentiment_list)

        count_text = " ".join(list(count_dict.keys()))
        # draw_wc(count_text)

        sentiment_dict = {}
        for items in add_sentiment_dict:
            sentiment_dict.update({items: add_sentiment_dict[items]/count_dict[items]})
        return sentiment_dict

#    sentiment_dict = get_good_sentiment_dict(sentiment_list)
    sentiment_dict = sentiment_list

    overall_senti = 0

    '''
    word_tokens = nltk.word_tokenize(text)
    if stem == True:
        snowball_stemmer = SnowballStemmer('english')
        word_tokens = [snowball_stemmer.stem(word) for word in word_tokens]

    word_count_list = []
    for words in word_tokens:
        if words in sentiment_dict:
            word_count_list.append(words)
    word_count_dict = Counter(word_count_list)
    ''' # word_tokens cannot solve multi-words problem


    result_dict = {}
    for keys in sentiment_dict:
        keys_count = text.count(keys, 0, len(text))
        if keys_count == 0:
            print("---"*20)
            print("freq==0!!!")
        result_dict.update({keys:[sentiment_dict[keys],keys_count]})

#    for word in word_tokens:
#        if word in sentiment_dict:
#            overall_senti = overall_senti + sentiment_dict[word]
    print(result_dict)
    return result_dict


#print(count_senti('transportation app issues',sentiment_list))
#print(count_senti('transportation app issues' , new_list, True))

