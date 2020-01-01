# activate before use

from collections import Counter
from google.cloud.language import types
from google.cloud import language_v1
from google.cloud.language_v1 import enums

def count_centi(text, sentiment_list, stem):
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

def google_sen(text):
    client = language_v1.LanguageServiceClient()
    result = {}
    try:
        document = types.Document(
            content=text,
            type=enums.Document.Type.PLAIN_TEXT)

        response = client.analyze_entity_sentiment(document=document)
        for e in response.entities:
            result[e.name] = e.sentiment.score
    except:
        print("google_sen Error here")

    return result

def google_sen_new(text_content):
    """
    Analyzing Entity Sentiment in a String
    Args:
      text_content The text content to analyze
    """
    # text_content = 'Grapes are good. Bananas are bad.' Available types: PLAIN_TEXT, HTML

    client = language_v1.LanguageServiceClient()
    type_ = enums.Document.Type.PLAIN_TEXT
    language = "en"
    document = {"content": text_content, "type": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = enums.EncodingType.UTF8
    response = client.analyze_entity_sentiment(document, encoding_type=encoding_type)
    result_dict = {}  # "entity":[]
    for entity in response.entities:
        result_list = []
        result_list.append(entity.name)  # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
        result_list.append(enums.Entity.Type(
            entity.type).name)  # Get the salience score associated with the entity in the [0, 1.0] range
        result_list.append(
            entity.salience)  # Get the aggregate sentiment expressed for this entity in the provided document.

        sentiment = entity.sentiment
        result_list.append(sentiment.score)
        result_list.append(sentiment.magnitude)

        result_dict[entity] = result_list

    return result_dict
# not using google_sen_new for now

def main_senti(text):
    if len(text) >= 3:
        sentiment_result = google_sen(text)
        #new_list = wordcloud_etc.give_stemmed_dict([sentiment_result])
        return count_senti(text, sentiment_result, False)
    else: return {}

