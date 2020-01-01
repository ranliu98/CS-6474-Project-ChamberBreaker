# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

def google_sen(text):

# Instantiates a client
    client = language.LanguageServiceClient()

# The text to analyze
# text = u'Hello, world!'
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)
# Detects the sentiment of the text
    response = client.analyze_entity_sentiment(document=document)
    entities = response.entities
    result = {}
    for e in entities:
        result[e.name] = e.sentiment.score

    return result
