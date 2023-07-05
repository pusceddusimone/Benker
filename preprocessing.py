import pandas as pd
import nltk
import string

from nltk import PorterStemmer
from nltk.corpus import stopwords


def slang_to_english(mess):
    """
Function which converts each slang term to their english part
    :param mess: message with slang to convert
    :return: message converted
    """
    mess = mess.split()
    slang_dict = {
        "lol": "laugh out loud",
        "omg": "oh my god",
        "btw": "by the way",
        "idk": "I don't know",
        "brb": "be right back",
        "afk": "away from keyboard",
        "thx": "thanks",
        "np": "no problem",
        "imo": "in my opinion",
        "fyi": "for your information",
        "gr8": "great",
        "b4": "before",
        "wanna": "want to",
        "gonna": "going to",
        "cuz": "because",
        "gimme": "give me",
        "gotta": "have got to",
        "kinda": "kind of",
        "sorta": "sort of",
        "ya": "you",
        "ya'll": "you all",
        "wassup": "what's up",
        "tho": "though",
        "sup": "what's up",
        "dunno": "don't know",
        "g2g": "got to go",
        "ttyl": "talk to you later",
        "lmao": "laughing my ass off",
        "rofl": "rolling on the floor laughing",
        "srsly": "seriously",
        "thnx": "thanks",
        "h8": "hate",
        "bff": "best friend forever",
        "fomo": "fear of missing out",
        "tbh": "to be honest",
        "af": "as f**k",
        "lit": "awesome",
        "on fleek": "perfect",
        "savage": "brutally honest",
        "thirsty": "desperate for attention",
        "woke": "aware of social issues",
        "bae": "before anyone else",
        "extra": "over the top",
        "flex": "show off",
        "ghost": "ignore someone",
        "hangry": "irritable when hungry",
        "lowkey": "secretly",
        "salty": "bitter or angry",
        "troll": "intentionally provoke others online",
        "yolo": "you only live once",
        "squad": "a close group of friends",
        "gig": "a job or performance",
        "chill": "relax or hang out",
        "dope": "cool or awesome",
        "swag": "style or confidence",
        "thicc": "curvy or voluptuous",
        "snack": "attractive person",
        "ship": "support or endorse a romantic relationship",
        "hella": "very or a lot",
        "sketchy": "suspicious or dubious",
        "GOAT": "Greatest of All Time",
        "FOMO": "Fear of Missing Out",
        "OOTD": "Outfit of the Day",
        "YOLO": "You Only Live Once",
        "ROFLMAO": "Rolling On the Floor Laughing My Ass Off"
    }
    for idx, word in enumerate(mess):
        if word in slang_dict:
            mess[idx] = slang_dict[word]
    return ' '.join(mess)


def remove_stopwords(mess):
    """
Function which removes punctuation and stopwords from the text
    :param mess: message with punctuation and stopwords
    :return: message without punctuation and stopwords
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])


def stemming(mess):
    """
Function which applies stemming to a message, PorterStemmer is used for this task
    :param mess: message to stem
    :return: message stemmed
    """
    return ' '.join([PorterStemmer().stem(word) for word in mess.split()])


def tokenize(mess):
    """
Function to tokenize a message
    :param mess: message to tokenize
    :return: message tokenized
    """
    return mess.split()


def preprocess(msg):
    """
Applies the removal of stopwords, stemming and tokenization to a message
    :param msg: message to preprocess
    :return: message preprocessed
    """
    msg = [msg]
    d = {'message': msg}
    df = pd.DataFrame(data=d)
    try:
        nltk.corpus.stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    df['message'] = df['message'].apply(remove_stopwords)
    df['message'] = df['message'].apply(stemming)
    df['message'] = df['message'].apply(tokenize)
    # df['message'] = df['message'].apply(slang_to_english)
    return df
