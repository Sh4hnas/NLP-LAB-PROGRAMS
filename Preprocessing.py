import re
import string
import nltk
import emoji
import contractions 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def expand_contractions(tweet):
    return contractions.fix(tweet)

def preprocess_tweet(tweet):
    tweet_lower = tweet.lower()
    print(f"After Lowercase: {tweet_lower}")
    
    sentence=sent_tokenize(tweet)
    print(f"After Sentence Tokenization: {sentence}")
    
    tweet_no_urls = re.sub(r'http\S+|www\S+|https\S+', '', tweet_lower)
    print(f"After URL Removal: {tweet_no_urls}")
   
    tweet_no_emojis = emoji.replace_emoji(tweet_no_urls, replace='')
    print(f"After Emoji Removal: {tweet_no_emojis}")
    
    tweet_no_contractions = expand_contractions(tweet_no_emojis)
    print(f"After Expanding Contractions: {tweet_no_contractions}")

    tweet_no_punctuation = tweet_no_contractions.translate(str.maketrans('', '', string.punctuation))
    print(f"After Punctuation Removal: {tweet_no_punctuation}")

    tokens = word_tokenize(tweet_no_punctuation)
    print(f"After  Word Tokenization: {tokens}")

    tokens_no_stopwords = [word for word in tokens if word not in stop_words]
    print(f"After Stopword Removal: {' '.join(tokens_no_stopwords)}")

    tokens_stemmed = [stemmer.stem(word) for word in tokens_no_stopwords]
    print(f"After Stemming: {' '.join(tokens_stemmed)}")

    tokens_lemmatized = [lemmatizer.lemmatize(word) for word in tokens_no_stopwords]
    print(f"After Lemmatization: {' '.join(tokens_lemmatized)}")

    cleaned_tweet = ' '.join(tokens_lemmatized)
    print(f"Final Processed Tweet: {cleaned_tweet}")
    print("-----")
    
    return cleaned_tweet

def read_tweets_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    return [tweet.strip() for tweet in tweets]

file_path = 'exp2.txt' 
tweets = read_tweets_from_file(file_path)

cleaned_tweets = [preprocess_tweet(tweet) for tweet in tweets]

for i, tweet in enumerate(cleaned_tweets):
    print(f"Original Tweet: {tweets[i]}")
    print(f"Processed Tweet: {tweet}")
  


