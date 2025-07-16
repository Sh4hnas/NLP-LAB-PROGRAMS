import nltk
from nltk.text import Text
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Read and preprocess the text
text="hi shahnas what are you doing shahnas is everything fine"
text= ''.join(char for char in text if char not in string.punctuation)
tokens= nltk.word_tokenize(text)
stopwords= set(nltk.corpus.stopwords.words('english'))
words_no_stopwords= [word for word in tokens if word not in stopwords]

# Stemming and Lemmatization
stemmer= PorterStemmer()
lemmatizer= WordNetLemmatizer()
stemmed_words= [stemmer.stem(word) for word in words_no_stopwords]
lemmatized_words= [lemmatizer.lemmatize(word) for word in words_no_stopwords]  

# Concordance
obj= Text(words_no_stopwords)
target= input("\nEnter the word to search: ").lower()
print(f"\nConcordance for '{target}':\n")
try:
    obj.concordance(target)
except:
    print(f"'{target}' not found in the text.")
