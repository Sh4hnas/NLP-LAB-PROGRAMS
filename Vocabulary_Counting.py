import nltk
from nltk.tokenize import word_tokenize

# Download 'punkt_tab' data
nltk.download('punkt_tab')

text= "This is a sample text. This text is for testing the vocabulary counting. The word text is repeated."

tokens= word_tokenize(text.lower())
total_tokens= len(tokens)
unique_tokens= len(set(tokens))

target_word= "text"
target_count= tokens.count(target_word)
percentage= (target_count / total_tokens) * 100

print("Total tokens:", total_tokens)
print("Unique tokens:", unique_tokens)
print(f"'{target_word}' count:", target_count)
print(f"'{target_word}' percentage: {round(percentage, 2)}%")