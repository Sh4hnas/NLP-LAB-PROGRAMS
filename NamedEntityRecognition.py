import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Input text
text= "Barack Obama was born in Hawaii. Elon Musk founded SpaceX."

# Tokenize and tag parts of speech
tokens= word_tokenize(text)
pos_tags= pos_tag(tokens)

# Perform named entity recognition
tree= ne_chunk(pos_tags)

# Pretty print the tree
print("\n Named Entity Recognition Tree:")
tree.pretty_print()
