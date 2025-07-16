import nltk
from nltk.tag import hmm
from nltk.corpus import treebank
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('treebank')

# Train HMM POS tagger
train_data= treebank.tagged_sents()
trainer= hmm.HiddenMarkovModelTrainer()
tagger= trainer.train_supervised(train_data)

# Get file path from user
file_path= input("Enter the path to your input file: ")

try:
    with open(file_path, 'r') as file:
        sentences= file.readlines()
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    sentences= []  # Ensures no NameError if file not found

# Process each sentence
if not sentences:
    print("No sentences to process.")
else:
    for i, sentence in enumerate(sentences, start=1):
        sentence= sentence.strip()
        if sentence:
            print(f"\nSentence {i}: {sentence}")
            tokens= word_tokenize(sentence)
            tagged= tagger.tag(tokens)
            for word, tag in tagged:
                print(f"{word}\t{tag}")
