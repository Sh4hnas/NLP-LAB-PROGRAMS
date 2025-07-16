from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_bow_and_compute_similarity(documents):
    # Create a CountVectorizer object to convert text documents to a matrix of token counts
    vectorizer= CountVectorizer()
    
    # Fit the model and transform the documents into a bag-of-words representation
    bow_matrix= vectorizer.fit_transform(documents)
    
    # Compute the cosine similarity matrix from the bag-of-words matrix
    similarity_matrix= cosine_similarity(bow_matrix)
    
    return bow_matrix,similarity_matrix

documents= [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

bow_matrix, similarity_matrix= create_bow_and_compute_similarity(documents)
print("Bag of Words Matrix:",bow_matrix.toarray())
print("Cosine Similarity Matrix:", similarity_matrix)