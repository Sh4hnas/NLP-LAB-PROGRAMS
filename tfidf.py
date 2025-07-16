from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents= [
    "the sky is blue",
    "the sun is bright",
    "the sun in the sky is bright",
    "we can see the shining sun, the bright sun"
]

# Create TF-IDF Vectorizer
vectorizer= TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix= vectorizer.fit_transform(documents)

# Get feature names (i.e., unique words)
feature_names= vectorizer.get_feature_names_out()

# Convert the result to a DataFrame for better readability
df= pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Print TF-IDF values
print(df)
