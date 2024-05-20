from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def bow(df):
    count_vectorizer = CountVectorizer(max_features=1000)  

    # Create BoW matrix
    X_bow = count_vectorizer.fit_transform(df['processed_text'])

    # Create a DataFrame with BoW features and 'Class_Index'
    df_bow = pd.DataFrame(X_bow.toarray(), columns=count_vectorizer.get_feature_names_out())
    df_bow['Class_Index'] = df['Class_Index']

    return df_bow

def ngrams(df):
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)  # You can adjust max_features as needed

    # Create a matrix of unigrams and bigrams
    X_ngrams = ngram_vectorizer.fit_transform(df['processed_text'])

    # Create a DataFrame with N-gram features and 'Class_Index'
    df_ngrams = pd.DataFrame(X_ngrams.toarray(), columns=ngram_vectorizer.get_feature_names_out())
    df_ngrams['Class_Index'] = df['Class_Index']

    return df_ngrams

def tfidf(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)

    # Create a TF-IDF matrix
    X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])

    # Create a DataFrame with TF-IDF features and 'Class_Index'
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    df_tfidf['Class_Index'] = df['Class_Index']

    return df_tfidf