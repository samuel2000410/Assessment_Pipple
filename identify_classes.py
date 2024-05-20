import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]
    return tokens

def identify_classes(df):
    # Combine the Title and Description columns
    df['combined_text'] = df['Title'] + " " + df['Description']

    # Tokenize and preprocess the combined text
    df['tokens'] = df['combined_text'].apply(preprocess_text)

    most_relevant_topics = []

    # Analyze topics per class
    for class_index, class_df in df.groupby('Class_Index'):

        # Create a dictionary and corpus for LDA
        dictionary = Dictionary(class_df['tokens'])
        corpus = [dictionary.doc2bow(text) for text in class_df['tokens']]

        # Initialize and fit LDA model
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=1)

        # Get the dominant topic for the class
        topic_distribution = [lda_model.get_document_topics(dictionary.doc2bow(text)) for text in class_df['tokens']]
        topic_counts = pd.Series([max(dist, key=lambda tup: tup[1])[0] for dist in topic_distribution]).value_counts()
        dominant_topic = topic_counts.idxmax()
        
        most_relevant_topics.append((class_index, dominant_topic))

        # Review topic representations
        print(f"Class {class_index} Topic Representations:")
        for idx, topic in lda_model.print_topics(num_topics=1, num_words=10):
            print(f"Topic {idx}: {topic}")