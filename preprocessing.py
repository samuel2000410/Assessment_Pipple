import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer and stop words list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocessing(df):
    # Combine 'Title' and 'Description' into a single text field
    df['text'] = df['Title'] + ' ' + df['Description']

    # Apply preprocessing to the text column
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df[['Class_Index', 'processed_text']]

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Apply stemming
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Join the tokens back into a single string
    return ' '.join(stemmed_tokens)