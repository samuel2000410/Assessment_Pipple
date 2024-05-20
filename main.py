import clean_data
import identify_classes
import data_description
import preprocessing
import feature_selection
import classification

def main():
    # Clean the data
    df = clean_data.clean_data('data/data.csv')

    # Get some information about data and describe it
    data_description.data_description(df)

    # Identify topics in the 4 classes
    identify_classes.identify_classes(df)

    # Preprocessing the data
    df = preprocessing.preprocessing(df)

    # Form feature space with tfidf
    df_tfidf = feature_selection.bow(df)
    
    # Apply machine learning to classify
    techniques = ['KNN', 'DecisionTree', 'NaiveBayes', 'NeuralNetwork', 'LR', 'RF']
    for technique in techniques:
        accuracy = classification.classification(df_tfidf, technique)
        print(f"{technique} Accuracy: {accuracy}")
    
if __name__ == "__main__":
    main()