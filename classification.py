from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text


def classification(df, technique):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Initialize classifier based on technique
    if technique == 'KNN':
        classifier = KNeighborsClassifier()
    elif technique == 'DecisionTree':
        classifier = DecisionTreeClassifier()
    elif technique == 'NaiveBayes':
        classifier = GaussianNB()
    elif technique == 'NeuralNetwork':
        classifier = MLPClassifier()
    elif technique == 'SVM':
        classifier = SVC()
    elif technique == 'LR':
        classifier = LogisticRegression()
    elif technique == 'RF':
        classifier = RandomForestClassifier()
    else:
        raise ValueError("Invalid technique specified.")

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Predict on test data
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Extracting an individual tree from the forest
    individual_tree = classifier.estimators_[0]

    # Exporting the tree as text
    tree_rules = export_text(individual_tree, feature_names=list(X_train.columns))
    print(tree_rules)

    return accuracy

def split_data(df):
    # Split the data into features (X) and target (y)
    X = df.drop(columns=['Class_Index'])  
    y = df['Class_Index']  

    return train_test_split(X, y, test_size=0.2, random_state=42)