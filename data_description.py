import matplotlib.pyplot as plt
import seaborn as sns

def data_description(df):
    # Basic information
    print("Basic Information:")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Number of rows: {df.shape[0]}")
    print("\nColumns:")
    print(df.columns.tolist())
    
    # Class distribution
    if 'Class_Index' in df.columns:
        class_counts = df['Class_Index'].value_counts()
        print("Class Distribution:")
        print(class_counts)
    
    # Missing values
    print("Missing Values:")
    missing_values = df.isnull().sum()
    print(missing_values)