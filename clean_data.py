import pandas as pd
import re 

def clean_data(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()[1:]

    data = []
    for line in lines:
        # Remove unwanted characters
        line = re.sub(r'[;;"\']+', '', line)

        # Split into columns by commas
        columns = line.split(',')

        # Create data; Join all columns after the second one with commas
        class_index = columns[0]
        title = columns[1]
        description = ','.join(columns[2:])  
        data.append([class_index, title, description])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Class_Index', 'Title', 'Description'])

    return df