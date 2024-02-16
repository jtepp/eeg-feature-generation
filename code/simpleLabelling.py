import os
import pandas as pd

def add_label_to_data(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Determine the label based on the filename
            if 'relaxing' in filename:
                label = 0
            elif 'concentrating' in filename:
                label = 1
            else:
                continue

            # Load the data
            df = pd.read_csv(os.path.join(directory, filename))

            # Add the label column
            df['Label'] = label

            # Save the modified data
            df.to_csv(os.path.join(directory, filename), index=False)

add_label_to_data('dataset/our_data_labelled')