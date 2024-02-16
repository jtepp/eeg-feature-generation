import os
import pandas as pd

def process_file(input_filename):
    # Determine the output filename based on the input filename
    prefix, number = input_filename.split('/')[-1].split('_')
    prefix = prefix.lower()
    number = number.split('.')[0]  # remove .txt extension
    # output_filename = f'../our_data/schaffer-{prefix}-{number}.csv'
    output_filename = f'schaffer-{prefix}-{number}.csv'

    # Read the file into a DataFrame, skipping the first 4 lines
    df = pd.read_csv(input_filename, skiprows=4, header=None)

    # Select only the columns with indices 1, 2, 3, 4, and 13
    df = df[[13, 1, 2, 3, 4]]

    # Write the DataFrame to the output file
    df.to_csv(output_filename, index=False, header=False)

# Example usage:
process_file('Relaxing_1.txt')
    

# for filename in os.listdir('.'):
#     if filename.endswith('.txt'):
#         process_file(filename)