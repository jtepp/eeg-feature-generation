import numpy as np
from src.EEG_feature_extraction import generate_feature_vectors_from_samples
# from EEG_feature_extraction import matrix_from_csv_file
import pandas as pd

def gen_testing_matrix(input_matrix):    
    vectors, header = generate_feature_vectors_from_samples(file_path=None,#'python/eeg-feature-generation/dataset/test/subjectc-concentrating-2.csv',
                                                            nsamples=200,
                                                            period=1.,
                                                            state=0,
                                                            remove_redundant=True,
                                                            cols_to_ignore=-1,
                                                            mtx=input_matrix)

    df = pd.DataFrame(data=vectors, columns=header)
    return df


# if __name__ == '__main__':
#     df = matrix_from_csv_file('python/eeg-feature-generation/demo_file.csv')
#     # df = pd.json_normalize(array)
#     # df = np.array(df)
#     matrix = gen_testing_matrix(df)#.drop('Label', axis=1) 
#     print(matrix)

