import os
import pickle
import numpy as np

data_folder = r'C:\Kaggle\RainForest_R0\Datasets\Mel_224_512_1'
file_name = 'Mel_224_512_1_testing'
data_file = os.path.join(data_folder, file_name + '.pickle')

# with open(data_file, 'rb') as input_file:
#     data, mel_parameters = pickle.load(input_file)
#
# data_shape1 = data.X.shape[1]
#
# del data, mel_parameters
# b = np.linspace(0, 1991, 1992)
num_splits = 12
N = 1992 # number of test files
seg_size = N // num_splits
a = np.arange(0, N, N // num_splits)
idx_ranges = []
for ii in range(num_splits):
    if ii == 0:
        idx_start = ii
        idx_end = idx_start + seg_size
    else:
        idx_start = idx_ranges[-1][1]
        idx_end = idx_start + seg_size
    idx_ranges.append([idx_start, idx_end])


# for i, idx in enumerate(a):
#     if i == 0:
#         idx_window = [0, N // num_splits]
#     if (i > 0) and (i < len(a) - 1):
#         # idx_window = [idx - (N // num_splits) + 1, idx]
#         idx_window = [idx + 1, idx + (N // num_splits)]
#     if i == len(a) - 1:
#         idx_window = [idx + 1, N]
#     idx_ranges.append(idx_window)


# c = []
for i, idx_range in enumerate(idx_ranges):
    # c.append(b[idx_range[0]:idx_range[1]])
    with open(data_file, 'rb') as input_file:
        data, mel_parameters = pickle.load(input_file)

    X_split = data.X[idx_range[0]:idx_range[1]]
    data.X = X_split
    data.mel_parameters = mel_parameters
    data.idx_range = idx_range
    data.idx_upper_limit = N
    data.idx_ranges = idx_ranges
    data.segment = [i, len(idx_ranges)]

    ds_name = file_name + '_' + str(i + 1) + 'of' + str(len(idx_ranges))

    with open(os.path.join(data_folder, ds_name + '.pickle'), 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

    del data, X_split, ds_name

print('Check Point')
