import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  
import h5py

input_csv = 'star_1M_balanced.csv'
output_h5 = 'star_01_normalized.h5'
chunk_size = 50000
num_features = 3121

scaler = MinMaxScaler()

print("1")
for chunk in pd.read_csv(input_csv, header=None, chunksize=chunk_size):

    scaler.partial_fit(chunk.iloc[:, :-1].values.astype(np.float32))


print("2")
with h5py.File(output_h5, 'w') as f:

    dset_X = f.create_dataset('X', (1200000, num_features), dtype='float32')
    dset_y = f.create_dataset('y', (1200000,), dtype='int64')
    
    ptr = 0
    for chunk in pd.read_csv(input_csv, header=None, chunksize=chunk_size):
        X = chunk.iloc[:, :-1].values.astype(np.float32)
        y = chunk.iloc[:, -1].values.astype(np.int64)
        
        num_rows = len(X)
        dset_X[ptr : ptr + num_rows] = scaler.transform(X)
        dset_y[ptr : ptr + num_rows] = y
        ptr += num_rows
        print(f"finished: {ptr}/1200000")

