import os
import re
import pandas as pd
import numpy as np

path = "sample_data"
N = 26


def read_corpus():
    _data = ""
    for _filename in filter(lambda p: p.endswith("txt"), os.listdir(path)):
        _filepath = os.path.join(path, _filename)
        with open(_filepath, mode='r') as f:
            _data += clean(f.read())
    return _data


def clean(text: str) -> str:
    return re.sub(r'[^a-zA-Z]', '', text).lower()


def get_columns():
    _columns = []
    for i in range(N):
        _columns.append(chr(ord("a") + i))
    return _columns


def get_matrix() -> pd.DataFrame:
    df = {i: {} for i in columns}
    for i in range(1, len(data_cleaned)):
        df[data_cleaned[i - 1]][data_cleaned[i]] = 1 + df[data_cleaned[i - 1]].get(data_cleaned[i], 0)
    matrix_data = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            matrix_data[i][j] = df[columns[i]].get(columns[j], 0)
    return pd.DataFrame(matrix_data, columns=columns, index=columns)


if __name__ == '__main__':
    data_cleaned = read_corpus()
    print("Data length: %d" % len(data_cleaned))
    columns = get_columns()
    matrix = get_matrix()
    #print(matrix.to_string())
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    #print(matrix.to_string())
    matrix["C"] = 1 + (matrix * np.log2(matrix.replace(0, np.nan))).sum(axis=1) / np.log2(N)
    print(matrix.to_string())