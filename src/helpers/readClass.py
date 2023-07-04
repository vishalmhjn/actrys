import pandas as pd
import matplotlib.pyplot as plt


class Paris_Counts:
    def __init__(self, path):
        self.path = path  # complete path to the file

    def read_file(self, day, time_col="t_1h"):
        """read one day data to reduce memory overload e.g. 2019-05-01"""
        df_sample = pd.read_csv(self.path, sep=";", nrows=10)
        df_sample_size = df_sample.memory_usage(index=True).sum()
        my_chunk = (1000000000 / df_sample_size) / 10
        my_chunk = int(my_chunk // 1)
        # create the iterator
        iter_csv = pd.read_csv(self.path, iterator=True, sep=";", chunksize=my_chunk)
        df_result = pd.concat(
            [chunk[chunk[time_col].str.contains(day)] for chunk in iter_csv]
        )
        return df_result

    def read_uber_speed(self):
        """read one day data to reduce memory overload e.g. 2019-05-01"""
        df_sample = pd.read_csv(self.path, nrows=10)
        df_sample_size = df_sample.memory_usage(index=True).sum()
        my_chunk = (1000000000 / df_sample_size) / 10
        my_chunk = int(my_chunk // 1)
        # create the iterator
        iter_csv = pd.read_csv(self.path, iterator=True, chunksize=my_chunk)

        df_result = pd.concat([chunk for chunk in iter_csv])

        return df_result
