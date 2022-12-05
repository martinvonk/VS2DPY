import os
import re
from io import BytesIO
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


class var_out:
    def __init__(
        self,
        path: str = "variables.out",
    ):
        self.path = path

        self.size = os.stat(path).st_size

        time_pointers, byte_diffs, shape = self.scan_file()
        self.time_pointers = time_pointers
        self.byte_diffs = byte_diffs
        self.shape = shape

        self.times = self.get_times()

    def scan_file(
        self,
    ):
        byte_count = []
        end_count = []
        time_count = []
        lines = None
        with open(self.path, "r+") as fo:
            line = fo.readline()
            while line:
                byte_count.append(fo.tell())
                if "TIME = " in line:
                    time_count.append(byte_count[-2])
                    end_count.append(byte_count[-1])
                    if len(time_count) == 2:
                        break
                    else:
                        lines = []
                if lines is not None:
                    lines.append(line)
                line = fo.readline()
            if len(time_count) == 1:
                time_count.append(self.size)
            byte_diff0 = (
                time_count[1] - time_count[0]
            )  # bytes of one full matrix including TIME = line
            byte_diff1 = end_count[0] - time_count[0]  # bytes of TIME = line
            byte_diff2 = byte_diff0 - byte_diff1  # bytes of one full matrix

        times_pointer = np.arange(time_count[0], self.size, byte_diff0)
        byte_diffs = [byte_diff0, byte_diff1, byte_diff2]
        shape = np.genfromtxt(lines[1:]).shape
        return times_pointer, byte_diffs, shape

    def get_times(
        self,
    ):
        times = []
        with open(self.path, "rb") as fo:
            for t in self.time_pointers:
                fo.seek(t)
                times.append(float(fo.read(self.byte_diffs[1]).split()[-2]))
        return times

    def get_data(self, timesteps: list[int] = None, times: list[float] = None):
        if not isinstance(timesteps, list):
            timesteps = [timesteps]

        if times is not None:
            if not isinstance(times, list):
                times = [times]
            timesteps = [self.times.index(x) for x in times]

        tps = [self.time_pointers[x] for x in timesteps]
        datas = np.empty((len(tps), self.shape[0], self.shape[1]))
        with open(self.path, "rb") as fo:
            for i, tp in enumerate(tps):
                fo.seek(tp + self.byte_diffs[1])
                datas[i] = np.genfromtxt(BytesIO(fo.read(self.byte_diffs[2])))
        return datas

    def get_all_data(
        self,
    ):
        return self.get_data(np.arange(len(self.time_pointers)).tolist())

    def plot(self, timesteps: list[int] = None, times: list[float] = None):

        if times is None:
            if timesteps is not None:
                times = [self.times[x] for x in timesteps]
            else:
                times = self.times

        datas = self.get_data(times=times)[:, 1:-1, 1:-1]
        for t, data in zip(times, datas):
            _, ax = plt.subplots(figsize=(self.shape[1] / 10, self.shape[0] / 10))
            ax.pcolormesh(data, vmin=np.min(datas), vmax=np.max(datas))
            ax.set_title(f"{t=}")


class bal_out:
    def __init__(self, path: str = "balance.out"):

        self.path = path

        with open(path, "r") as fo:
            header = []
            colnames = ["time"]
            for _ in range(3):
                header.append(re.split(r"\s{2,}", fo.readline()))
            for j in range(1, len(header[1]) - 1):
                name = f"{header[0][j+1]} " + f"{header[1][j]} " + f"{header[2][j]}"
                colnames.append(name.replace("- ", "").replace(" + ", "+").lower())
            df = read_csv(fo, names=colnames, delim_whitespace=True, index_col=0)
        self.df = df
        self.columns = colnames[1:]
        self.times = df.index.to_list()

    def get_data(
        self,
        timesteps: list[int] = None,
        times: list[float] = None,
        columns: list = None,
    ):
        if columns is None:
            columns = self.columns

        if times is not None:
            if not isinstance(times, list):
                times = [times]
            return self.df.loc[times, columns]
        else:
            if timesteps is None:
                return self.df.iloc[:, [self.df.columns.get_loc(c) for c in columns]]
            else:
                if not isinstance(timesteps, list):
                    timesteps = [timesteps]
                return self.df.iloc[
                    timesteps, [self.df.columns.get_loc(c) for c in columns]
                ]

    def plot(
        self,
        timesteps: list[int] = None,
        times: list[float] = None,
        term: str = "time step",  # total, rate
        **kwargs,
    ):
        columns = [x for x in self.columns if term in x]
        df = self.get_data(timesteps, times, columns)
        ax = df.plot(**kwargs)
        return ax
