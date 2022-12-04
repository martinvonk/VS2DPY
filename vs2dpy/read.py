import os
from io import BytesIO
from numpy import genfromtxt, arange, empty


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

        times_pointer = arange(time_count[0], self.size, byte_diff0)
        byte_diffs = [byte_diff0, byte_diff1, byte_diff2]
        shape = genfromtxt(lines[1:]).shape
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

        if not isinstance(times, list):
            times = [times]

        if times != [None]:
            timesteps = [self.times.index(x) for x in times]

        tps = [self.time_pointers[x] for x in timesteps]
        datas = empty((len(tps), self.shape[0], self.shape[1]))
        with open(self.path, "rb") as fo:
            for i, tp in enumerate(tps):
                fo.seek(tp + self.byte_diffs[1])
                datas[i] = genfromtxt(BytesIO(fo.read(self.byte_diffs[2])))
        return datas

    def get_all_data(
        self,
    ):
        return self.get_data(arange(len(self.time_pointers)).tolist())
