# %%
import re
import pickle
import numpy as np
from pandas import read_csv, DataFrame


def variables_out(path='', to_pickle=False):
    """_summary_

    Parameters
    ----------
    path : str, optional
        path to variables.out file, by default ''

    Returns
    -------
    h: dict
        Dictionary of variables.out with timesteps as keys
    """
    with open(f'{path}variables.out') as f:
        fo = f.read().splitlines()

    t_idx = [i for i, s in enumerate(fo) if ' TIME =    ' in s]
    tsteps = [float(fo[t_idx[i]].split()[-2]) for i in range(len(t_idx))]

    h = {}
    delt = t_idx[1] - t_idx[0]
    for i, t in enumerate(tsteps):
        dat = np.array([dr.split() for dr
                        in fo[t_idx[i]+2:t_idx[i]+delt-1]], dtype=float)
        h[t] = dat[1:-1, 1:-1]

    if pickle:
        with open('ph.pickle', 'wb') as handle:
            pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return h


def balance_out(path=''):
    """_summary_

    Parameters
    ----------
    path : str, optional
        _description_, by default ''

    Returns
    -------
    pandas.DataFrame
        dataframe with output of read_balance file
    """
    with open(f'{path}balance.out', 'r') as f:
        header = []
        colnames = ['time']
        for _ in range(3):
            header.append(re.split(r'\s{2,}', f.readline()))
        for j in range(1, len(header[1])-1):
            name = (f'{header[0][j+1]} '+f'{header[1][j]} '+f'{header[2][j]}')
            colnames.append(name.replace('- ', '').replace(' + ', '+').lower())
        df = read_csv(f, names=colnames, delim_whitespace=True, index_col=0)
    return df


def outfile(path=''):
    num_lines = sum(1 for line in open(f'{path}vs2drt.out'))

    with open(f'{path}vs2drt.out') as f:
        [f.readline() for _ in range(17)]
        # SPACE AND TIME CONSTANTS
        c = np.array([], dtype=int)
        for i in range(7):
            if i == 0:
                c = np.append(c, int(float(f.readline().split()[-2])))
            else:
                c = np.append(c, int(float(f.readline().split()[-1])))
        # SOLUTION OPTIONS
        s = []
        search = []
        d = {}
        for i in range(14):
            s.append(f.readline().split()[-1])
        if s[11] == 'T':
            search.append('TOTAL HEAD')
            d[search[-1]] = {}
        if s[10] == 'T':
            search.append('PRESSURE HEAD')
            d[search[-1]] = {}
        if s[9] == 'T':
            search.append('SATURATION')
            d[search[-1]] = {}
        if s[8] == 'T':
            search.append('MOISTURE CONTENT')
            d[search[-1]] = {}
        if s[12] == 'T':
            search.append('X-VELOCITY')
            d[search[-1]] = {}
            search.append('Z-VELOCITY')
            d[search[-1]] = {}
        if s[7] == 'T':
            search.append('MASS BALANCE')
            d[search[-1]] = {}
        # GRID SPACING IN VERTICAL DIRECTION
        z = np.array([], dtype=float)
        for i in range(int(np.ceil(c[4]/10))):
            z = np.append(z, np.array(f.readline().split(), float))
        # GRID SPACING IN HORIZONTAL OR RADIAL DIRECTION
        f.readline()
        x = np.array([], dtype=float)
        for i in range(int(np.ceil(c[5]/10))):
            x = np.append(x, np.array(f.readline().split(), float))
        # REST
        time = np.array([], dtype=float)
        timestep = np.array([], dtype=int)
        for i in range(num_lines):
            line = f.readline()
            if 'TOTAL ELAPSED TIME' in line:
                time = np.append(time, float(line.split()[-2]))
                timestep = np.append(timestep, int(f.readline().split()[-1]))
                if len(timestep) == 1:
                    [f.readline() for i in range(2)]
                    if 'PRESSURE HEAD' in f.readline():
                        [f.readline() for i in range(3)]
                        aap = np.empty((len(z)-2, len(x)-2))
                        for j in range(len(z)-2):
                            aap[j, :] = np.array(
                                f.readline().split()[1:], dtype=float)
                        d['PRESSURE HEAD'][time[-1]] = aap
                    else:
                        continue
                else:
                    [f.readline() for i in range(2)]
                    for ky in d:
                        [f.readline() for i in range(4)]
                        aap = np.empty((len(z)-2, len(x)-2))
                        for j in range(len(z)-2):
                            aap[j, :] = np.array(
                                f.readline().split()[1:], dtype=float)
                        d[ky][time[-1]] = aap

    return x, z, d


def get_gwt_1D(pressure_head, depth):
    sign = np.signbit(pressure_head)
    if sign.any() == False:
        gwl = depth[0]
    elif sign.sum() == len(pressure_head):
        gwl = depth[-1]
    else:
        idx = np.where(np.diff(sign))[0]
        if len(idx) > 1:
            idx = idx[0]
        gwl = (0 - pressure_head[idx + 1]) * (depth[idx] - depth[idx + 1]) / \
              (pressure_head[idx] - pressure_head[idx + 1]) + depth[idx + 1]
    return gwl


def get_gwt_2D(data, z):
    gwt = np.array([])
    for i in range(data.shape[1]):
        gwt = np.append(gwt, get_gwt_1D(np.flip(data[:, i]), z))
    return gwt

def get_gwt_intime(h, x, z, to_csv=False):
    gwt_d = {}
    for ky in h:
        gwt_d[ky] = get_gwt_2D(data=h[ky], z=z)

    gwl = DataFrame(gwt_d, index=x).transpose()

    if to_csv:
        gwl.to_csv('groundwaterlevel.csv')
    
    return gwl


if __name__ == '__main__':
    x, z, d = read_vs2drt_out(path = '../own/test08/')


# %%
