# %%
import re
import pickle
import numpy as np
from pandas import read_csv, DataFrame
from tqdm import tqdm

def variables_out(path='', tmax=None, byte_corr = 105):
    """_summary_

    Parameters
    ----------
    path : str, optional
        _description_, by default ''
    tmax : _type_, optional
        _description_, by default None
    to_pickle : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    
    with open(f'{path}variables.out') as f:
        pbar = tqdm(desc='Timestep', total=int(tmax))
        tsps = []
        bl = []
        line = ''
        while 'TIME' not in line:
            line = f.readline()
            bl.append(len(line)+1)
        line = f.readline()
        bl.append(len(line)+1)
        while 'TIME' not in line:
            line = f.readline()
            bl.append(len(line)+1)
        bts = np.sum(bl[:-2]) - byte_corr
        f.seek(0,0)
        chunk = [1]
        while len(chunk) > 0:
            chunk = f.read(bts)
            if len(chunk) == 0:
                break
            l = chunk.splitlines()
            tsps.append(float(l[1].split()[-2]))
            # print(l)
            aap = np.array([dr.split() for dr in l[3:]], dtype=float)
            with open(f'{path}ph/ph_{tsps[-1]}.npy', 'wb') as fo:
                np.save(fo, aap)
            pbar.update(1)
                

def variables_out2(path='', tmax=None, msize=5000, to_file=False):

    with open(f'{path}variables.out') as f:
        pbar = tqdm(desc='Timestep', total=int(tmax))
        tsps = []
        t1 = []
        line = ''
        while 'TIME' not in line:
            line = f.readline()
        tsps.append(float(line.split()[-2]))
        line = f.readline()
        while 'TIME' not in line:
            line = f.readline()
            t1.append(line)
        dat = np.array([dr.split() for dr
                            in  t1[:-2]], dtype=float)
        aap = np.zeros((msize * dat.shape[0], dat.shape[1]))
        aap[:dat.shape[0], :] = dat
        cnt = dat.shape[0]

        tsps.append(float(line.split()[-2]))
        pbar.update(2)

        for line in f:
            lsp = line.split()

            if 'TIME' in lsp:
                tsps.append(float(lsp[-2]))
                pbar.update(1)
            elif len(lsp) == dat.shape[1]:
                l = np.array(lsp, dtype=float)
                aap[cnt, :] = l
                cnt += 1

            if cnt == msize * dat.shape[0]:
                if to_file:
                    with open(f'ph1_{int(tsps[-1])}.npy', 'wb') as fo:
                        np.save(fo, aap)
                aap = np.zeros((msize * dat.shape[0], dat.shape[1]))
                cnt = 0

    if to_file:
        with open(f'ph1_{int(tsps[-1])}.npy', 'wb') as fo:
            np.save(fo, aap)

    return tsps

    # with open(f'{path}variables.out') as f:
    #     pbar = tqdm(desc='Timestep', total=int(tmax))
    #     line = ''
    #     tsps = []
    #     while 'TIME' not in line:
    #         line = f.readline()
    #     tsps.append(float(line.split()[-2]))
    #     # prepare to read output t
    #     cnt = 0
    #     aap = []
    #     while line:
    #         while tsps[-1] < tmax+1:
    #             t = []
    #             line = f.readline()
    #             while 'TIME' not in line:
    #                 line = f.readline()
    #                 t.append(line)
    #             dat = np.array([dr.split() for dr
    #                             in  t[0:-2]], dtype=float)[1:-1, 1:-1]
    #             if tsps[-1] in np.arange(1.0, tmax, 10000):
    #                 cnt = 0
    #                 if to_file:
    #                     with open(f'ph_{int(tsps[-1])}.npy', 'wb') as fo:
    #                         np.save(fo, aap)
    #                 aap = np.zeros((10000, dat.shape[0], dat.shape[1]))
    #             aap[cnt] = dat
    #             tsps.append(float(line.split()[-2]))
    #             pbar.update(1)
    #             cnt += 1
    #             if tsps[-1] == tmax:
    #                 it = len(t)
    #                 t = []
    #                 line = f.readline()
    #                 for _ in range(it):
    #                     line = f.readline()
    #                     t.append(line)
    #                 dat = np.array([dr.split() for dr
    #                             in  t[0:-2]], dtype=float)[1:-1, 1:-1]
    #                 aap[cnt] = dat
    #                 pbar.update(1)
    #                 if to_file:
    #                     with open(f'ph{int(tsps[-1])}.npy', 'wb') as fo:
    #                         np.save(fo, aap)
    #                 break

    #     return aap

    # # with open(f'{path}variables.out') as f:
    #     fo = f.read().splitlines()


    # t_idx = [i for i, s in enumerate(fo) if ' TIME =    ' in s]
    # tsteps = [float(fo[t_idx[i]].split()[-2]) for i in range(len(t_idx))]

    # h = {}
    # delt = t_idx[1] - t_idx[0]
    # for i, t in enumerate(tsteps):
    #     dat = np.array([dr.split() for dr
    #                     in fo[t_idx[i]+2:t_idx[i]+delt-1]], dtype=float)
    #     h[t] = dat[1:-1, 1:-1]

    # if to_pickle:
    #     with open('ph.pickle', 'wb') as handle:
    #         pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    """_summary_

    Parameters
    ----------
    path : str, optional
        _description_, by default ''

    Returns
    -------
    _type_
        _description_
    """    
    num_lines = sum(1 for line in open(f'{path}vs2drt.out'))
    print(f'{num_lines=}')
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
        for i in tqdm(range(num_lines)):
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
    """_summary_

    Parameters
    ----------
    pressure_head : _type_
        _description_
    depth : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
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
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    z : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    gwt = np.array([])
    for i in range(data.shape[1]):
        gwt = np.append(gwt, get_gwt_1D(np.flip(data[:, i]), z))
    return gwt


def get_gwt_intime(h, x, z, to_csv=False):
    """_summary_

    Parameters
    ----------
    h : _type_
        _description_
    x : _type_
        _description_
    z : _type_
        _description_
    to_csv : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """    
    gwt_d = {}
    for ky in h:
        gwt_d[ky] = get_gwt_2D(data=h[ky], z=z)

    gwl = DataFrame(gwt_d, index=x).transpose()

    if to_csv:
        gwl.to_csv('groundwaterlevel.csv')
    
    return gwl

    
# %%
# old variables out function using dictionary and pickle
# def variables_out(path='', tmax=None, to_pickle=False):
#     """_summary_

#     Parameters
#     ----------
#     path : str, optional
#         _description_, by default ''
#     tmax : _type_, optional
#         _description_, by default None
#     to_pickle : bool, optional
#         _description_, by default False

#     Returns
#     -------
#     _type_
#         _description_
#     """
    
#     with open(f'{path}variables.out') as f:
#         pbar = tqdm(desc='Timestep', total=tmax)
#         line = ''
#         tsps = []
#         h = {}
#         while 'TIME' not in line:
#             line = f.readline()
#         tsps.append(float(line.split()[-2]))
#         # prepare to read output t
#         while tsps[-1] < tmax+1:
#             if tsps[-1] in np.arange(8760.0, tmax, 8760.0):
#                 if to_pickle:
#                     with open(f'ph_{tsps[-1]}.pickle', 'wb') as handle:
#                         pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)
#                 h = {}
#             t = []
#             line = f.readline()
#             while 'TIME' not in line:
#                 line = f.readline()
#                 t.append(line)
#             dat = np.array([dr.split() for dr
#                             in  t[0:-2]], dtype=float)
#             h[tsps[-1]] = dat[1:-1, 1:-1]
#             tsps.append(float(line.split()[-2]))
#             pbar.update(1)
#             if tsps[-1] == tmax:
#                 it = len(t)
#                 t = []
#                 line = f.readline()
#                 for _ in range(it):
#                     line = f.readline()
#                     t.append(line)
#                 dat = np.array([dr.split() for dr
#                             in  t[0:-2]], dtype=float)
#                 h[tsps[-1]] = dat[1:-1, 1:-1]
#                 pbar.update(1)
#                 if to_pickle:
#                     with open(f'ph_{tsps[-1]}.pickle', 'wb') as handle:
#                         pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)
#                 break

#     return h