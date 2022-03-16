# %%
import numpy as np

def write_dat(prec, evapt, scf, H, x, z, timestep='hour', folder=''):
    n = prec - evapt
    pet = np.abs(n[n <= 0]).values
    e = pet * scf
    t = pet * (1 - scf)
    nrech = len(n)
    NPV = (n <= 0).sum()
    if NPV > 0:
        BCIT = 'T'
        SRES = np.full(NPV, 0.02)
        HA = np.full(NPV, -10000.0)
        ETSIM = 'T'
        RD = np.full(NPV, 2.0)
        RAbase = np.full(NPV, 0.005)
        RAtop = np.full(NPV, 0.005)
        Hroot = np.full(NPV, -150)
    else:
        BCIT = 'F'
        ETSIM = 'F'
    jtex = np.pad(np.full((len(z), len(x)), 2), pad_width=1, mode='constant', constant_values=1)
    a = ['\n',
         f'{len(n)} 0. 0.            /A-2 -- TMAX, STIM, ANG\n',
         f'm   {timestep} g   J          /A-3 -- ZUNIT, TUNIT, CUNX, HUNX\n',
         f'{len(x)+1} {len(z)+1}                 /A-4 -- NXR, NLY\n',
         f'{nrech} -10000000           /A-5 -- NRECH, NUMT\n',
         'F F F F                /A-6 -- RAD, ITSTOP, HEAT, SOLUTE\n',
         'F F T T F              /A-12 -- F11P, F7P, F8P, F9P, F6P\n',
         'F F T F T              /A-13 -- THPT, SPNT, PPNT, HPNT, VPNT\n',
         '0 1                    /A-14 -- IFAC, FACX. A-15 begins next line: DXR\n',
         ''+' '.join(str(i) for i in np.full(len(x)+1, np.diff(x)[0])) +'\n',
         '0 1                    /A-17 -- JFAC, FACZ. A-18 begins next line: DELZ\n',
         ''+' '.join(str(i) for i in np.full(len(z)+1, np.diff(z)[0])) +' /End A-18\n',
         f'{len(n)}                    /A-20 -- NPLT. A-21 begins next line: PLTIM\n',
         ''+' '.join(str(i) for i in (range(len(n)))) + '\n',
         '-33                     /A-24 -- NMB9\n',
         '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33     /A-25 -- MB9\n']
    b = ['1.0E-4 0.7 0.5          /B-1 -- EPS, HMAX, WUS\n',
         '2 500                  /B-4 -- MINIT, ITMAX\n',
         'T                      /B-5 -- PHRD\n',
         '2 6                    /B-6 -- NTEX, NPROP\n',
         '1                      /B-7 -- HFT hydraulicFunctionType\n',
         '1                      /B-8 -- ITEX. B-9 to begin next line: HK\n',
         '1.0 0.0 0.0 0.0 0.0 0.0 0.0\n',
         '2                      /B-8 -- ITEX. B-9 to begin next line: HK\n',
         '1.0 0.0292 1.0E-4 0.496 0.847 0.15 4.8 \n',
         '0                      /B-12 -- IROW. B-13 begins next line: JTEX\n']
    b_2 = [(''+' '.join(str(i) for i in line) + ' \n') for line in jtex[:-2, :]]
    b_2.extend(''+' '.join(str(i) for i in jtex[-1, :]) + ' /End B-13\n')
    b_3 = ['2 1.0                  /B-15 -- IREAD, FACTOR\n',
         f'{np.abs(H)} {H}               /B-16 -- DWTX, HMIN\n',
         f'{BCIT} {ETSIM}                    /B-18 -- BCIT, ETSIM\n'
         f'{NPV} 1.0                  /B-19 -- NPV, ETCYC\n',
         ''+' '.join(str(i) for i in e) + '     /B-20 -- PEVAL\n',
         ''+' '.join(str(i) for i in SRES) + '     /B-21 -- RDC(1,J)\n',
         ''+' '.join(str(i) for i in HA) + '     /B-22 -- RDC(2,J)\n',
         ''+' '.join(str(i) for i in t) + '     /B-23 -- PTVAL\n',
         ''+' '.join(str(i) for i in RD) + '     /B-24 -- RDC(3,J)\n',
         ''+' '.join(str(i) for i in RAbase) + '     /B-25 -- RDC(4,J)\n',
         ''+' '.join(str(i) for i in RAtop) + '     /B-26 -- RDC(5,J)\n',
         ''+' '.join(str(i) for i in Hroot) + '     /B-27 -- RDC(6,J)\n']
    b.extend(b_2)
    b.extend(b_3)
    a.extend(b)

    for i, r in enumerate(n, start=1):
        if r > 0:
            BCIT = 'F'
            ETSIM = 'F'
            eb = 2
            delt = 0.001
        elif r <= 0:
            BCIT = 'T'
            ETSIM = 'T'
            r = 0.0
            eb = 5
            if r == 0.0:
                delt = 1
            else:
                delt = 0.01
        seep_cells = (np.where(z <= np.abs(H))[0] + 1)[1:]
        head_cells = (np.where(z > np.abs(-2))[0] + 1)
        c = [f'1.0 {delt}             /C-1 -- TPER, DELT (Recharge Period {i})\n',
            '1.5 0.5 1.0E-4 0.01    /C-2 -- TMLT, DLTMX, DLTMIN, TRED\n',
            '1000.0 0.0             /C-3 -- DSMAX, STERR\n',
            '0.0                    /C-4 -- POND\n',
            'F                      /C-5 -- PRNT\n',
            f'{BCIT} {ETSIM} T                  /C-6 -- BCIT, ETSIM, SEEP\n',
            '1                      /C-7 -- NFCS\n',
            f'{len(seep_cells)} 0                   /C-8 -- JJ, JLAST. C-9 begins next line: J, N\n',
            f''.join(str(i) + ' 2\n' for i in seep_cells),
            '0                      /C-10 -- IBC\n',
            f''.join(str(i) + ' 2 3 0.0              /C-11 -- JJ, NN, NTX, PFDUM\n' for i in seep_cells),
            f''.join(str(i) + f' 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n' for i in head_cells),
            f''.join(f'2 {str(i+1)} {eb} {r}      /C-11 -- JJ, NN, NTX, PFDUM\n' for i in range(len(x))),
            f'-999999                /C-19 -- End of data for recharge period {i}\n']
        a.extend(c)
    a.extend('-999999 /End of input data file')
    with open(f'{folder}vs2drt.dat', 'w') as f:
        for item in a:
            f.write(item)

#%%
            # '81 2\n',
            # '80 2\n',
            # '79 2\n',
            # '78 2\n',
            # '77 2\n',
            # '76 2\n',
            # '75 2\n',
            # '74 2\n',
            # '73 2\n',
            # '72 2\n',
            # '71 2\n',
            # '70 2\n',
            # '69 2\n',
            # '68 2\n',
            # '67 2\n',
            # '66 2\n',
            # '65 2\n',
            # '64 2\n',
            # '63 2\n',
            # '62 2\n',
            # '61 2\n',
            # '60 2\n',
            # '59 2\n',
            # '58 2\n',
            # '57 2\n',
            # '56 2\n',
            # '55 2\n',
            # '54 2\n',
            # '53 2\n',
            # '52 2\n',
            # '51 2\n',
            # '50 2\n',
            # '49 2\n',
            # '48 2\n',
            # '47 2\n',
            # '46 2\n',
            # '45 2\n',
            # '44 2\n',
            # '43 2\n',
            # '42 2\n',
            # '41 2\n',
            # '40 2\n',
            # '39 2\n',
            # '38 2\n',
            # '37 2\n',
            # '36 2\n',
            # '35 2\n',
            # '34 2\n',
            # '33 2\n',
            # '32 2\n',
            # '31 2\n',
            # '30 2\n',
            # '29 2\n',
            # '28 2\n',
            # '27 2\n',
            # '26 2\n',
            # '25 2\n',
            # '24 2\n',
            # '23 2\n',
            # '22 2\n',
            # '21 2\n',
            # '20 2\n',
            # '19 2\n',
            # '18 2\n',
            # '17 2\n',
            # '16 2\n',
            # '15 2\n',
            # '14 2\n',
            # '13 2\n',
            # '12 2\n',
            # '11 2\n',
            # '10 2\n',
            # '9 2\n',
            # '8 2\n',
            # '7 2\n',
            # '6 2\n',
            # '5 2\n',
            # '4 2\n',
            # '3 2\n',
            # '2 2\n',
            # '2 2 3 0.0              /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '3 2 3 0.0              /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '4 2 3 0.0              /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '5 2 3 0.0              /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '6 2 3 0.0              /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '7 2 3 0.0              /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '8 2 3 0.0              /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '9 2 3 0.0              /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '10 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '11 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '12 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '13 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '14 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '15 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '16 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '17 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '18 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '19 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '20 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '21 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '22 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '23 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '24 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '25 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '26 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '27 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '28 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '29 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '30 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '31 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '32 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '33 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '34 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '35 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '36 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '37 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '38 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '39 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '40 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '41 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '42 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '43 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '44 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '45 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '46 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '47 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '48 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '49 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '50 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '51 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '52 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '53 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '54 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '55 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '56 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '57 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '58 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '59 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '60 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '61 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '62 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '63 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '64 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '65 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '66 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '67 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '68 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '69 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '70 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '71 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '72 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '73 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '74 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '75 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '76 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '77 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '78 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '79 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '80 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # '81 2 3 0.0             /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'82 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'83 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'84 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'85 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'86 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'87 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'88 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'89 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'90 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'91 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'92 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'93 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'94 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'95 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'96 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'97 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'98 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'99 2 4 {H}            /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'100 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'101 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'102 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'103 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'104 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'105 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'106 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'107 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'108 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'109 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'110 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'111 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'112 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'113 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'114 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'115 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'116 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'117 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'118 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'119 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'120 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'121 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'122 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'123 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'124 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'125 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'126 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'127 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'128 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'129 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'130 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'131 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'132 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'133 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'134 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'135 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'136 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'137 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'138 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'139 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'140 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'141 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'142 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'143 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'144 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'145 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'146 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'147 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'148 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'149 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'150 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'151 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'152 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'153 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'154 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'155 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'156 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'157 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'158 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'159 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'160 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'161 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'162 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'163 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'164 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'165 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'166 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'167 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'168 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'169 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'170 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'171 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'172 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'173 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'174 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'175 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'176 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'177 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'178 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'179 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'180 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'181 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'182 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'183 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'184 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'185 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'186 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'187 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'188 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'189 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'190 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'191 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'192 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'193 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'194 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'195 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'196 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'197 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'198 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'199 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'200 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'201 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'202 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'203 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'204 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'205 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'206 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'207 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'208 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'209 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'210 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'211 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'212 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'213 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'214 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'215 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'216 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'217 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'218 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'219 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'220 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'221 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'222 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'223 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'224 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'225 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'226 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'227 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'228 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'229 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'230 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'231 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'232 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'233 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'234 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'235 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'236 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'237 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'238 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'239 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'240 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'241 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'242 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'243 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'244 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'245 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'246 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'247 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'248 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'249 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'250 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'251 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'252 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'253 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'254 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'255 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'256 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'257 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'258 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'259 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'260 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'261 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'262 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'263 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'264 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'265 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'266 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'267 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'268 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'269 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'270 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'271 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'272 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'273 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'274 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'275 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'276 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'277 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'278 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'279 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'280 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'281 2 4 {H}           /C-11 -- JJ, NN, NTX, PFDUM\n',
            # f'2 3 {eb} {r}      /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 4 {eb} {r}      /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 5 {eb} {r}      /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 6 {eb} {r}      /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 7 {eb} {r}      /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 8 {eb} {r}      /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 9 {eb} {r}      /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 10 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 11 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 12 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 13 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 14 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 15 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 16 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 17 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 18 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 19 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 20 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 21 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 22 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 23 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 24 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 25 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 26 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 27 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 28 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 29 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 30 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 31 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 32 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 33 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 34 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 35 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 36 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 37 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 38 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 39 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 40 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 41 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 42 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 43 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 44 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 45 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 46 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 47 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 48 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 49 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 50 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 51 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 52 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 53 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 54 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 55 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 56 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 57 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 58 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 59 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 60 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 61 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 62 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 63 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 64 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 65 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 66 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 67 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 68 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 69 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 70 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 71 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 72 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 73 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 74 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 75 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 76 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 77 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 78 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 79 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 80 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 81 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 82 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 83 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 84 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 85 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 86 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 87 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 88 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 89 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 90 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 91 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 92 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 93 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 94 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 95 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 96 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 97 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 98 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 99 {eb} {r}     /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 100 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 101 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 102 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 103 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 104 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 105 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 106 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 107 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 108 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 109 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 110 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 111 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 112 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 113 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 114 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 115 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 116 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 117 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 118 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 119 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 120 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 121 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 122 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 123 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 124 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 125 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 126 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 127 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 128 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 129 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 130 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 131 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 132 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 133 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 134 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 135 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 136 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 137 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 138 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 139 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 140 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 141 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 142 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 143 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 144 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 145 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 146 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 147 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 148 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 149 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 150 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 151 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 152 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 153 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 154 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 155 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 156 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 157 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 158 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 159 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 160 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 161 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 162 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 163 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 164 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 165 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 166 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 167 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 168 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 169 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 170 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 171 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 172 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 173 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 174 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 175 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 176 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 177 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 178 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 179 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 180 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 181 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 182 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 183 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 184 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 185 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 186 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 187 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 188 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 189 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 190 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 191 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 192 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 193 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 194 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 195 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 196 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 197 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 198 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 199 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',
#             f'2 200 {eb} {r}    /C-11 -- JJ, NN, NTX, PFDUM\n',            