#%%
import os
import numpy as np
import pandas as pd
from collections import OrderedDict


class Model:
    def __init__(
        self,
        ws: str,
        exe: str = "vs2drt.exe",
        titl: str = "Model created with VS2DPY",
        tmax: float = 1.0,
        stim: float = 0.0,
        zunit: str = "m",
        tunit: str = "sec",  # hour day year
    ):

        if not os.path.exists(ws):
            os.mkdir(ws)
            print(f"Directory {ws} created")

        self.ws = ws

        if not os.path.exists(exe):
            raise OSError("Executable not found.")
        else:
            self.exe = exe

        if len(titl) > 80:
            raise ValueError("TITL can't have more than 80 characters")
        else:
            self.titl = titl  # A-1

        self.tmax = tmax  # A-2
        self.stim = stim  # A-2
        self.zunit = zunit  # A-3
        self.tunit = tunit  # A-3

        # define_output
        self.nrech = None  # A-5
        self.numt = None  # A-5
        self.f11p = None  # A-12
        self.f7p = None  # A-12
        self.f8p = None  # A-12
        self.f9p = None  # A-12
        self.f6p = None  # A-12
        self.thpt = None  # A-13
        self.spnt = None  # A-13
        self.ppnt = None  # A-13
        self.hpnt = None  # A-13
        self.vpnt = None  # A-13
        self.nplt = None  # A-20
        self.pltim = None  # A-21
        self.nobs = None  # A-22
        self.obsrowncoln = None  # A-23
        self.nmb9 = None  # A-24
        self.mb9 = None  # A-25
        self.numbf = None  # B-33
        self.maxcells = None  # B-33
        self.idbf = None  # B-34
        self.numcells = None  # B-34
        self.bcrowncoln = None  # B-35
        self.prnt = None  # C-5

        # define_domain
        self.nxr = None  # A-4
        self.nly = None  # A-4
        self.dxr = None  # A-15
        self.delz = None  # A-18

        # define_solver
        self.eps = None  # B-1
        self.hmax = None  # B-1
        self.wus = None  # B-1
        self.minit = None  # B-4
        self.itmax = None  # B-4
        self.delt = None  # C-1
        self.tmlt = None  # C-2
        self.dltmx = None  # C-2
        self.dltmin = None  # C-2
        self.tred = None  # C-2
        self.dsmax = None  # C-3
        self.sterr = None  # C-3

        # define_soil
        self.ntex = None  # B-6
        self.hft = None  # B-7
        self.nprop = None  # B-6
        self.textures = None  # B-8 and B-9
        self.jtex = None  # B-13

        # define_initialc
        self.phrd = None  # B-5
        self.iread = None  # B-15
        self.factor = None  # B-15
        self.dwtx = None  # B-16
        self.hmin = None  # B-16

        # define_evap
        self.bcit = None  # B-18
        self.etsim = None  # B-18
        self.npv = None  # B-19
        self.etcyc = None  # B-19
        self.peval = None  # B-20
        self.rdc1 = None  # B-21
        self.rdc2 = None  # B-22
        self.ptval = None  # B-23
        self.rdc3 = None  # B-24
        self.rdc4 = None  # B-25
        self.rdc5 = None  # B-26
        self.rdc6 = None  # B-27

        # define_rp
        self.tper = None  # C-1
        self.bc = None  # C-4 - C-11

    def define_output(
        self,
        nrech: int = 1,
        numt: int = 1,
        f11p: bool = False,
        f7p: bool = False,
        f8p: bool = True,
        f9p: bool = True,
        f6p: bool = False,
        thpt: bool = False,
        spnt: bool = False,
        ppnt: bool = False,
        hpnt: bool = False,
        vpnt: bool = False,
        nplt: int = 1,
        pltim: np.ndarray = None,
        nobs: int = 0,
        obsrowncoln: list[tuple] = None,
        nmb9: int = 1,
        mb9: np.ndarray = None,
        numbf: int = 0,
        maxcells: int = 0,
        idbf: int = 0,
        numcells: int = 0,
        bcrowncoln: list[tuple] = None,
        prnt: bool = False,
    ):

        self.nrech = nrech  # A-5
        self.numt = numt  # A-5

        self.f11p = f11p  # A-12
        self.f7p = f7p  # A-12
        self.f8p = f8p  # A-12
        self.f9p = f9p  # A-12
        self.f6p = f6p  # A-12
        self.thpt = thpt  # A-13
        self.spnt = spnt  # A-13
        self.ppnt = ppnt  # A-13
        self.hpnt = hpnt  # A-13
        self.vpnt = vpnt  # A-13

        if self.f8p:
            self.nplt = nplt  # A-20
            if len(pltim) != nplt:
                raise ValueError("Number of entries must be equal to NPLT")
            self.pltim = pltim  # A-21
        if self.f11p:
            self.nobs = nobs  # A-22
            if len(obsrowncoln) != nobs:
                raise ValueError("Number of entries must be equal to NOBS")
            self.obsrowncoln = obsrowncoln  # A-23
        if self.f9p:
            if np.abs(nmb9) > 73:
                raise ValueError("Number must be less than 73")
            self.nmb9 = nmb9  # A-24
            if mb9 is None:
                mb9 = np.arange(np.abs(nmb9))
            if len(mb9) != nmb9:
                raise ValueError("Number of entries must be equal to NMB9")
            self.mb9 = mb9  # A-25
        if self.f7p:
            self.numbf = numbf  # B-33
            self.maxcells = maxcells  # B-33
            self.idbf = idbf  # B-34
            self.numcells = numcells  # B-34
            if len(bcrowncoln) != numcells:
                raise ValueError("Number of entries must be equal to NUMCELLS")
            self.bcrowncoln = bcrowncoln  # B-35

        self.prnt = prnt  #  C-5

    def define_domain(
        self,
        nxr: int = 1,
        nly: int = 1,
        dxr: np.ndarray = None,
        delz: np.ndarray = None,
    ):
        self.nxr = nxr  # A-4
        self.nly = nly  # A-4
        if dxr is None:  # ifac = 0
            self.dxr = np.linspace(0, 1, num=nxr)  # A-15
        else:
            if len(dxr) != nxr:
                raise ValueError("Number of entries must be equal to NXR")
            self.dxr = dxr  # A-15
        if delz is None:  # jfac = 0
            self.delz = np.linspace(0, 1, num=nly)
        else:
            if len(delz) != nly:
                raise ValueError("Number of entries must equal NLY")
            self.delz = delz  # A-18

    def define_solver(
        self,
        eps: float = 0.0001,
        hmax: float = 0.7,
        wus: float = 0.5,
        minit: int = 2,
        itmax: int = 10,
        delt: float = 0.1,
        tmlt: float = 1.5,
        dltmx: float = 1.0,
        dltmin: float = 0.0001,
        tred: float = 0.01,
        dsmax: float = 100,
        sterr: float = 0.0,
    ):
        self.eps = eps  # B-1
        if 1.2 <= hmax <= 0.4:
            print(f"Relaxation parameter outside of general range")
        self.hmax = hmax  # B-1
        self.wus = wus  # B-1
        self.minit = minit  # B-4
        self.itmax = itmax  # B-4
        self.delt = delt  # C-1
        self.tmlt = tmlt  # C-2
        self.dltmx = dltmx  # C-2
        self.dltmin = dltmin  # C-2
        self.tred = tred  # C-2
        self.dsmax = dsmax  # C-3
        self.sterr = sterr  # C-3

    def define_soil(
        self,
        ntex: int = 1,
        nprop: int = 6,
        hft: int = 1,
        textures: dict = {0: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        jtex: np.ndarray = None,
    ):
        self.ntex = ntex  # B-6
        self.hft = hft  # B-7
        if hft in (0, 1, 4):
            if nprop != 6:
                raise ValueError(
                    "When using Brooks and Corey, van Genuchten or Nimmo-Rossi functions, set NPROP to 6"
                )
        elif hft in (2,):
            if nprop != 8:
                raise ValueError("When using Haverkamp functions, set NPROP to 8")
        self.nprop = nprop  # B-6
        self.textures = textures  # B-8 and B-9

        # IROW = 0 # B-12
        if jtex is None:
            self.jtex = np.ones((10, 1), dtype=int)
            self.jtex = np.pad(self.jtex, pad_width=1, mode="constant")
        else:
            if jtex.ndim != 2:
                raise ValueError("JTEX must be 2-dimensional")
            self.jtex = jtex.astype(int)  # B-13

    def define_initialc(
        self,
        phrd: bool = True,
        factor: float = 1.0,
        iread: int = 0,
        dwtx: float = None,
        hmin: float = None,
    ):
        self.phrd = phrd  # B-5
        self.iread = iread  # B-15
        self.factor = factor  # B-15

        if self.iread == 2:
            self.dwtx = dwtx  # B-16
            if hmin > 0:
                raise ValueError("HMIN must be negative")
            self.hmin = hmin  # B-16
            if self.factor != 1.0:
                raise ValueError("FACTOR should be equal to 1.0 if IREAD=2")
        elif self.iread == 1:
            raise NotImplementedError()
        elif self.iread == 0:
            print("all initial conditions are set equal to factor")
            # if len(self.factor) <= 1:
            # Make sure FACTOR is a good array
        elif self.iread == 3:
            raise NotImplementedError()

    def define_evap(
        self,
        bcit: bool = False,
        etsim: bool = False,
        npv: int = None,
        etcyc: float = None,
        peval: np.ndarray = None,
        rdc1: np.ndarray = None,  # SRES
        rdc2: np.ndarray = None,  # HA
        ptval: np.ndarray = None,
        rdc3: np.ndarray = None,  # RD
        rdc4: np.ndarray = None,  # RAbase
        rdc5: np.ndarray = None,  # RAtop
        rdc6: np.ndarray = None,  # Hroot
    ):
        self.bcit = bcit  # B-18
        self.etsim = etsim  # B-18
        if bcit or etsim:
            self.npv = npv  # B-19
            self.etcyc = etcyc  # B-19
        if bcit:
            for name, x in zip((peval, rdc1, rdc2), ("peval", "rdc1", "rdc2")):
                if len(x) != self.npv:
                    raise ValueError(f"Number of entries of {name} must equal NPV")
            self.peval = peval  # B-20
            self.rdc1 = rdc1  # B-21
            self.rdc2 = rdc2  # B-22
        if etsim:
            for name, x in zip(
                (ptval, rdc3, rdc4, rdc5, rdc6), ("peval", "rdc1", "rdc2")
            ):
                if len(x) != self.npv:
                    raise ValueError(f"Number of entries of {name} must equal NPV")
            self.ptval = ptval  # B-23
            self.rdc3 = rdc3  # B-24
            self.rdc4 = rdc4  # B-25
            self.rdc5 = rdc5  # B-26
            self.rdc6 = rdc6  # B-27

    def create_bc(
        self,
        bcitrp: bool = False,
        etsimrp: bool = False,
        seep: bool = False,
        pond: float = 0.0,
        nfcs: int = 1,
        jj: int = 1,
        jlast: int = 0,
        jspx: list[tuple] = None,
        ibc: int = 0,
        ntx: np.ndarray = None,
    ):
        bc = {}
        bc["pond"] = pond  # C-4
        bc["bcitrp"] = bcitrp  # C-6
        bc["etsimrp"] = etsimrp  # C-6
        bc["seep"] = seep  # C-6
        if seep:
            bc["nfcs"] = nfcs  # C-6
            bc["jj"] = jj  # C-6
            bc["jlast"] = jlast  # C-6
            if len(jspx) != jj:
                raise ValueError(f"Number of entries of JSPX must be equal to JJ")
            bc["jspx"] = jspx
        bc["ibc"] = ibc  # C-10
        if ibc == 0:
            if ntx is None:
                bc["ntx"] = np.array(
                    [
                        [1, 1, 2, 0.1],
                        [self.jtex.shape[0] - 2, self.jtex.shape[1] - 2, 7, 0.0],
                    ]
                )
            else:
                bc["ntx"] = ntx  # C-11
        return bc

    def define_rp(
        self,
        tper: np.ndarray = np.array([1.0]),
        bc: dict[dict] = None,
    ):
        self.tper = tper  # C-1
        if len(tper) != len(bc):
            raise ValueError(
                f"Number of entries of TPER must be equal to boundary conditions"
            )
        if bc is None:
            self.bc = self.create_bc()
        else:
            self.bc = bc  # C-4 - C-11

    def write_input(self):
        A = self.write_A()
        B = self.write_B()
        C = self.write_C()
        ABC = list(A.values()) + list(B.values()) + list(C.values())

        return ABC

    def write_A(self):
        A = OrderedDict()
        A["A01"] = f"{self.titl} /A-1 -- TITL\n"
        A["A02"] = f"{self.tmax} {self.stim} 0. /A-2 -- TMAX, STIM, ANG\n"
        A["A03"] = f"{self.zunit} {self.tunit} g J /A-3 -- ZUNIT, TUNIT, CUNX, HUNX\n"
        A["A04"] = f"{self.nxr} {self.nly} /A-4 -- NXR, NLY\n"
        A["A05"] = f"{self.nrech} {self.numt} /A-5 -- NRECH, NUMT\n"
        A["A06"] = f"F F F F /A-6 -- RAD, ITSTOP, HEAT, SOLUTE\n"
        A_12 = [
            "T" if x else "F"
            for x in (self.f11p, self.f7p, self.f8p, self.f9p, self.f6p)
        ]
        A["A12"] = " ".join(A_12) + " /A-12 -- F11P, F7P, F8P, F9P, F6P\n"
        A_13 = [
            "T" if x else "F"
            for x in (self.thpt, self.spnt, self.ppnt, self.hpnt, self.vpnt)
        ]
        A["A13"] = " ".join(A_13) + " /A-13 -- THPT, SPNT, PPNT, HPNT, IFAC\n"
        A["A14"] = f"0 0 /A-14 -- IFAC, FACX. A-15 begins next line: DXR\n"
        A["A15"] = f"{' '.join(self.dxr.astype(str))} \n"
        A["A17"] = f"0 1 /A-17 -- JFAC, FACZ. A-18 begins next line: DELZ\n"
        A["A18"] = f"{' '.join(self.delz.astype(str))} /End A-18\n"
        if self.f8p:
            A["A20"] = f"{self.nplt} /A-20 -- NPLT. A-21 begins next line: PLTIM\n"
            A["A21"] = f"{' '.join(self.pltim.astype(str))} /A-21\n"
        if self.f11p:
            A["A22"] = f"{self.nobs} /A-22 -- NOBS\n"
            A["A23"] = f"{self.obsrowncoln} /A-23 -- ROW(N), COL(N),N=1,NOBS)\n"
        if self.f9p:
            A["A24"] = f"{self.nmb9} /A-24 -- NMB9\n"
            A["A25"] = f"{' '.join((self.mb9+1).astype(str))} /A-25 -- MB9\n"
        return A

    def write_B(self):
        B = OrderedDict()
        B["B01"] = f"{self.eps} {self.hmax} {self.wus} /B-1 -- EPS, HMAX, WUS\n"
        B["B04"] = f"{self.minit} {self.itmax} /B-4 -- MINIT, ITMAX\n"
        B["B05"] = f"{['T' if x else 'F' for x in (self.phrd,)][0]} /B-5 -- PHRD\n"
        B["B06"] = f"{self.ntex} /B-6 -- NTEX, NPROP"
        B["B07"] = f"{self.hft} /B-7 -- HFT hydraulicFunctionType\n"
        B["B08"] = ""  # also B09
        for ky in self.textures:
            B["B08"] += f"{ky+1} /B-8 -- ITEX. B-9 to begin next line: HK\n"
            B["B08"] += f"{' '.join(self.textures[ky].astype(str))}\n"
        B["B12"] = f"0 /B-12 -- IROW. B-13 begins next line: JTEX\n"
        B["B13"] = ""
        for i, row in enumerate(self.jtex.astype(str)):
            if i == len(self.jtex) - 1:
                B["B13"] += f"{' '.join(row)} /End B-13\n"
            else:
                B["B13"] += f"{' '.join(row)}\n"
        B["B15"] = f"{self.iread} {self.factor} /B-15 -- IREAD, FACTOR\n"
        if self.iread == 2:
            B["B16"] = f"{self.dwtx} {self.hmin} /B-16 -- DWTX, HMIN\n"
        B_18 = ["T" if x else "F" for x in (self.bcit, self.etsim)]
        B["B18"] = f"{' '.join(B_18)} /B-18 -- BCIT, ETSIM\n"
        if self.bcit or self.etsim:
            B["B19"] = f"{self.npv} {self.etcyc} /B-19 -- NPV, ETCYC\n"
        if self.bcit:
            B["B20"] = f"{' '.join(self.peval.astype(str))} /B-20 -- PEVAL\n"
            B["B21"] = f"{' '.join(self.rdc1.astype(str))} /B-21 -- RDC(1,J)\n"
            B["B22"] = f"{' '.join(self.rdc2.astype(str))} /B-21 -- RDC(2,J)\n"
        if self.etsim:
            B["B23"] = f"{' '.join(self.ptval.astype(str))} /B-23 -- PTVAL\n"
            B["B24"] = f"{' '.join(self.rdc3.astype(str))} /B-21 -- RDC(3,J)\n"
            B["B25"] = f"{' '.join(self.rdc4.astype(str))} /B-21 -- RDC(4,J)\n"
            B["B26"] = f"{' '.join(self.rdc5.astype(str))} /B-21 -- RDC(5,J)\n"
            B["B27"] = f"{' '.join(self.rdc6.astype(str))} /B-21 -- RDC(6,J)\n"
        if self.f7p:
            B["B33"] = f"{self.numbf} {self.maxcells} /B-33 -- NUMBF, MAXCELLS\n"
            B["B34"] = f"{self.idbf} {self.numcells} /B-34 -- IDBF, NUMCELLS\n"
            B["B35"] = f"{self.bcrowncoln} (ROW(N),COL(N),N=1,NUMCELLS)\n"
        return B

    def write_C(self):
        C = OrderedDict()
        fs = f"0{str(list(self.bc.keys())[-1])}d"  # format specifier key
        for ky, bc in self.bc.items():
            C[
                f"{ky:{fs}}_C01"
            ] = f"1.0 {self.delt} /C-1 -- TPER, DELT (Recharge Period {ky})\n"
            C[
                f"{ky:{fs}}_C02"
            ] = f"{self.tmlt} {self.dltmx} {self.dltmin} {self.tred} /C-2 -- TMLT, DLTMX, DLTMIN, TRED\n"
            C[f"{ky:{fs}}_C03"] = f"{self.dsmax} {self.sterr} /C-3 -- DSMAX, STERR\n"
            C[f"{ky:{fs}}_C04"] = f"{bc['pond']} /C-4 -- POND\n"
            C[f"{ky:{fs}}_C05"] = f"{self.prnt} /C-5 -- PRNT\n"
            C_05 = [
                "T" if x else "F" for x in (bc["bcitrp"], bc["etsimrp"], bc["seep"])
            ]
            C[f"{ky:{fs}}_C06"] = f"{' '.join(C_05)} /C-6 -- BCIT, ETSIM, SEEP\n"
            if bc["seep"]:
                C[f"{ky:{fs}}_C07"] = f"{bc['nfcs']} /C-7 -- NFCS\n"
                C[
                    f"{ky:{fs}}_C08"
                ] = f"{bc['jj']} {bc['jlast']}/C-8 --  JJ, JLAST. C-9 begins next line: J, N\n"
                C[f"{ky:{fs}}_C09"] = ""
                for i in range(bc["jj"]):
                    C[f"{ky:{fs}}_C09"] += f"{bc['jtx'][i]} \n"
            C[f"{ky:{fs}}_C10"] = f"{bc['ibc']} /C-10 -- IBC\n"
            C[f"{ky:{fs}}_C11"] = ""
            for jjnnntx in bc["ntx"]:
                jj = int(jjnnntx[0] + 1)
                nn = int(jjnnntx[1] + 1)
                ntx = int(jjnnntx[2])
                pfdum = float(jjnnntx[3])
                C[
                    f"{ky:{fs}}_C11"
                ] += f"{jj} {nn} {ntx} {pfdum} /C-11 -- JJ, NN, NTX, PFDUM\n"
            C[
                f"{ky:{fs}}_C19"
            ] = f"-999999 /C-19 -- End of data for recharge period {ky}\n"
        C[f"{ky:{fs}}_C99"] = f"-999999 /End of input data file"
        return C

    def read(self, path="vs2drt.dat"):
        textures = {}
        bc = {}
        delt = []
        tmlt = []
        dltmx = []
        dltmin = []
        tred = []
        dsmax = []
        sterr = []
        with open(path, "r+") as fo:
            line = fo.readline()
            if line == "\n":
                title = ""
            else:
                title = line.split("/")[0]

            line = fo.readline()
            while line:
                ls = line.split("/")[0]
                if "/A-2 " in line:
                    tmax, stim, ang = ls.split()
                elif "/A-3 " in line:
                    zunit, tunit, cunx, hunx = ls.split()
                elif "/A-4 " in line:
                    nxr, nly = ls.split()
                elif "/A-5 " in line:
                    nrech, numt = ls.split()
                elif "/A-6 " in line:
                    rad, itstop, heat, solute = [
                        True if x == "T" else False for x in ls.split()
                    ]
                elif "/A-12" in line:
                    f11p, f7p, f8p, f9p, f6p = [
                        True if x == "T" else False for x in ls.split()
                    ]
                elif "/A-13" in line:
                    thpt, spnt, ppnt, hpnt, vpnt = [
                        True if x == "T" else False for x in ls.split()
                    ]
                elif "/A-14 " in line:
                    vals = ls.split()
                    ifac = int(vals[0])
                    facx = int(vals[1])
                    if " A-15 " in line:
                        dxr = np.array([])
                        line = fo.readline()
                        while ("/A-17 " not in line) and line:
                            if "/End" in line:
                                arr = np.array(line.split("/")[0].split())
                            else:
                                arr = np.array(line.split())
                            dxr = np.append(dxr, arr)
                            line = fo.readline()
                        else:
                            dxr = dxr.astype(float)
                            continue
                elif "/A-17 " in line:
                    vals = ls.split()
                    jfacx = int(vals[0])
                    facx = int(vals[1])
                    if " A-18 " in line:
                        delz = np.array([])
                        line = fo.readline()
                        while "/End A-18" not in line:
                            arr = np.array(line.split())
                            delz = np.append(delz, arr)
                            line = fo.readline()
                        else:
                            arr = np.array(line.split("/")[0].split())
                            delz = np.append(delz, arr).astype(float)
                            line = fo.readline()
                            continue
                elif "/A-20 " in line:
                    nplt = int(ls)
                    if " A-21 " in line:
                        pltim = np.array([])
                        line = fo.readline()
                        while "/A-24 " not in line:
                            if "/End" in line:
                                arr = np.array(line.split("/")[0].split())
                            else:
                                arr = np.array(line.split())
                                pltim = np.append(pltim, arr)
                            line = fo.readline()
                        else:
                            pltim = pltim.astype(float)
                            continue
                elif "/A-24 " in line:
                    nmb9 = int(ls)
                elif "/A-25" in line:
                    mb9 = np.array(ls.split()).astype(int)
                elif "/B-1 " in line:
                    vals = ls.split()
                    eps = float(vals[0])
                    hmax = float(vals[1])
                    wus = float(vals[2])
                elif "/B-4 " in line:
                    vals = ls.split()
                    minit = int(vals[0])
                    itmax = int(vals[1])
                elif "/B-5 " in line:
                    phrd = [True if x == "T" else False for x in ls]
                elif "/B-6 " in line:
                    vals = ls.split()
                    ntex = int(vals[0])
                    nprop = int(vals[1])
                elif "/B-7 " in line:
                    hft = int(ls)
                elif "/B-8 " in line:
                    itex = int(ls)
                    if " B-9 " in line:
                        line = fo.readline()
                        hk = np.array(line.split()).astype(float)  # B-9
                        textures[itex] = hk
                elif "/B-12 " in line:
                    irow = int(ls)
                    if " B-13" in line:
                        jtex = np.array([])
                        line = fo.readline()
                        while "/End B-13" not in line:
                            arr = np.array(line.split())
                            jtex = np.append(jtex, arr, axis=0)
                            line = fo.readline()
                        else:
                            arr = np.array(line.split("/")[0].split())
                            jtex = np.append(jtex, arr, axis=0).astype(int)
                            line = fo.readline()
                            continue
                elif "/B-15 " in line:
                    vals = ls.split()
                    iread = int(vals[0])
                    factor = float(vals[1])
                elif "/B-16 " in line:
                    vals = ls.split()
                    dwtx = float(vals[0])
                    hmin = float(vals[0])
                elif "/B-18 " in line:
                    bcit, etsim = [True if x == "T" else False for x in ls.split()]
                elif "/C-1 " in line:
                    jj = []
                    nn = []
                    ntx = []
                    pfdum = []
                    rp = int(line.split("(")[-1].split(")")[0].split()[-1])
                    bc[rp] = {}
                    bc[rp]["tper"] = ls.split()[0]
                    delt.append(ls.split()[1])
                    line = fo.readline()
                    while "-999999" not in line:
                        if "/C-2 " in line:
                            vals = line.split("/")[0].split()
                            tmlt.append(float(vals[0]))
                            dltmx.append(float(vals[1]))
                            dltmin.append(float(vals[2]))
                            tred.append(float(vals[3]))
                        elif "/C-3 " in line:
                            vals = line.split("/")[0].split()
                            dsmax.append(float(vals[0]))
                            sterr.append(float(vals[1]))
                        elif "/C-4 " in line:
                            bc[rp]["pond"] = float(line.split("/")[0])
                        elif "/C-5 " in line:
                            ls = line.split("/")[0].split()
                            bc[rp]["prnt"] = [True if x == "T" else False for x in ls]
                        elif "/C-6 " in line:
                            ls = line.split("/")[0].split()
                            bc[rp]["bcitrp"], bc[rp]["etsimrp"], bc[rp]["seep"] = [
                                True if x == "T" else False for x in ls
                            ]
                        elif "/C-7 " in line:
                            bc[rp]["nfcs"] = int(line.split("/")[0])
                        elif "/C-8 " in line:
                            ls = line.split("/")[0].split()
                            bc[rp]["jj"] = int(ls[0])
                            bc[rp]["jlast"] = int(ls[0])
                            if " C-9 " in line:
                                jspx = []
                                line = fo.readline()
                                while " /C-10 " not in line:
                                    j, n = line.split()
                                    jspx.append((int(j), int(n)))
                                    line = fo.readline()
                                else:
                                    continue
                        elif "/C-10 " in line:
                            ibc = int(line.split("/")[0])
                        elif "/C-11 " in line:
                            vals = line.split("/")[0].split()
                            jj.append(int(vals[0]))
                            nn.append(int(vals[1]))
                            ntx.append(int(vals[2]))
                            pfdum.append(float(vals[3]))
                        line = fo.readline()
                elif "-999999 /End" in line:
                    print("Reached end of file")
                else:
                    print(f"{line} ignored")
                line = fo.readline()


#%%
if __name__ == "__main__":
    # print(["T" if x else "F" for x in (True, True, False)])
    ml = Model("test", "test/test.exx")
    # ml.define_output(pltim=np.arange(1))
    # ml.define_domain()
    # ml.define_soil()
    # ml.define_initialc()
    # ml.define_solver()
    # bc = ml.get_bc()
    # ml.define_rp(bc={0: bc})
    # abc = ml.write_input()
    l = ml.read(path="vs2drt1.dat")
# print(l)

# %%
