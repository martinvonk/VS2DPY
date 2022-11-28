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
        self.bcit = None
        self.etsim = None
        self.npv = None
        self.etcyc = None
        self.peval = None
        self.rdc1 = None
        self.rdc2 = None
        self.ptval = None
        self.rdc3 = None
        self.rdc4 = None
        self.rdc5 = None
        self.rdc6 = None

        # define_boundaryc

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
    ):
        self.eps = eps  # B-1
        if 1.2 <= hmax <= 0.4:
            print(f"Relaxation parameter outside of general range")
        self.hmax = hmax  # B-1
        self.wus = wus  # B-1
        self.minit = minit  # B-4
        self.itmax = itmax  # B-4

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
            self.jtex = np.zeros((10, 1), dtype=int)
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
        self.bcit = bcit
        self.etsim = etsim
        if bcit or etsim:
            self.npv = npv
            self.etcyc = etcyc
        if bcit:
            for name, x in zip((peval, rdc1, rdc2), ("peval", "rdc1", "rdc2")):
                if len(x) != self.npv:
                    raise ValueError(f"Number of entries of {name} must equal NPV")
            self.peval = peval
            self.rdc1 = rdc1
            self.rdc2 = rdc2
        if etsim:
            for name, x in zip(
                (ptval, rdc3, rdc4, rdc5, rdc6), ("peval", "rdc1", "rdc2")
            ):
                if len(x) != self.npv:
                    raise ValueError(f"Number of entries of {name} must equal NPV")
            self.ptval = ptval
            self.rdc3 = rdc3
            self.rdc4 = rdc4
            self.rdc5 = rdc5
            self.rdc6 = rdc6

    def write_input(self):
        A = self.write_A()
        B = self.write_B()
        ABC = list(A.values()) + list(B.values())

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


if __name__ == "__main__":
    # print(["T" if x else "F" for x in (True, True, False)])
    ml = Model("test", "test/test.exx")
    ml.define_output(pltim=np.arange(1))
    ml.define_domain()
    ml.define_soil()
    ml.define_initialc()
    ml.define_solver()
    ABC = ml.write_input()
ABC
