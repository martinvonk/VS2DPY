#%%
import os
import numpy as np
from collections import OrderedDict
from subprocess import PIPE, STDOUT, Popen
from .read import var_out, bal_out


class Model:
    def __init__(
        self,
        ws: str,
        exe: str = "vs2drt",
        titl: str = "Model created with VS2DPY",
        tmax: float = 1.0,
        stim: float = 0.0,
        zunit: str = "m",
        tunit: str = "sec",
    ):
        """Initialize model

        Parameters
        ----------
        ws : str
            Workspace directory where files are stored
        exe : str, optional
            Executable path, by default 'vs2drt' which is the case
            if the executable is added to the
        titl : str, optional
            Description of the model, by default 'Model created with VS2DPY'
            String can't have more than 80 characters
        tmax : float, optional
            Maximum simulation time, by default 1.0
        stim : float, optional
            Initial time, by default 0.0
        zunit : str, optional
            Units used for length, by default 'm'
        tunit : str, optional
            Units used for time, by default 'sec' Other alternatives are
            'hour', 'day', 'year', etc.

        """

        if not os.path.exists(ws):
            os.mkdir(ws)
            print(f"Directory {ws} created")

        self.ws = ws

        if exe is None:
            self.exe = "C:/Program Files/USGS/vs2drti-1.6.0/bin/vs2drt.exe"
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
        self.itstop = None  # A-6
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
        nmb9: int = -33,
        mb9: np.ndarray = None,
        numbf: int = 0,
        maxcells: int = 0,
        idbf: int = 0,
        numcells: int = 0,
        bcrowncoln: list[tuple] = None,
        prnt: bool = False,
    ):
        """Define which output is provided

        file 6 = vs2drt.out
        file 7 = file07.out?
        file 8 = variables.out
        file 9 = balance.out?
        file 11 = obsPoints.out?

        Parameters
        ----------
        nrech : int, optional
            Number of recharge periods, by default 1 Set NRECH to a negative
            number to output binary values of head and concentration at
            selected observation times to file fort.12. Selecting this option
            allows the simulation to be restarted at any observation time;
            however, it may require a large amount of disk storage space.
        numt : int, optional
            Maximum number of time steps, by default 1 If enhanced precision in
            print out to file 9 and file 11 is desired, set NUMT equal to a
            negative number
        f11p : bool, optional
            Logical variable, if concentration, head, moisture content, and
            saturation at selected observation points are to be written to file
            11 at end of each time step, by default False
        f7p : bool, optional
            Logical variable, if fluxes through selected boundary faces are
            output to file07.out for each time step (boundary faces are
            specified on input lines B-33 to B-35), by default False
        f8p : bool, optional
            Logical variable, if output of pressure heads, concentrations, and
            temperatures to file 8 is desired at selected observation times;
            otherwise F8P=F, by default True
        f9p : bool, optional
            Logical variable, if one-line mass balance summary for each time
            step is to be written to file 9, by default True
        f6p : bool, optional
            Logical variable, if if mass balance is to be written to file
            6 for each time step; otherwise False if mass balance is to be
            written to file 6 only at observation times and ends of recharge
            periods, by default False
        thpt : bool, optional
            Logical variable, if volumetric moisture contents are to be
            written to file 6, by default False
        spnt : bool, optional
            Logical variable, if saturations are to be written to file 6, by
            default False
        ppnt : bool, optional
            Logical variable if pressure heads are to be written to file 6, by
            default False
        hpnt : bool, optional
            Logical variable, if total heads are to be written to file 6, by
            default False
        vpnt : bool, optional
            Logical variable, if velocities are to be written to file 6, by
            default False
        nplt : int, optional
            Number of elapsed times at which to write pressure heads,
            temperatures, and concentrations to file 8 and heads, temperatures,
            concentrations, saturations, moisture contents, and/or velocities
            to file 6, by default 1
        pltim : np.ndarray, optional
            Elapsed times at which pressure heads, temperatures, and
            concentrations are written to file 8, and heads, concentrations,
            temperatures, saturations, velocities, and/or moisture contents to
            file 6, by default None
        nobs : int, optional
            Number of observation points for which heads, temperatures,
            concentrations, moisture contents, and saturations are to be
            written to file 11. Set NOBS equal to a negative number if output
            to file 11 is desired only at selected output times rather than at
            each time step, by default 0
        obsrowncoln : list[tuple], optional
            Row and column number for each observation, by default None
        nmb9 : int, optional
            Total number of mass balance components written to file 9; number
            must be less than 73. Set NMB9 equal to a negative number if output
            to file 9 is desired only at selected output times rather than at
            each time step, by default -33
        mb9 : np.ndarray, optional
            The index number of each mass balance component to be written to
            file 9, by default None
        numbf : int, optional
            Number of boundary faces for which fluxes will be calculated and
            output to file file07.out, by default 0
        maxcells : int, optional
            Maximum number of cells on any boundary face, by default 0
        idbf : int, optional
            Boundary face identifier (integer), by default 0
        numcells : int, optional
            Number of finite difference cells on this boundary face, by default 0
        bcrowncoln : list[tuple], optional
            Row and column number of each cell on this boundary face, by default None
        prnt : bool, optional
            Logical variable, if heads, temperatures, concentration, moisture
            contents, and (or) saturations are to be printed to file 6 after
            each time step; PRNT=F if they are to be written to file 6 only at
            observation times and ends of recharge periods, by default False

        """

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
                mb9 = np.arange(1, np.abs(nmb9) + 1)
            if len(mb9) != np.abs(nmb9):
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
        """Define domain and grid spacing

        Parameters
        ----------
        nxr : int, optional
            Number of cells in horizontal or radial direction, by default 1
        nly : int, optional
            Number of cells in vertical direction, by default 1
        dxr : np.ndarray, optional
            Grid spacing in horizontal or radial direction. Number of entries
            must equal NXR, L, by default None
        delz : np.ndarray, optional
            Grid spacing in vertical direction; number of entries must equal
            NLY, L, by default None

        """
        self.nxr = nxr  # A-4
        self.nly = nly  # A-4
        if dxr is None:  # ifac = 0
            self.dxr = np.full((nxr), 1 / nxr)  # A-15
        else:
            if len(dxr) != nxr:
                raise ValueError("Number of entries must be equal to NXR")
            self.dxr = dxr  # A-15
        if delz is None:  # jfac = 0
            self.delz = np.full((nly), 1 / nly)
        else:
            if len(delz) != nly:
                raise ValueError("Number of entries must equal NLY")
            self.delz = delz  # A-18

    def define_solver(
        self,
        itstop: bool = True,
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
        """Define solving attributes

        Parameters
        ----------
        itstop : bool, optional
            Logical variable, if simulation is to terminate after
            ITMAX iterations in one time step, by default True
        eps : float, optional
            Head closure criterion for iterative solution of flow equation, by
            default 0.0001
        hmax : float, optional
            Relaxation parameter for iterative solution. Value is generally in
            the range of 0.4 to 1.2, by default 0.7
        wus : float, optional
            Weighting option for inter-cell relative hydraulic conductivity:
            WUS=1 for full upstream weighting. WUS=0.5 for arithmetic mean.
            WUS=0.0 for geometric mean, by default 0.5
        minit : int, optional
            Minimum number of iterations per time step, by default 2
        itmax : int, optional
            Maximum number of iterations per time step, by default 10
        delt : float, optional
            Length of initial time step for this period, T, by default 0.1
        tmlt : float, optional
            Multiplier for time step length, by default 1.5
        dltmx : float, optional
            Maximum allowed length of time step, by default 1.0
        dltmin : float, optional
            Minimum allowed length of time step, by default 0.0001
        tred : float, optional
            Factor by which time-step length is reduced if convergence is not
            obtained in ITMAX iterations. Values usually should be in the range
            0.1 to 0.5. If no reduction of time-step length is desired, input a
            value of 0.0, by default 0.01
        dsmax : float, optional
            Maximum allowed change in head per time step for this period, by
            default 100
        sterr : float, optional
            Steady-state head criterion; when the maximum change in head
            between successive time steps is less than STERR, the program
            assumes that steady state has been reached for this period and
            advances to next recharge period, by default 0.0
        """

        self.itstop = itstop  # A-6
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
        textures: dict = {1: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        jtex: np.ndarray = None,
    ):
        """Define soil attributes

        Parameters
        ----------
        ntex : int, optional
            Number of textural classes or lithologies having different values
            of hydraulic conductivity, specific storage, and (or) constants in
            the functional relations among pressure head, relative
            conductivity, and moisture content, by default 1
        nprop : int, optional
            Number of flow properties to be read in for each textural class.
            When using Brooks and Corey, van Genuchten or Nimmo-Rossi
            functions, set NPROP=6; when using Haverkamp functions, set
            NPROP=8, by default 6
        hft : int, optional
            Hydraulic function type, HFT=0 for Brooks-Corey; HFT=1 for van
            Genuchten; HFT=2 for Haverkamp; HFT=3 for tabular data; and HFT=4
            for Rossi-Nimmo, by default 1 (van Genuchten)
        textures : dict, optional
            Dictionary with textures, by default {1: np.array([1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0])} which is the inactive cell texture. The key
            of the dictionary has to be an intiger (ITEX) with the texture as
            defined in jtex. Definitions for the remaining sequential values
            are dependent upon which functional relation is selected to
            represent the nonlinear coefficients. Five different functional
            relations are allowed as defined by HFT: (0) Brooks-Corey, (1) van
            Genuchten, (2) Haverkamp, and (4) Rossi-Nimmo. In the following
            descriptions, definitions for the different functional relations
            are indexed by the above numbers.The value of the dictionary has to
            be an numpy array (HK(ITEX,entry)) in the following order:
            entry 1:
            Saturated hydraulic conductivity (K) in the x-coordinate direction
            for class ITEX
            entry 2:
            Specific storage (Ss) for class ITEX
            entry 3:
            Porosity (θs) for class ITEX. Must be larger than 0.
            entry 4:
            (0) hb, Brooks-Corey bubbling pressure head (must be less than 0);
            (1) alpha, van Genuchten alpha as defined by van Genuchten(1980);
            (2) A', Haverkamp parameter (must be less than 0.0),;
            (4) Ψ0, Rossi-Nimmo parameter;
            entry 5:
            (0) Residual moisture content (θr).
            (1) Residual moisture content (θr).
            (2) Residual moisture content (θr).
            (4) ΨD, Rossi-Nimmo parameter.
            entry 6:
            (0) λ, Brooks-Corey pore-size distribution index;
            (1) n, van Genuchten parameter, β' in Healy (1990) and Lappala
            and others (1987);
            (2) B', Haverkamp parameter;
            (4) λ, Rossi-Nimmo parameter;
            entry 7:
            (0) Not used;
            (1) Not used;
            (2) alpha, Haverkamp parameter (must be less than 0.0);
            (4) Not used;
            entry 8:
            (0) Not used;
            (1) Not used;
            (2) β, Haverkamp parameter;
            (4) Not used;
        jtex : np.ndarray, optional
            Indices for textural class for each node, read in row by row. There
            must be NXR*NLY entries, by default None

        """
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
        iread: int = 0,
        factor: float = 1.0,
        dwtx: float = None,
        hmin: float = None,
    ):
        """Define initial conditions

         Parameters
         ----------
         phrd : bool, optional
             Logical variable, if initial conditions are read in as pressure heads; PHRD=F
             if initial conditions are read in as moisture contents, by default True
        iread : int, optional
             If IREAD=0, all initial conditions in terms of pressure head or
             moisture content as determined by the value of PHRD are set equal
             to FACTOR. If IREAD=2 initial conditions are defined in terms of
             pressure head, and an equilibrium profile is specified above a free
             -water surface at a depth of DWTX until a pressure head of HMIN is
             reached. All pressure heads above this are set to HMIN, by default
             0
         factor : float, optional
             Multiplier or constant value, depending on value of IREAD, for
             initial conditions, by default 1.0
         dwtx : float, optional
             Depth to free-water surface above which an equilibrium profile is
             computed, by default None
         hmin : float, optional
             Minimum pressure head to limit height of equilibrium profile. Must
             be negative, by default None

        """
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
        rdc1: np.ndarray = None,
        rdc2: np.ndarray = None,
        ptval: np.ndarray = None,
        rdc3: np.ndarray = None,  # RD
        rdc4: np.ndarray = None,  # RAbase
        rdc5: np.ndarray = None,  # RAtop
        rdc6: np.ndarray = None,  # Hroot
    ):
        """Define evaporation attributes

        Parameters
        ----------
        bcit : bool, optional
            Logical variable, if evaporation is to be simulated at any time
            during the simulation, by default False
        etsim : bool, optional
            Logical variable, ETSIM=T if evapotranspiration (plant-root
            extraction) is to be simulated at any time during the simulation,
            by default False
        npv : int, optional
            Number of ET periods to be simulated. NPV values for each variable
            ETSIM=T required for the evaporation and (or) evapotranspiration
            options must be entered on the following lines. If ET variables are
            held constant throughout the simulation code, NPV = 1, by default
            None
        etcyc : float, optional
            Length of each ET period, by default None
        peval : np.ndarray, optional
            Potential evaporation rate (PEV) at beginning of each ET period.
            Number of entries must equal NPV. To conform with the sign
            convention used in most existing equations for potential
            evaporation, all entries must be greater than or equal to 0. The
            program multiplies all nonzero entries by -1 so that the
            evaporative flux is treated as a sink rather than a source, by
            default None
        rdc1 : np.ndarray, optional
            Surface resistance to evaporation (SRES) at beginning of ET period.
            For a uniform soil, SRES is equal to the reciprocal of the distance
            from the top active node to land surface, or 2/DELZ(2). If a
            surface crust is present, SRES may be decreased to account for the
            added resistance to water movement through the crust. Number of
            entries must equal NPV, by default None
        rdc2 : np.ndarray, optional
            Pressure potential of the atmosphere (HA) at beginning of each ET
            period; may be estimated using equation 6 of Lappala and others
            (1987) Number of entries must equal NPV, by default None
        peval : np.ndarray, optional
            Potential evapotranspiration rate (PET) at beginning of each ET
            period, L/T. Number of entries must equal NPV. As with PEV, all
            values must be greater than or equal to 0, by default None
        rdc3 : np.ndarray, optional
            Rooting depth (RD) at beginning of each ET period. Number of
            entries must equal NPV, by default None
        rdc4 : np.ndarray, optional
            Root activity (RAbase) at base of root zone at beginning of each ET
            period Number of entries must equal NPV, by default None
        rdc5 : np.ndarray, optional
            Root activity (RAtop) at top of root zone at beginning of each ET
            period, Number of entries must equal NPV. Note: Values for root
            activity generally are determined empirically, but typically range
            from 0 to 3x10^4 m/m3 . As programmed, root activity varies
            linearly from land surface to the base of the root zone, and its
            distribution with depth at any time is represented by a trapezoid.
            In general, root activities will be greater at land surface than at
            the base of the root zone, by default None
        rdc6 : np.ndarray, optional
            Pressure head in roots (HROOT) at beginning of each ET period.
            Number of entries must equal NPV, by default None

        """
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
        seepf: dict = {},
        # jj: int = 1,
        # jlast: int = 0,
        # jspx: list[tuple] = None,
        ibc: int = 0,
        ntx: np.ndarray = None,
    ):
        """Create boundary condtition dictionary for a recharge period

        Parameters
        ----------
        bcitrp : bool, optional
            Logical variable, if evaporation is to be simulated for this
            recharge period, by default False
        etsimrp : bool, optional
            Logical variable, if evapotranspiration (plant-root extraction) is
            to be simulated for this recharge period, by default False
        seep : bool, optional
            Logical variable, if seepage faces are to be simulated for this
            recharge period, by default False
        pond : float, optional
            Maximum allowed height of ponded water for constant flux nodes, by
            default 0.0
        nfcs : int, optional
            Number of possible seepage faces, by default 1
        seepf : dict, optional
            Seepage face dictionary containing the JJ, JLAST and JSPX for each
            seepage face. Per possible seepage face, the key has to be updated
            such that the dictionary has "jj_0", "jj_1", ..., "jj_n" for n =
            NFCS. The same holds for "jlast_n" and "jspx_n" where the value of
            seepf["jspx_n"] has to be a list of tuples for the seepage face
            nodes, by default {}
        ibc : int, optional
            Code for reading in boundary conditions by individual node (IBC=0)
            or by row or column (IBC=1). Only one code may be used for each
            recharge period, and all boundary conditions for period must be
            input in the sequence for that code., by default 0
        ntx : np.ndarray, optional

            Numpy array with four columns for JJ, NN, NTX and PFDUM. Each row
            is a node where a boundary condition is defined, by default None JJ
            is the row number. NN is the row column, NTX is the boundary
            condition with:
            NTX=0 for no specified boundary (needed for resetting some nodes
            after initial recharge period);
            NTX=1 for specified pressure head;
            NTX=2 for specified flux per unit horizontal surface area [L/T];
            NTX=3 for possible seepage face;
            NTX=4 for specified total head;
            NTX=5 for evaporation;
            NTX=6 for specified volumetric flow in units [L3/T];
            NTX=7 for gravity drain. (The gravity drain boundary condition
            allows gravity driven vertical flow out of the domain assuming a
            unit vertical hydraulic gradient. Flow into the domain cannot
            occur.)
            PFDUM is the specified head for NTX=1 or 4 or specified flux for
            NTX=2 or 6. If codes 0, 3, 5, or 7 are specified, the line should
            contain a dummy value for PFDUM.

        Returns
        -------
        dictionary
            Dictionary with the boundary condition of a recharge period

        """
        bc = {}
        bc["pond"] = pond  # C-4
        bc["bcitrp"] = bcitrp  # C-6
        bc["etsimrp"] = etsimrp  # C-6
        bc["seep"] = seep  # C-6
        if seep:
            bc["nfcs"] = nfcs  # C-6
            bc["seepf"] = seepf
            seepfkeys = list(seepf.keys())
            if len(seepfkeys) != nfcs:
                raise ValueError(f"Number of entries of SEEPF must be equal to JJ")
            for x in range(nfcs):
                kys = (f"jj_{x}", f"jlast_{x}", f"jspx_{x}")
                for ky in kys:
                    if ky not in seepfkeys:
                        raise ValueError(f"{ky} must be in in SEEPF dictionary")
                if len(seepf[kys[2]]) != seepf[kys[0]]:
                    raise ValueError(
                        f"Number of entries of JSPX  in SEEPF dictionary must be equal to JJ"
                    )
            # bc["seepf"]["jj"] = jj  # C-6
            # bc["seepf"]["jlast"] = jlast  # C-6
            # bc["jspx"] = jspx
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
        """Define recharge period

        Parameters
        ----------
        tper : np.ndarray, optional
            Lenght of this recharge period, by default np.array([1.0])
        bc : dict[dict], optional
            Dictionary timestep as key and boundary condition dicitionary,
            created with create_bc() function as values boundary conditions, by
            default None

        """
        self.tper = tper  # C-1
        if len(tper) != len(bc):
            raise ValueError(
                f"Number of entries of TPER must be equal to boundary conditions"
            )
        if bc is None:
            self.bc = self.create_bc()
        else:
            self.bc = bc  # C-4 - C-11

    def write_input(self, ignore_settings=True):
        """Generate vs2drt.dat and vs2drt.fil input files

        Parameters
        ----------
        ignore_settings : bool, optional
            Ignores settings for vs2drt.fil input file and outputs all
            filetypes, by default True
        """
        # vs2drt.dat file
        A = self.write_A()
        B = self.write_B()
        C = self.write_C()
        ABC = list(A.values()) + list(B.values()) + list(C.values())
        with open(f"{self.ws}/vs2drt.dat", "w") as fo:
            fo.writelines(ABC)
        # vs2drt.fil file
        fil = self.write_fil(ignore_settings=ignore_settings)
        with open(f"{self.ws}/vs2drt.fil", "w") as fo:
            fo.writelines(fil)

    def write_A(self) -> OrderedDict:
        """Write part A of vs2drt.dat input file

        Returns
        -------
        OrderedDict

        """
        A = OrderedDict()
        A["A01"] = f"{self.titl}\n"
        A["A02"] = f"{self.tmax} {self.stim} 0. /A-2 -- TMAX, STIM, ANG\n"
        A[
            "A03"
        ] = f"{self.zunit}   {self.tunit} g   J /A-3 -- ZUNIT, TUNIT, CUNX, HUNX\n"
        A["A04"] = f"{self.nxr} {self.nly} /A-4 -- NXR, NLY\n"
        A["A05"] = f"{self.nrech} {self.numt} /A-5 -- NRECH, NUMT\n"
        A_06 = ["F"] + ["T" if x else "F" for x in (self.itstop,)] + ["F", "F"]
        A["A06"] = f"{' '.join(A_06)} /A-6 -- RAD, ITSTOP, HEAT, SOLUTE\n"
        A_12 = [
            "T" if x else "F"
            for x in (self.f11p, self.f7p, self.f8p, self.f9p, self.f6p)
        ]
        A["A12"] = " ".join(A_12) + " /A-12 -- F11P, F7P, F8P, F9P, F6P\n"
        A_13 = [
            "T" if x else "F"
            for x in (self.thpt, self.spnt, self.ppnt, self.hpnt, self.vpnt)
        ]
        A["A13"] = " ".join(A_13) + " /A-13 -- THPT, SPNT, PPNT, HPNT, VPNT\n"
        A["A14"] = f"0 1 /A-14 -- IFAC, FACX. A-15 begins next line: DXR\n"
        A["A15"] = f"{' '.join(self.dxr.astype(str))} \n"
        A["A17"] = f"0 1 /A-17 -- JFAC, FACZ. A-18 begins next line: DELZ\n"
        A["A18"] = f"{' '.join(self.delz.astype(str))} /End A-18\n"
        if self.f8p:
            A["A20"] = f"{self.nplt} /A-20 -- NPLT. A-21 begins next line: PLTIM\n"
            A["A21"] = f"{' '.join(self.pltim.astype(str))}\n"
        if self.f11p:
            A["A22"] = f"{self.nobs} /A-22 -- NOBS. A-23 begins next line: J, N\n"
            A["A23"] = ""
            for jn in self.obsrowncoln:
                A["A23"] += f"{jn[0]} {jn[1]}\n"
        if self.f9p:
            A["A24"] = f"{self.nmb9} /A-24 -- NMB9\n"
            A["A25"] = f"{' '.join((self.mb9).astype(str))} /A-25 -- MB9\n"
        return A

    def write_B(self) -> OrderedDict:
        """Write part B of vs2drt.dat input file

        Returns
        -------
        OrderedDict

        """
        B = OrderedDict()
        B["B01"] = f"{self.eps} {self.hmax} {self.wus} /B-1 -- EPS, HMAX, WUS\n"
        B["B04"] = f"{self.minit} {self.itmax} /B-4 -- MINIT, ITMAX\n"
        B["B05"] = f"{['T' if x else 'F' for x in (self.phrd,)][0]} /B-5 -- PHRD\n"
        B["B06"] = f"{self.ntex} {self.nprop} /B-6 -- NTEX, NPROP\n"
        B["B07"] = f"{self.hft} /B-7 -- HFT hydraulicFunctionType\n"
        B["B08"] = ""  # also B09
        for ky in self.textures:
            B["B08"] += f"{ky} /B-8 -- ITEX. B-9 to begin next line: HK\n"
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

    def write_C(self) -> OrderedDict:
        """Write part C of vs2drt.dat input file

        Returns
        -------
        OrderedDict

        """
        C = OrderedDict()
        fs = f"0{str(list(self.bc.keys())[-1])}d"  # format specifier key
        for ky, bc in self.bc.items():
            C[
                f"{ky:{fs}}_C01"
            ] = f"{self.tper[ky-1]} {self.delt} /C-1 -- TPER, DELT (Recharge Period {ky})\n"
            C[
                f"{ky:{fs}}_C02"
            ] = f"{self.tmlt} {self.dltmx} {self.dltmin} {self.tred} /C-2 -- TMLT, DLTMX, DLTMIN, TRED\n"
            C[f"{ky:{fs}}_C03"] = f"{self.dsmax} {self.sterr} /C-3 -- DSMAX, STERR\n"
            C[f"{ky:{fs}}_C04"] = f"{bc['pond']} /C-4 -- POND\n"
            C_05 = ["T" if x else "F" for x in (self.prnt,)]
            C[f"{ky:{fs}}_C05"] = f"{' '.join(C_05)} /C-5 -- PRNT\n"
            C_06 = [
                "T" if x else "F" for x in (bc["bcitrp"], bc["etsimrp"], bc["seep"])
            ]
            C[f"{ky:{fs}}_C06"] = f"{' '.join(C_06)} /C-6 -- BCIT, ETSIM, SEEP\n"
            if bc["seep"]:
                C[
                    f"{ky:{fs}}_C07"
                ] = f"{bc['nfcs']} /C-7 -- NFCS\n"  # moet hier iets mee, loopen door aantal seepage faces
                for k in range(bc["nfcs"]):
                    C[
                        f"{ky:{fs}}_C08_{k}"
                    ] = f"{bc['seepf'][f'jj_{k}']} {bc['seepf'][f'jlast_{k}']} /C-8 -- JJ, JLAST. C-9 begins next line: J, N\n"
                    C[f"{ky:{fs}}_C09_{k}"] = ""
                    for i in range(bc["seepf"][f"jj_{k}"]):
                        vals = bc["seepf"][f"jspx_{k}"][i]
                        C[f"{ky:{fs}}_C09_{k}"] += f"{vals[0]} {vals[1]}\n"
            C[f"{ky:{fs}}_C10"] = f"{bc['ibc']} /C-10 -- IBC\n"
            C[f"{ky:{fs}}_C11"] = ""
            for jjnnntx in bc["ntx"]:
                jj = int(jjnnntx[0])
                nn = int(jjnnntx[1])
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

    def write_fil(self, ignore_settings=True) -> list:
        """Write input for vs2drt.fil file

        Parameters
        ----------
        ignore_settings : bool, optional
            Ignores settings for vs2drt.fil input file and outputs all
            filetypes, by default True

        Returns
        -------
        list

        """
        lines = ["vs2drt.dat\n", "vs2drt.out\n"]
        if self.f7p or ignore_settings:
            lines.append("file07.out\n")
        if self.f8p or ignore_settings:
            lines.append("variables.out\n")
        if self.f9p or ignore_settings:
            lines.append("balance.out\n")
        if self.f11p or ignore_settings:
            lines.append("obsPoints.out\n")
        lines.append("# vs2drt1.1\n")
        return lines

    def read(self, path="vs2drt.dat"):
        """Read vs2drt.dat file and add attributes to model

        Parameters
        ----------
        path : str, optional
            Path to vs2drt.dat file, by default "vs2drt.dat"

        """
        textures = {}
        bc = {}
        seepf = None
        tper = []
        delt = []
        tmlt = []
        dltmx = []
        dltmin = []
        tred = []
        dsmax = []
        sterr = []
        prnt = []
        with open(path, "r+") as fo:
            line = fo.readline()
            if line == "\n":
                self.title = ""
            else:
                self.title = line.split("/")[0]

            line = fo.readline()
            while line:
                ls = line.split("/")[0]
                if "/A-2 " in line:
                    vals = ls.split()
                    self.tmax = float(vals[0])
                    self.stim = float(vals[1])
                    # _ = float(vals[2])  # ang
                elif "/A-3 " in line:
                    self.zunit, self.tunit, _, _ = ls.split()
                elif "/A-4 " in line:
                    vals = ls.split()
                    self.nxr = int(vals[0])
                    self.nly = int(vals[1])
                elif "/A-5 " in line:
                    vals = ls.split()
                    self.nrech = int(vals[0])
                    self.numt = int(vals[1])
                elif "/A-6 " in line:
                    _, self.itstop, _, _ = [
                        True if x == "T" else False for x in ls.split()
                    ]
                elif "/A-12" in line:
                    self.f11p, self.f7p, self.f8p, self.f9p, self.f6p = [
                        True if x == "T" else False for x in ls.split()
                    ]
                elif "/A-13" in line:
                    self.thpt, self.spnt, self.ppnt, self.hpnt, self.vpnt = [
                        True if x == "T" else False for x in ls.split()
                    ]
                elif "/A-14 " in line:
                    vals = ls.split()
                    ifac = int(vals[0])
                    facx = int(vals[1])
                    if ifac != 0:
                        raise NotImplementedError
                    if facx != 1:
                        raise NotImplementedError
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
                            self.dxr = dxr.astype(float)
                            continue
                elif "/A-17 " in line:
                    vals = ls.split()
                    jfac = int(vals[0])
                    facz = int(vals[1])
                    if jfac != 0:
                        raise NotImplementedError
                    if facz != 1:
                        raise NotImplementedError
                    if " A-18 " in line:
                        delz = np.array([])
                        line = fo.readline()
                        while "/End A-18" not in line:
                            arr = np.array(line.split())
                            delz = np.append(delz, arr)
                            line = fo.readline()
                        else:
                            arr = np.array(line.split("/")[0].split())
                            self.delz = np.append(delz, arr).astype(float)
                            line = fo.readline()
                            continue
                elif "/A-20 " in line:
                    self.nplt = int(ls)
                    if " A-21 " in line:
                        pltim = np.array([])
                        line = fo.readline()
                        while "/" not in line:
                            arr = np.array(line.split())
                            pltim = np.append(pltim, arr)
                            line = fo.readline()
                        else:
                            self.pltim = pltim.astype(float)
                            continue
                elif "/A-22" in line:
                    self.nobs = int(ls)
                    if " A-23" in line:
                        obsrowncoln = []
                        line = fo.readline()
                        while "/" not in line:
                            vals = line.split()
                            obsrowncoln.append((int(vals[0]), int(vals[1])))
                            line = fo.readline()
                        else:
                            self.obsrowncoln = obsrowncoln
                            continue
                elif "/A-24 " in line:
                    self.nmb9 = int(ls)
                elif "/A-25" in line:
                    self.mb9 = np.array(ls.split()).astype(int)
                elif "/B-1 " in line:
                    vals = ls.split()
                    self.eps = float(vals[0])
                    self.hmax = float(vals[1])
                    self.wus = float(vals[2])
                elif "/B-4 " in line:
                    vals = ls.split()
                    self.minit = int(vals[0])
                    self.itmax = int(vals[1])
                elif "/B-5 " in line:
                    self.phrd = [True if x == "T" else False for x in ls]
                elif "/B-6 " in line:
                    vals = ls.split()
                    self.ntex = int(vals[0])
                    self.nprop = int(vals[1])
                elif "/B-7 " in line:
                    self.hft = int(ls)
                elif "/B-8 " in line:
                    itex = int(ls)
                    if " B-9 " in line:
                        line = fo.readline()
                        hk = np.array(line.split()).astype(float)  # B-9
                        textures[itex] = hk
                elif "/B-12 " in line:
                    self.irow = int(ls)
                    if " B-13" in line:
                        jtex = None
                        line = fo.readline()
                        while "/End B-13" not in line:
                            arr = np.array([line.split()])
                            if jtex is None:
                                jtex = arr
                            else:
                                jtex = np.append(jtex, arr, axis=0)
                            line = fo.readline()
                        else:
                            arr = np.array([line.split("/")[0].split()])
                            self.jtex = np.append(jtex, arr, axis=0).astype(int)
                            line = fo.readline()
                            continue
                elif "/B-15 " in line:
                    vals = ls.split()
                    self.iread = int(vals[0])
                    self.factor = float(vals[1])
                elif "/B-16 " in line:
                    vals = ls.split()
                    self.dwtx = float(vals[0])
                    self.hmin = float(vals[1])
                elif "/B-18 " in line:
                    self.bcit, self.etsim = [
                        True if x == "T" else False for x in ls.split()
                    ]
                elif "/C-1 " in line:
                    rp = int(line.split("(")[-1].split(")")[0].split()[-1])
                    bc[rp] = {}
                    bc[rp]["ntx"] = []
                    tper.append(float(ls.split()[0]))
                    delt.append(float(ls.split()[1]))
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
                            prnt.append([True if x == "T" else False for x in ls][0])
                        elif "/C-6 " in line:
                            ls = line.split("/")[0].split()
                            bc[rp]["bcitrp"], bc[rp]["etsimrp"], bc[rp]["seep"] = [
                                True if x == "T" else False for x in ls
                            ]
                        elif "/C-7 " in line:
                            bc[rp]["nfcs"] = int(line.split("/")[0])
                            seepf = {}
                            k = 0
                        elif "/C-8 " in line:
                            ls = line.split("/")[0].split()
                            seepf[f"jj_{k}"] = int(ls[0])
                            seepf[f"jlast_{k}"] = int(ls[1])
                            seepf[f"jspx_{k}"] = []
                            line = fo.readline()  # go to C-9
                            while "/C-10 " not in line:
                                if "/C-8 " in line:
                                    k += 1
                                    ls = line.split("/")[0].split()
                                    seepf[f"jj_{k}"] = int(ls[0])
                                    seepf[f"jlast_{k}"] = int(ls[1])
                                    seepf[f"jspx_{k}"] = []
                                    line = fo.readline()  # go to C-9
                                vals = line.split()
                                j = int(vals[0])
                                n = int(vals[1])
                                seepf[f"jspx_{k}"].append((j, n))
                                line = fo.readline()
                            else:
                                bc[rp]["seepf"] = seepf
                                continue
                        elif "/C-10 " in line:
                            bc[rp]["ibc"] = int(line.split("/")[0])
                        elif "/C-11 " in line:
                            vals = line.split("/")[0].split()
                            bc[rp]["ntx"].append(
                                (
                                    int(vals[0]),
                                    int(vals[1]),
                                    int(vals[2]),
                                    float(vals[3]),
                                )
                            )
                        line = fo.readline()
                elif "-999999 /End" in line:
                    print("Reached end of file")
                else:
                    print(f"{line} ignored")
                line = fo.readline()
        self.textures = textures
        self.bc = bc
        self.tper = np.array(tper)
        if len(np.unique(delt)) != 1:
            self.delt = delt
        else:
            self.delt = delt[0]
        if len(np.unique(tmlt)) != 1:
            self.tmlt = tmlt
        else:
            self.tmlt = tmlt[0]
        if len(np.unique(dltmx)) != 1:
            self.dltmx = dltmx
        else:
            self.dltmx = dltmx[0]
        if len(np.unique(dltmin)) != 1:
            self.dltmin = dltmin
        else:
            self.dltmin = dltmin[0]
        if len(np.unique(tred)) != 1:
            self.tred = tred
        else:
            self.tred = tred[0]
        if len(np.unique(dsmax)) != 1:
            self.dsmax = dsmax
        else:
            self.dsmax = dsmax[0]
        if len(np.unique(sterr)) != 1:
            self.sterr = sterr
        else:
            self.sterr = sterr[0]
        if len(np.unique(prnt)) != 1:
            self.prnt = prnt
        else:
            self.prnt = prnt[0]

    def run_model(
        self,
        silent=False,
        report=True,
    ):
        """
        This function will run the model using subprocess.Popen. It
        communicates with the model's stdout asynchronously and reports
        progress to the screen.

        Parameters
        ----------
        silent : boolean
            Echo run information to screen (default is True).
        report : boolean, optional
            Save stdout lines to a list (buff) which is returned
            by the method . (default is True).

        Returns
        -------
        (success, buff)
        success : boolean
        buff : list of lines of stdout
        """
        success = True
        buff = []

        # convert normal_msg to a list of lower case str for comparison
        normal_msg = ["Simulation terminated"]

        # create a list of arguments to pass to Popen
        argv = []
        # argv.append(f"cd {self.ws}")
        argv.append(self.exe)

        # run the model with Popen
        proc = Popen(argv, stdout=PIPE, stderr=STDOUT, cwd=self.ws)

        while True:
            line = proc.stdout.readline().decode("utf-8")
            if line:
                for msg in normal_msg:
                    if msg in line.lower():
                        success = False
                line = line.rstrip("\r\n")
                if not silent:
                    print(line)
                if report:
                    buff.append(line)
            else:
                break
        return success, buff

    def read_var_out(self):
        return var_out(path=f"{self.ws}/variables.out")

    def read_bal_out(self):
        return bal_out(path=f"{self.ws}/balance.out")
