import numpy as np
from pandas import Series


def get_gwt_1D(pressure_head: np.ndarray, depth: np.ndarray) -> float:
    sign = np.signbit(pressure_head)
    if sign.any() == False:
        gwl = depth[0]
    elif sign.sum() == len(pressure_head):
        gwl = depth[-1]
    else:
        idx = np.where(np.diff(sign))[0]
        if len(idx) > 1:
            idx = idx[0]
        gwl = (0 - pressure_head[idx + 1]) * (depth[idx] - depth[idx + 1]) / (
            pressure_head[idx] - pressure_head[idx + 1]
        ) + depth[idx + 1]
    return gwl


def get_gwt_2D(data: np.ndarray, z: np.ndarray) -> np.ndarray:
    gwt = np.array([])
    for i in range(data.shape[1]):
        gwt = np.append(gwt, get_gwt_1D(np.flip(data[:, i]), z))
    return gwt


class ptf:
    def __init__(
        self,
        sand: float,
        silt: float,
        clay: float,
    ):

        if sand is None:
            if silt and clay is not None:
                sand = 100 - silt - clay
            else:
                raise ValueError("Sand percentage 'sand' could not be calculated")
        if silt is None:
            if sand and clay is not None:
                silt = 100 - sand - clay
            else:
                raise ValueError("Silt percentage 's' could not be calculated")
        if clay is None:
            if sand and silt is not None:
                clay = 100 - sand - silt
            else:
                raise ValueError("Clay percentage 'c' could not be calculated")

        self.sand = sand  # sand percentage
        self.silt = silt  # silt percentage
        self.clay = clay  # clay percentage

    def schaap(
        self,
        d: float = None,
        th33: float = None,
        th1500: float = None,
        version: int = 3,
    ):
        from rosetta import rosetta, SoilData

        sd = SoilData.from_array([[self.sand, self.silt, self.clay, d, th33, th1500]])
        theta_r, theta_s, alpha_, n_, ks_ = tuple(rosetta(version, sd)[0][0])
        l = 0.5
        return theta_r, theta_s, np.exp(ks_), np.exp(alpha_), np.exp(n_), l

    def wosten(self, om: float, d: float, ts: bool = False):
        topsoil = 1.0 * ts

        theta_s = (
            0.7919
            + 0.001691 * self.clay
            - 0.29619 * d
            - 0.000001419 * self.silt**2
            + 0.0000821 * om**2
            + 0.02427 * self.clay**-1
            + 0.01113 * self.silt**-1
            + 0.01472 * np.log(self.silt)
            - 0.0000733 * om * self.clay
            - 0.000619 * d * self.clay
            - 0.001183 * d * om
            - 0.0001664 * topsoil * self.silt
        )
        alpha_ = (
            -14.96
            + 0.03135 * self.clay
            + 0.0351 * self.silt
            + 0.646 * om
            + 15.29 * d
            - 0.192 * topsoil
            - 4.671 * d**2
            - 0.000781 * self.clay**2
            - 0.00687 * om**2
            + 0.0449 * om**-1
            + 0.0663 * np.log(self.silt)
            + 0.1482 * np.log(om)
            - 0.04546 * d * self.silt
            - 0.4852 * d * om
            + 0.00673 * topsoil * self.clay
        )
        n_ = (
            -25.23
            - 0.02195 * self.clay
            + 0.0074 * self.silt
            - 0.1940 * om
            + 45.5 * d
            - 7.24 * d**2
            - 0.0003658 * self.clay**2
            + 0.002885 * om**2
            - 12.81 * d**-1
            - 0.1524 * self.silt**-1
            - 0.01958 * om**-1
            - 0.2876 * np.log(self.silt)
            - 0.0709 * np.log(om)
            - 44.6 * np.log(d)
            - 0.02264 * d * self.clay
            + 0.0896 * d * om
            + 0.00718 * topsoil * self.clay
        )
        l_ = (
            0.0202
            + 0.0006193 * self.clay**2
            - 0.001136 * om**2
            - 0.2316 * np.log(om)
            - 0.03544 * d * self.clay
            + 0.00283 * d * self.silt
            + 0.0488 * d * om
        )
        ks_ = (
            7.755
            + 0.0352 * self.silt
            + 0.93 * topsoil
            - 0.967 * d**2
            - 0.000484 * self.clay**2
            - 0.000322 * self.silt**2
            + 0.001 * self.silt**-1
            - 0.0748 * om**-1
            - 0.643 * np.log(self.silt)
            - 0.01398 * d * self.clay
            - 0.1673 * d * om
            + 0.02986 * topsoil * self.clay
            - 0.03305 * topsoil * self.silt
        )
        theta_r = 0.01
        return theta_r, theta_s, np.exp(ks_), np.exp(alpha_), np.exp(n_), np.exp(l_)

    def cosby(
        self,
    ):
        k01 = 3.100
        k02 = 15.70
        k03 = 0.300
        k04 = 0.505
        k05 = 0.037
        k06 = 0.142
        k07 = 2.170
        k08 = 0.630
        k09 = 1.580
        k10 = 0.600
        k11 = 0.640
        k12 = 1.260
        c = self.clay / 100
        s = self.sand / 100
        b = k01 + k02 * c - k03 * s
        theta_s = k04 - k05 * c - k06 * s
        psi_s = 0.01 * 10 ** (k07 - k08 * c - k09 * s)
        ks = 10 ** (-k10 - k11 * c + k12 * s) * 25.2 / 3600
        theta_r = 0.0
        labda = 1 / b
        ks = ks * 8640000 / 1000  # kg/m2/s to cm/d
        psi_s = psi_s * 100  # m to cm
        return theta_r, theta_s, ks, psi_s, labda

    def pedotransfer_bc(
        self,
    ):
        val = self.cosby()
        return Series(
            val, index=["theta_r", "theta_s", "ks [cm/d]", "psi_s [cm]", "labda"]
        )

    def pedotransfer_vg(
        self,
        d: float = None,
        th33: float = None,
        th1500: float = None,
        version: int = 3,
        om: float = None,
        ts: float = False,
    ):
        if d is None or om is None:
            val = self.schaap(
                d=d,
                th33=th33,
                th1500=th1500,
                version=version,
            )
        else:
            val = self.wosten(om=om, d=d, ts=ts)
        return Series(
            val, index=["theta_r", "theta_s", "ks [cm/d]", "alpha [1/cm]", "n", "l"]
        )

    def pedotransfer_mfusg(
        self,
        d: float = None,
        use_gamma: bool = False,
    ):
        val_vg = self.pedotransfer_vg(d=d)
        val_bc = self.pedotransfer_bc()
        alpha = val_vg.loc["alpha [1/cm]"]
        beta = val_vg.loc["n"]
        sr = val_vg.loc["theta_r"] / val_vg.loc["theta_s"]
        if use_gamma:
            gamma = 1 - 1 / beta
            brook = 1 + 2 / gamma
        else:
            labda = val_bc.loc["labda"]
            brook = 2 / labda + 3
        ks = val_vg.loc["ks [cm/d]"]

        return Series(
            data=(alpha, beta, sr, brook, ks),
            index=["alpha [1/cm]", "beta", "sr", "brook", "ks [cm/d]"],
        )
