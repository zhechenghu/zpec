from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit, minimize
from copy import deepcopy
from .spec import ObsSpec


class StarRV:
    def __init__(
        self,
        observed_spec_base: ObsSpec,
        observed_spec_target: ObsSpec,
        vel_range=(-300, 300),
        d_vel=0.1,
        ignore_wave_range_list=None,
    ) -> None:
        self.observed_spec_base = deepcopy(observed_spec_base)
        self.observed_spec_target = deepcopy(observed_spec_target)
        if ignore_wave_range_list is not None:
            for ignore_wave_range in ignore_wave_range_list:
                # we only flatten the observed spectrum
                # because the base spectrum could be a template,
                # which do not contain any telluric lines
                self.observed_spec_target.flatten_spectrum(ignore_wave_range)
        self.vel_arr = np.arange(vel_range[0], vel_range[1], d_vel)
        return

    def get_ccf(self):
        """
        Determines the radial velocity based on the observed spectra.

        Args:
            vel_range (tuple, optional): The velocity range (in km/s) to consider. Defaults to (-300, 300).
            d_vel (float, optional): The velocity step (in km/s) to consider. Defaults to 0.1.
            wave_range (tuple, optional): The wavelength range (in nm) to consider. Defaults to (600, 670).

        Returns:
            float: The determined radial velocity.
        """
        base_spec = self.observed_spec_base.normalized_spectrum
        # wave_range = (np.min(base_spec["waveobs"]), np.max(base_spec["waveobs"]))
        # print(wave_range)
        target_spec = self.observed_spec_target.normalized_spectrum
        ccf_list = []
        ccf_err_list = []
        for i in range(len(self.vel_arr)):
            # 1. compute the shifted wave length of the target spectrum
            target_spec_shifted = np.copy(target_spec)
            target_spec_shifted["waveobs"] *= 1 - self.vel_arr[i] / 299792.458
            # note that here we use a minus sign
            # In the next step, we would interpolate the base spectrum rather than the target spectrum
            # at the shifted wavelength, which is equivalent to shift the base spectrum
            # to the opposite direction of the target spectrum

            # 2. interpolate the base spectrum rather than the target spectrum to make the CCF smooth
            base_spec_interp_model = interpolate.interp1d(
                base_spec["waveobs"], base_spec["flux"], kind="linear"
            )
            base_spec_interp_flux = base_spec_interp_model(
                target_spec_shifted["waveobs"]
            )

            # 3. compute the CCF
            # if the velocity is at the actual RV, the CCF should be the highest
            ccf = np.sum(target_spec_shifted["flux"] * base_spec_interp_flux)
            ccf_err = np.sqrt(
                np.sum(target_spec_shifted["err"] ** 2 * base_spec_interp_flux**2)
            )
            ccf_list.append(ccf)
            ccf_err_list.append(ccf_err)
        self.ccf = np.array(ccf_list)
        self.ccf_err = np.array(ccf_err_list)
        return self.vel_arr, self.ccf, self.ccf_err

    def fit_ccf(self):
        """
        Fits the CCF with a Gaussian function.

        Returns:
            float: The best fit radial velocity.
            float: The error of the best fit radial velocity.
        """

        def gaussian_2ndpoly(x, a, mu, sigma, b, c, d):
            return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b * x**2 + c * x + d

        def get_chi2(theta, x, y, yerr):
            a, mu, sigma, b, c, d = theta
            model = a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b * x**2 + c * x + d
            chi2 = np.sum((y - model) ** 2 / yerr**2)
            return chi2

        popt, pcov = curve_fit(
            gaussian_2ndpoly,
            self.vel_arr,
            self.ccf,
            p0=[np.max(self.ccf), 0, 1, 100, 0.0, 0.0],
            sigma=self.ccf_err,
        )
        self.rv = popt[1]
        self.rv_err = np.sqrt(pcov[1, 1])

        self.ccf_model_params_dict = {
            "a": popt[0],
            "mu": popt[1],
            "sigma": popt[2],
            "b": popt[3],
            "c": popt[4],
            "d": popt[5],
        }

        self.ccf_model = gaussian_2ndpoly(self.vel_arr, *popt)
        return self.rv, self.rv_err

    def determine_radial_velocity(
        self,
    ):
        if self.observed_spec_base.normalized_spectrum is None:
            self.observed_spec_base.normalize_whole_spectrum()
        if self.observed_spec_target.normalized_spectrum is None:
            self.observed_spec_target.normalize_whole_spectrum()

        # The spectrum should already in heliocentric frame
        self.get_ccf()
        self.fit_ccf()
        return self.rv, self.rv_err

    def plot_ccf(self, ax: plt.Axes, ccf_sty_kw: dict = {}):
        ax.plot(self.vel_arr, self.ccf, **ccf_sty_kw)
        ax.fill_between(
            self.vel_arr,
            self.ccf - self.ccf_err,
            self.ccf + self.ccf_err,
            alpha=0.2,
            color=ccf_sty_kw.get("color", "C0"),
        )
        ax.set_xlabel("Velocity [km/s]")
        ax.set_ylabel("CCF")
        ax.set_title("Cross-correlation function")
        return

    def plot_ccf_model(self, ax: plt.Axes, ccf_sty_kw: dict = {}):
        ax.plot(self.vel_arr, self.ccf_model, **ccf_sty_kw)
        ax.set_xlabel("Velocity [km/s]")
        ax.set_ylabel("CCF")
        ax.set_title("Cross-correlation function")
        return
