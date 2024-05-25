from astropy.io import fits
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from copy import deepcopy
import emcee
import os
from . import ispec
from .utils import SpecUtils

#################################################################################
## --- iSpec directory -------------------------------------------------------------
# import os, sys
#
# ispec_dir = "/home/zchu/iSpec/"
# sys.path.insert(0, os.path.abspath(ispec_dir))
# import ispec


class ObsSpec:
    """
    Reads spectrum data from a FITS file created by pypeit, and reduce it.

    Parameters
    ----------
    spec_path : str
        Path to the spectrum file.
    spectrum_type : str
        "obj" or "sky", the type of the spectrum.
    spectrum_format="fits", "txt", "pkl"
        The format of the spectrum file.
    wave_range : tuple, optional
        The wavelength range of the spectrum in nm, by default None.
    """

    def __init__(
        self,
        spec_path: str,
        spectrum_type: str,
        spectrum_format="fits",
        date_obs=None,
        vel_corr=None,
        vel_type=None,
        wave_range=None,
        errfac=1.0,
    ) -> None:
        def restrict_wave_range(wave_nm, flux, ferr, wave_range):
            if wave_range is not None:
                wave_mask = (wave_nm > wave_range[0]) & (wave_nm < wave_range[1])
                wave_nm_new = wave_nm[wave_mask]
                flux_new = flux[wave_mask]
                ferr_new = ferr[wave_mask]
            else:
                wave_nm_new = wave_nm
                flux_new = flux
                ferr_new = ferr
            spectrum_array = np.array([wave_nm_new, flux_new, ferr_new]).T
            dtype = [("waveobs", "f8"), ("flux", "f8"), ("err", "f8")]
            spectrum = np.zeros(spectrum_array.shape[0], dtype=dtype)
            spectrum["waveobs"] = spectrum_array[:, 0]
            spectrum["flux"] = spectrum_array[:, 1]
            spectrum["err"] = spectrum_array[:, 2]
            return spectrum

        self.spec_path = spec_path
        if spectrum_format == "pkl":
            with open(spec_path, "rb") as f:
                data = pickle.load(f)
            self.errfac = data.errfac
            self.DATE_OBS = data.DATE_OBS
            self.VEL_CORR = data.VEL_CORR
            self.VEL_TYPE = data.VEL_TYPE
            self.spectrum = data.spectrum
            self.wave_range = wave_range
            self.spectrum = restrict_wave_range(
                self.spectrum["waveobs"],
                self.spectrum["flux"],
                self.spectrum["err"],
                self.wave_range,
            )

            self.normalized_spectrum = None
            return
        self.wave_range = wave_range
        self.errfac = errfac
        if spectrum_format == "fits":
            with fits.open(spec_path) as hdul:
                for hdu in hdul:
                    data, header = hdu.data, hdu.header
                    if "EXT0000" in header:
                        self.DATE_OBS = header["DATE-OBS"]
                    elif "OPT_COUNTS_SKY" in header.values():
                        # list all values in header
                        # for key, value in header.items():
                        #    if value == "OPT_COUNTS_SKY":
                        wave_nm = data["OPT_WAVE"] / 10
                        self.VEL_CORR = header["VEL_CORR"]
                        self.VEL_TYPE = "heliocentric"
                        if spectrum_type == "obj":
                            flux = data["OPT_COUNTS"]
                            ferr = data["OPT_COUNTS_SIG"] * self.errfac
                        elif spectrum_type == "sky":
                            flux = data["OPT_COUNTS_SKY"]
                            ferr = data["OPT_COUNTS_SIG_DET"] * self.errfac
        elif spectrum_format == "txt":
            if date_obs is None or vel_corr is None or vel_type is None:
                raise ValueError(
                    "Please provide date_obs, vel_corr, and vel_type when using txt format."
                )
            self.DATE_OBS = date_obs
            self.VEL_CORR = vel_corr
            self.VEL_TYPE = vel_type
            data = np.loadtxt(spec_path)
            wave_nm = data[:, 0]
            flux = data[:, 1]
            ferr = data[:, 2] * self.errfac

        self.spectrum = restrict_wave_range(wave_nm, flux, ferr, self.wave_range)
        self.normalized_spectrum = None
        return

    def reset_errfac(self, errfac):
        self.errfac = errfac
        self.spectrum["err"] = self.spectrum["err"] * self.errfac
        return

    def flatten_spectrum(self, wave_range):
        """
        Flatten the spectrum in a given wavelength range with the median value,
        in order to remove the information in this range.
        """
        self.normalize_whole_spectrum()
        mask = (self.spectrum["waveobs"] > wave_range[0]) & (
            self.spectrum["waveobs"] < wave_range[1]
        )
        median_err = np.median(self.spectrum["err"][mask])
        mask_points_num = np.sum(mask)
        self.spectrum["err"][mask] = median_err
        self.spectrum["flux"][mask] = self.continuum_flux[mask] + np.random.normal(
            scale=median_err, size=mask_points_num
        )
        self.normalized_spectrum = None
        return

    def normalize_whole_spectrum(
        self,
        model="Splines",
        degree=2,
        nknots=None,
        from_resolution=None,
        order="median+max",
        median_wave_range=0.05,
        max_wave_range=1.0,
        ignore=np.array([[656.5, 655.0, 657.5]]),
    ):
        """
        Use the whole spectrum but ignoring some strong lines,

        Parameters
        ----------
        star_spectrum : [type]
            [description]
        model : str, optional
            "Splines" or "Polynomy", by default "Splines"
        degree : int, optional
            Degree of the polynomial, by default 2
        nknots : int, optional
            Number of knots, by default 1 spline every 5 nm
        ignore : np.array, optional
            Array of lines to ignore, by default np.array([[656.5, 655.0, 657.5]]),
            i.e. the H-alpha line
        """
        # star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")

        # --- Continuum fit -------------------------------------------------------------
        # model = "Splines"  # "Polynomy"
        # degree = 2
        # nknots = None  # Automatic: 1 spline every 5 nm
        # from_resolution = None

        # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
        # order = "median+max"
        # median_wave_range = 0.05
        # max_wave_range = 1.0
        dt = np.dtype([("wave_peak", "f8"), ("wave_base", "f8"), ("wave_top", "f8")])
        ignore_array = np.zeros(len(ignore), dtype=dt)
        ignore_array["wave_peak"] = ignore[:, 0]
        ignore_array["wave_base"] = ignore[:, 1]
        ignore_array["wave_top"] = ignore[:, 2]
        star_continuum_model = ispec.fit_continuum(
            self.spectrum,
            ignore=ignore_array,
            from_resolution=from_resolution,
            nknots=nknots,
            degree=degree,
            median_wave_range=median_wave_range,
            max_wave_range=max_wave_range,
            model=model,
            order=order,
            automatic_strong_line_detection=True,
            strong_line_probability=0.5,
            use_errors_for_fitting=True,
        )
        self.continuum_flux = star_continuum_model(self.spectrum["waveobs"])

        # --- Continuum normalization ---------------------------------------------------
        self.normalized_spectrum = ispec.normalize_spectrum(
            self.spectrum, star_continuum_model, consider_continuum_errors=False
        )
        # self.normalized_spectrum = np.copy(self.spectrum)
        # self.normalized_spectrum["flux"] = self.spectrum["flux"] / self.continuum_flux
        # self.normalized_spectrum["err"] = self.spectrum["err"] / self.continuum_flux
        return

    def normalize_with_lower_envelope(self, window_size=50, degree=2, percentile=5):
        temp_dataframe = pd.DataFrame(self.spectrum)

        # Applying the custom function to the rolling window using lambda
        temp_dataframe["lower_envelope"] = (
            temp_dataframe["flux"]
            .rolling(window=window_size, min_periods=1)
            .apply(lambda x: np.percentile(x, percentile), raw=True)
        )

        # Polynomial fitting to the lower envelope
        p = np.polyfit(
            temp_dataframe["waveobs"], temp_dataframe["lower_envelope"], degree
        )
        print(p)
        self.continuum_flux = np.polyval(p, temp_dataframe["waveobs"])

        self.normalized_spectrum = np.copy(self.spectrum)
        self.normalized_spectrum["flux"][:] = 1.0
        self.normalized_spectrum["err"][:] = 0.0
        self.normalized_spectrum["flux"] = self.spectrum["flux"] / self.continuum_flux
        self.normalized_spectrum["err"] = self.spectrum["err"] / self.continuum_flux
        return

    def to_heliocentric(self) -> None:
        if self.VEL_TYPE != "heliocentric":
            self.spectrum["waveobs"] *= self.VEL_CORR
            self.VEL_TYPE = "heliocentric"
        return

    def to_earthentric(self) -> None:
        if self.VEL_TYPE != "earthentric":
            self.spectrum["waveobs"] /= self.VEL_CORR
            self.VEL_TYPE = "earthentric"
        return

    def shift_spec(self, vcorr) -> None:
        c = 299792.458
        self.spectrum["waveobs"] *= 1 + vcorr / c
        return

    def to_pkl(self, pkl_path: str) -> None:
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)
        return

    def plot_continuum_fit(
        self, ax: plt.Axes, flux_sty_kw: dict = {}, continuum_sty_kw: dict = {}
    ):
        # --- Plotting ------------------------------------------------------------------
        ax.plot(self.spectrum["waveobs"], self.spectrum["flux"], **flux_sty_kw)
        ax.fill_between(
            self.spectrum["waveobs"],
            self.spectrum["flux"] - self.spectrum["err"],
            self.spectrum["flux"] + self.spectrum["err"],
            alpha=0.2,
            color=flux_sty_kw.get("color", "C0"),
        )
        ax.plot(self.spectrum["waveobs"], self.continuum_flux, **continuum_sty_kw)
        return

    def plot_normalized_spectrum(self, ax: plt.Axes, flux_sty_kw: dict = {}):
        ax.plot(
            self.normalized_spectrum["waveobs"],
            self.normalized_spectrum["flux"],
            **flux_sty_kw,
        )
        ax.fill_between(
            self.normalized_spectrum["waveobs"],
            self.normalized_spectrum["flux"] - self.normalized_spectrum["err"],
            self.normalized_spectrum["flux"] + self.normalized_spectrum["err"],
            alpha=0.2,
            color=flux_sty_kw.get("color", "C0"),
        )
        ax.set_title("Continuum normalization")
        return


class ModelSpec:
    def __init__(self, model_spec_path: str, wave_range=None) -> None:
        self.model_path = model_spec_path
        model_spec_arr = np.loadtxt(model_spec_path)
        wave_nm = model_spec_arr[:, 0] / 10
        flux = model_spec_arr[:, 1]
        if wave_range is not None:
            wave_mask = (wave_nm > wave_range[0]) & (wave_nm < wave_range[1])
            wave_nm = wave_nm[wave_mask]
            flux = flux[wave_mask]
        spectrum_array = np.array([wave_nm, flux]).T
        dtype = [("waveobs", "f8"), ("flux", "f8")]
        self.spectrum = np.zeros(spectrum_array.shape[0], dtype=dtype)
        self.spectrum["waveobs"] = spectrum_array[:, 0]
        self.spectrum["flux"] = spectrum_array[:, 1]
        return


class SynthSpec:
    def __init__(
        self,
        wave_range,
        resolution,
        teff,
        logg,
        meh,
    ) -> None:
        self.wave_range = wave_range
        self.resolution = resolution
        current_file_folder = os.path.dirname(os.path.abspath(__file__))
        self.spectrum_data_path = f"{current_file_folder}/data/phoenix"

        assert teff >= 2300 and teff <= 12000
        assert logg >= 0.0 and logg <= 6.0
        assert meh >= -4.0 and meh <= 0.5
        self.teff = teff
        self.logg = logg
        self.meh = meh

        self._teff_step = 100
        self._teff_low = 2300
        self._teff_high = 12000
        self._teff_list = np.arange(self._teff_low, self._teff_high, self._teff_step)
        self._logg_step = 0.5
        self._logg_low = 0.0
        self._logg_high = 6.0
        self._logg_list = np.arange(self._logg_low, self._logg_high, self._logg_step)
        self._meh_list = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
        self.hires_wave = None
        self.hires_flux = None
        self.lowres_wave = None
        self.lowres_flux = None

        return

    def get_synth_spec(self, teff_grid, logg_grid, meh_grid):
        with fits.open(
            f'{self.spectrum_data_path}/lte{teff_grid:05d}-{logg_grid:.1f}{"+" if meh_grid >0.1 else "-"}{abs(meh_grid):.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        ) as hdu:
            hires_flux = hdu[0].data
        hires_wave = np.arange(3000, 10000, 0.1)
        return hires_wave, hires_flux

    def interpolator(self):
        # find the nearest two teff, logg, meh
        teff_idx1 = np.where(self._teff_list <= self.teff)[0][-1]
        teff_1 = self._teff_list[teff_idx1]
        teff_idx2 = np.where(self._teff_list > self.teff)[0][0]
        teff_2 = self._teff_list[teff_idx2]
        logg_idx1 = np.where(self._logg_list <= self.logg)[0][-1]
        logg_1 = self._logg_list[logg_idx1]
        logg_idx2 = np.where(self._logg_list > self.logg)[0][0]
        logg_2 = self._logg_list[logg_idx2]
        meh_idx1 = np.where(np.array(self._meh_list) <= self.meh)[0][-1]
        meh_1 = self._meh_list[meh_idx1]
        meh_idx2 = np.where(np.array(self._meh_list) > self.meh)[0][0]
        meh_2 = self._meh_list[meh_idx2]

        def combine_spec(flux1, flux2, weight1, weight2):
            return flux1 * weight1 + flux2 * weight2

        flux_dteff = []
        for i, teff in enumerate([teff_1, teff_2]):
            flux_dteff_dlogg = []
            for j, logg in enumerate([logg_1, logg_2]):
                flux_dteff_dlogg_dmeh = []
                for k, meh in enumerate([meh_1, meh_2]):
                    # idx_str = f"{teff:05d}-{logg:.1f}{"+" if meh >0.1 else "-"}{abs(meh):.1f}"
                    wave, flux = self.get_synth_spec(teff, logg, meh)
                    flux_dteff_dlogg_dmeh.append(flux)
                weight_meh1 = (meh_2 - self.meh) / (meh_2 - meh_1)
                weight_meh2 = 1 - weight_meh1
                flux_dteff_dlogg.append(
                    combine_spec(
                        flux_dteff_dlogg_dmeh[0],
                        flux_dteff_dlogg_dmeh[1],
                        weight_meh1,
                        weight_meh2,
                    )
                )
            weight_logg1 = (logg_2 - self.logg) / (logg_2 - logg_1)
            weight_logg2 = 1 - weight_logg1
            flux_dteff.append(
                combine_spec(
                    flux_dteff_dlogg[0], flux_dteff_dlogg[1], weight_logg1, weight_logg2
                )
            )
        weight_teff1 = (teff_2 - self.teff) / (teff_2 - teff_1)
        weight_teff2 = 1 - weight_teff1
        flux = combine_spec(flux_dteff[0], flux_dteff[1], weight_teff1, weight_teff2)
        # since the wavelength are the different, we need to interpolate the flux
        self.hires_wave = wave
        self.hires_flux = flux
        return wave, flux

    def get_spec(self):
        if self.hires_wave is None or self.hires_flux is None:
            self.interpolator()

        self.lowres_wave, self.lowres_flux, _, _ = SpecUtils.lower_reslution(
            self.hires_wave,
            self.hires_flux,
            self.wave_range,
            self.resolution,
            padding=2,
            save_padding=1,
        )
        return self.lowres_wave, self.lowres_flux
