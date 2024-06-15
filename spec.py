from astropy.io import fits
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from copy import deepcopy
import os
from . import ispec
from .path import _PHOENIX_DATA_DIR
from .utils import SpecUtils, PlotUtils
import tempfile


class BaseSpec:
    """
    Base class for spectrum.
    """

    def __init__(
        self,
        wave_nm,
        flux,
        ferr=None,
        wave_range=None,
        errfac=1.0,
    ) -> None:
        self.wave_range = wave_range
        self.errfac = errfac
        if ferr is None:
            ferr = np.zeros_like(flux)
        self.spectrum = BaseSpec.restrict_wave_range(wave_nm, flux, ferr, wave_range)
        self.reset_errfac(self.errfac)
        self.normalized_spectrum = None
        return

    def __str__(self) -> str:
        spectrum_info = (
            f"Wavelength Range: {self.wave_range}\n"
            f"Error Factor: {self.errfac}\n"
            f"Spectrum Length: {len(self.spectrum)}"
        )
        return spectrum_info

    @staticmethod
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

    def reset_errfac(self, errfac):
        new_errfac = errfac
        self.spectrum["err"] = self.spectrum["err"] * new_errfac / self.errfac
        self.errfac = new_errfac
        return

    def flatten_spectrum(self, wave_range):
        """
        Flatten the spectrum in a given wavelength range with the continuum,
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

    def shift_spec(self, vcorr) -> None:
        c = 299792.458
        self.spectrum["waveobs"] *= 1 + vcorr / c
        if self.normalized_spectrum is not None:
            self.normalized_spectrum["waveobs"] *= 1 + vcorr / c
        return

    def interpolate_new_wave(self, new_wave):
        """
        Interpolate the spectrum to a new wavelength grid.
        """
        assert np.min(new_wave) >= np.min(self.spectrum["waveobs"])
        assert np.max(new_wave) <= np.max(self.spectrum["waveobs"])

        f = interpolate.interp1d(
            self.spectrum["waveobs"],
            self.spectrum["flux"],
        )
        new_flux = f(new_wave)
        # assume the errors also have a pattern for interpolate
        f = interpolate.interp1d(
            self.spectrum["waveobs"],
            self.spectrum["err"],
        )
        new_err = f(new_wave)
        self.spectrum = np.array(
            [new_wave, new_flux, new_err],
            dtype=[("waveobs", "f8"), ("flux", "f8"), ("err", "f8")],
        )
        return

    def to_pkl(self, pkl_path: str) -> None:
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)
        return

    def plot_spectrum(self, ax: plt.Axes, rv_corr=None, flux_sty_kw: dict = {}):
        if rv_corr is None:
            wave = self.spectrum["waveobs"]
        else:
            c = 299792.458
            wave = self.spectrum["waveobs"] * (1 + rv_corr / c)
        ax.plot(
            wave,
            self.spectrum["flux"],
            **flux_sty_kw,
        )
        ax.fill_between(
            wave,
            self.spectrum["flux"] - self.spectrum["err"],
            self.spectrum["flux"] + self.spectrum["err"],
            alpha=0.2,
            color=flux_sty_kw.get("color", "C0"),
        )

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
        # ax.set_title("Continuum normalization")
        return


class ObsSpec(BaseSpec):
    """
    Reads spectrum data and reduce it.
    Supported formats: fits, txt, pkl.
    The fits file should be generated by the `pypeit` pipeline.
    The txt file should be in the format of (wave_nm, flux, ferr)
    The pkl file should be a pickled object of the ObsSpec class.

    Parameters
    ----------
    spec_path : str
        Path to the spectrum file.
    spectrum_type : str
        "obj" or "sky", the type of the spectrum. This is used only when reading fits file.
    spectrum_format="fits", "txt", "pkl"
        The format of the spectrum file.
    wave_range : tuple, optional
        The wavelength range of the spectrum in nm, by default None.
    """

    def __init__(
        self,
        spec_path: str,
        spectrum_type="obj",
        spectrum_format="fits",
        date_obs=None,
        vel_corr=None,
        vel_type=None,
        wave_range=None,
        errfac=1.0,
    ) -> None:
        self.spec_path = spec_path
        if spectrum_format == "pkl":
            self._process_pkl(spec_path, wave_range)
            return
        else:
            self.wave_range = wave_range
            self.errfac = errfac
            if spectrum_format == "fits":
                self._process_fits(spec_path, spectrum_type)
            elif spectrum_format == "txt":
                self._process_txt(spec_path, date_obs, vel_corr, vel_type)

        self.reset_errfac(self.errfac)
        self.normalized_spectrum = None
        return

    def __str__(self) -> str:
        spectrum_info = (
            f"Spectrum Path: {self.spec_path}\n"
            f"Date Obs: {self.DATE_OBS}\n"
            f"Velocity Correction: {self.VEL_CORR}\n"
            f"Velocity Type: {self.VEL_TYPE}\n"
            f"Wavelength Range: {self.wave_range}\n"
            f"Error Factor: {self.errfac}\n"
            f"Spectrum Length: {len(self.spectrum)}"
        )
        return spectrum_info

    def _process_pkl(self, pkl_path: str, wave_range) -> None:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.errfac = data.errfac
        self.DATE_OBS = data.DATE_OBS
        self.VEL_CORR = data.VEL_CORR
        self.VEL_TYPE = data.VEL_TYPE
        self.wave_range = wave_range
        self.specturm = data.spectrum
        self.spectrum = self.restrict_wave_range(
            self.spectrum["waveobs"],
            self.spectrum["flux"],
            self.spectrum["err"],
            self.wave_range,
        )
        self.normalized_spectrum = None
        return

    def _process_fits(self, fits_path: str, spectrum_type) -> None:
        with fits.open(fits_path) as hdul:
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
        self.spectrum = self.restrict_wave_range(wave_nm, flux, ferr, self.wave_range)
        return

    def _process_txt(self, txt_path: str, date_obs, vel_corr, vel_type) -> None:
        if date_obs is None or vel_corr is None or vel_type is None:
            raise ValueError(
                "Please provide date_obs, vel_corr, and vel_type when using txt format."
            )
        self.DATE_OBS = date_obs
        self.VEL_CORR = vel_corr
        self.VEL_TYPE = vel_type
        data = np.loadtxt(txt_path)
        wave_nm = data[:, 0]
        flux = data[:, 1]
        ferr = data[:, 2] * self.errfac
        self.spectrum = self.restrict_wave_range(wave_nm, flux, ferr, self.wave_range)
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


class ModelSpec(BaseSpec):
    """
    Reads a model spectrum and reduce it.
    The biggest difference between this class and ObsSpec is
    1. The model spectrum does not have a date_obs, vel_type. One only need to define the rv of the model.
    2. The error of model spectrum is always zero.
    """

    def __init__(
        self,
        txt_spec_file=None,
        wave_nm=None,
        flux=None,
        wave_range=None,
        rv=0.0,
    ) -> None:
        if txt_spec_file is not None:
            self._process_txt(txt_spec_file)
        else:
            super().__init__(wave_nm, flux, ferr=None, wave_range=wave_range)
        self.rv = rv
        self.normalized_spectrum = None
        return

    def _process_txt(self, txt_path: str) -> None:
        data = np.loadtxt(txt_path)
        wave_nm = data[:, 0]
        flux = data[:, 1]
        ferr = np.zeros_like(flux)
        self.spectrum = self.restrict_wave_range(wave_nm, flux, ferr, self.wave_range)
        return


class SynthSpec:
    def __init__(
        self,
        wave_range_nm,
        resolution,
        teff,
        logg,
        feh,
    ) -> None:
        self.wave_range_nm = wave_range_nm
        self.resolution = resolution
        self.spectrum_data_path = _PHOENIX_DATA_DIR

        assert teff >= 2300 and teff <= 12000
        assert logg >= 0.0 and logg <= 6.0
        assert feh >= -4.0 and feh <= 1.0
        self.teff = teff
        self.logg = logg
        self.feh = feh

        self.param_df = PlotUtils._generate_params_dataframe()

        self._teff_list = np.sort(np.array(self.param_df["Teff"].unique()))
        self._logg_list = np.sort(np.array(self.param_df["logg"].unique()))
        self._feh_list = np.sort(np.array(self.param_df["feh"].unique()))
        self.hires_wave_nm = None
        self.hires_flux = None
        self.lowres_wave_nm = None
        self.lowres_flux = None

        return

    def _read_synth_spec(self, teff_grid, logg_grid, meh_grid):
        with fits.open(
            f'{self.spectrum_data_path}/lte{teff_grid:05d}-{logg_grid:.2f}{"+" if meh_grid >0.1 else "-"}{abs(meh_grid):.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        ) as hdu:
            hires_flux = hdu[0].data
        hires_wave = np.arange(3000, 10000, 0.1)
        return hires_wave, hires_flux

    def _get_neighbour(self):
        # find the nearest two teff, logg, meh
        teff_idx1 = np.where(self._teff_list <= self.teff)[0][-1]
        if teff_idx1 == len(self._teff_list) - 1:
            teff_idx1 -= 1
        teff_1 = self._teff_list[teff_idx1]
        teff_2 = self._teff_list[teff_idx1 + 1]
        logg_idx1 = np.where(self._logg_list <= self.logg)[0][-1]
        if logg_idx1 == len(self._logg_list) - 1:
            logg_idx1 -= 1
        logg_1 = self._logg_list[logg_idx1]
        logg_2 = self._logg_list[logg_idx1 + 1]
        feh_idx1 = np.where(np.array(self._feh_list) <= self.feh)[0][-1]
        if feh_idx1 == len(self._feh_list) - 1:
            feh_idx1 -= 1
        feh_1 = self._feh_list[feh_idx1]
        feh_2 = self._feh_list[feh_idx1 + 1]
        return teff_1, logg_1, feh_1, teff_2, logg_2, feh_2

    def _get_weight(self):
        teff_1, logg_1, feh_1, teff_2, logg_2, feh_2 = self._get_neighbour()
        weight_teff1 = (teff_2 - self.teff) / (teff_2 - teff_1)
        weight_teff2 = 1 - weight_teff1
        weight_logg1 = (logg_2 - self.logg) / (logg_2 - logg_1)
        weight_logg2 = 1 - weight_logg1
        weight_feh1 = (feh_2 - self.feh) / (feh_2 - feh_1)
        weight_feh2 = 1 - weight_feh1
        return (
            weight_teff1,
            weight_teff2,
            weight_logg1,
            weight_logg2,
            weight_feh1,
            weight_feh2,
        )

    def _interpolator(self):
        teff_1, logg_1, feh_1, teff_2, logg_2, feh_2 = self._get_neighbour()
        (
            weight_teff1,
            weight_teff2,
            weight_logg1,
            weight_logg2,
            weight_feh1,
            weight_feh2,
        ) = self._get_weight()

        def combine_spec(flux1, flux2, weight1, weight2):
            return flux1 * weight1 + flux2 * weight2

        flux_dteff = []
        for i, teff in enumerate([teff_1, teff_2]):
            flux_dteff_dlogg = []
            for j, logg in enumerate([logg_1, logg_2]):
                flux_dteff_dlogg_dfeh = []
                for k, feh in enumerate([feh_1, feh_2]):
                    wave_A, flux = self._read_synth_spec(teff, logg, feh)
                    flux_dteff_dlogg_dfeh.append(flux)
                flux_dteff_dlogg.append(
                    combine_spec(
                        flux_dteff_dlogg_dfeh[0],
                        flux_dteff_dlogg_dfeh[1],
                        weight_feh1,
                        weight_feh2,
                    )
                )
            flux_dteff.append(
                combine_spec(
                    flux_dteff_dlogg[0], flux_dteff_dlogg[1], weight_logg1, weight_logg2
                )
            )
        flux = combine_spec(flux_dteff[0], flux_dteff[1], weight_teff1, weight_teff2)
        # since the wavelength are the different, we need to interpolate the flux
        self.hires_wave_nm = wave_A / 10
        self.hires_flux = flux
        return self.hires_wave_nm, self.hires_flux

    def get_spec(self):
        if self.hires_wave_nm is None or self.hires_flux is None:
            self._interpolator()

        self.lowres_wave_nm, self.lowres_flux, _, _ = SpecUtils.lower_resolution(
            self.hires_wave_nm,
            self.hires_flux,
            self.wave_range_nm,
            self.resolution,
            padding=2,
            save_padding=1,
        )
        model_spec = ModelSpec(
            wave_nm=self.lowres_wave_nm,
            flux=self.lowres_flux,
            wave_range=self.wave_range_nm,
            rv=0,
        )

        return model_spec
