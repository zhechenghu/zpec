from astropy.io import fits
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from copy import deepcopy
from PyAstronomy import pyasl
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
        mask = (self.spectrum["waveobs"] > wave_range[0]) * (
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

    def shift_spec(self, vcorr_km) -> None:
        c = 299792.458
        self.spectrum["waveobs"] *= 1 + vcorr_km / c
        if self.normalized_spectrum is not None:
            self.normalized_spectrum["waveobs"] *= 1 + vcorr_km / c
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
        spectrum_array = np.array([new_wave, new_flux, new_err]).T
        dtype = [("waveobs", "f8"), ("flux", "f8"), ("err", "f8")]
        self.spectrum = np.zeros(spectrum_array.shape[0], dtype=dtype)
        self.spectrum["waveobs"] = spectrum_array[:, 0]
        self.spectrum["flux"] = spectrum_array[:, 1]
        self.spectrum["err"] = spectrum_array[:, 2]
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
    Supported formats: fits, txt, pkl, h5.
    The fits file should be generated by the `pypeit` pipeline.
    The txt file should be in the format of (wave_nm, flux, ferr)
    The pkl file should be a pickled object of the ObsSpec class.
    The h5 file should have spectrum/{waveobs,flux,err} datasets and root attrs.

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
        elif spectrum_format == "h5":
            self._process_h5(spec_path, wave_range)
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
        # self.errfac = data.errfac
        # self.DATE_OBS = data.DATE_OBS
        # self.VEL_CORR = data.VEL_CORR
        # self.VEL_TYPE = data.VEL_TYPE
        # self.wave_range = wave_range
        # self.spectrum = np.copy(data.spectrum)
        self.errfac = data["errfac"]
        self.DATE_OBS = data["DATE_OBS"]
        self.VEL_CORR = data["VEL_CORR"]
        self.VEL_TYPE = data["VEL_TYPE"]
        self.wave_range = data["wave_range"]
        self.spectrum = np.copy(data["spectrum"])
        self.spectrum = self.restrict_wave_range(
            self.spectrum["waveobs"],
            self.spectrum["flux"],
            self.spectrum["err"],
            wave_range,
        )
        self.wave_range = wave_range
        self.normalized_spectrum = None
        return

    def _process_h5(self, h5_path: str, wave_range) -> None:
        import h5py
        with h5py.File(h5_path, "r") as f:
            self.errfac = float(f.attrs["errfac"])
            self.DATE_OBS = f.attrs["DATE_OBS"]
            self.VEL_CORR = float(f.attrs["VEL_CORR"])
            self.VEL_TYPE = f.attrs["VEL_TYPE"]
            wave_nm = f["spectrum/waveobs"][:]
            flux = f["spectrum/flux"][:]
            ferr = f["spectrum/err"][:]
        self.spectrum = self.restrict_wave_range(wave_nm, flux, ferr, wave_range)
        self.wave_range = wave_range
        self.normalized_spectrum = None
        return

    def to_h5(self, h5_path: str) -> None:
        import h5py
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("spectrum")
            grp.create_dataset("waveobs", data=self.spectrum["waveobs"])
            grp.create_dataset("flux", data=self.spectrum["flux"])
            grp.create_dataset("err", data=self.spectrum["err"])
            f.attrs["DATE_OBS"] = self.DATE_OBS
            f.attrs["VEL_CORR"] = self.VEL_CORR
            f.attrs["VEL_TYPE"] = self.VEL_TYPE
            f.attrs["errfac"] = float(self.errfac)
            if self.wave_range is not None:
                f.attrs["wave_range"] = np.array(self.wave_range)
            else:
                f.attrs["wave_range"] = "None"
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

    def to_pkl(self, pkl_path: str) -> None:
        save_data = {
            "spectrum": np.copy(self.spectrum),
            "errfac": np.copy(self.errfac),
            "DATE_OBS": self.DATE_OBS,
            "VEL_CORR": self.VEL_CORR,
            "VEL_TYPE": self.VEL_TYPE,
            "wave_range": np.copy(self.wave_range),
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(save_data, f)
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
        self.wave_range = wave_range
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
    """
    Reads a synthetic spectrum and reduce it. The original unit of the flux is erg/s/cm^2/cm.
    """
    def __init__(
        self,
        wave_range_nm,
        resolution,
        teff,
        logg,
        feh,
        alpha=0,
    ) -> None:
        self.wave_range_nm = wave_range_nm
        if wave_range_nm[0] < 300:
            raise ValueError("The minimum wavelength is too short, please use a longer range.")
        if wave_range_nm[1] > 2500:
            raise ValueError("The maximum wavelength is too long, please use a shorter range.")

        self.resolution = resolution
        self.spectrum_data_path = _PHOENIX_DATA_DIR

        assert teff >= 2300 and teff <= 12000
        assert logg >= 0.0 and logg <= 6.0
        assert feh >= -4.0 and feh <= 1.0
        assert alpha >= -0.2 and alpha <= 1.2
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

    def _read_synth_spec(self, teff_grid, logg_grid, feh_grid, alpha_grid=0.0):
        teff_str = f"{teff_grid:05d}"
        logg_str = f"{logg_grid:.2f}"
        feh_str = f'{"+" if feh_grid >0.1 else "-"}{abs(feh_grid):.1f}'
        if np.abs(alpha_grid) < 0.01:
            path = f"{self.spectrum_data_path}/lte{teff_str}-{logg_str}{feh_str}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        else:
            alpha_str = f'{"+" if alpha_grid >0.1 else "-"}{abs(alpha_grid):.2f}'
            path = f"{self.spectrum_data_path}/lte{teff_str}-{logg_str}{feh_str}.Alpha={alpha_str}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        with fits.open(path) as hdu:
            # load wavelength
            crval1 = hdu[0].header['CRVAL1']
            cdelt1 = hdu[0].header['CDELT1']
            naxis1 = hdu[0].header['NAXIS1']
            pixel_indices = np.arange(naxis1)
            log10_wavelengths = crval1 + pixel_indices * cdelt1
            hires_wave = np.exp(log10_wavelengths)

            # load flux
            hires_flux = hdu[0].data
        #hires_wave = np.arange(3000, 10000, 0.1)
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
        # TODO: inplement alpha enhancement
        if self.hires_wave_nm is None or self.hires_flux is None:
            self._interpolator()

        # self.lowres_wave_nm, self.lowres_flux, _, _ = SpecUtils.lower_resolution(
        #    self.hires_wave_nm,
        #    self.hires_flux,
        #    self.wave_range_nm,
        #    self.resolution,
        #    padding=2,
        #    save_padding=1,
        # )

        wave_range_mask = (self.hires_wave_nm > self.wave_range_nm[0]) & (
            self.hires_wave_nm < self.wave_range_nm[1]
        )
        self.lowres_wave_nm = self.hires_wave_nm[wave_range_mask]
        hires_flux = self.hires_flux[wave_range_mask]
        self.lowres_flux, fwhm = pyasl.instrBroadGaussFast(
            self.lowres_wave_nm,
            hires_flux,
            self.resolution,
            edgeHandling="firstlast",
            fullout=True,
            equid=True
        )
        model_spec = ModelSpec(
            wave_nm=self.lowres_wave_nm,
            flux=self.lowres_flux,
            wave_range=self.wave_range_nm,
            rv=0,
        )

        return model_spec


class ISpecSynthSpec:
    """
    Synthesize spectra using iSpec radiative transfer codes (SPECTRUM, MOOG, etc.).

    Unlike SynthSpec (which interpolates pre-computed PHOENIX grids), this class
    generates spectra on-the-fly via radiative transfer through model atmospheres.

    Requires:
        - iSpec installation at ispec_dir (default ~/iSpec)
        - ispec conda env (Python 3.10) due to compiled synthesizer module
        - multiprocessing.set_start_method("fork") on macOS

    Parameters
    ----------
    wave_range_nm : tuple
        (wave_base, wave_top) in nm. Must be within 420-920 nm for GESv6 linelist.
    resolution : float
        Spectral resolution R = lambda / delta_lambda.
    teff : float
        Effective temperature in K.
    logg : float
        Surface gravity log(g).
    feh : float
        Metallicity [Fe/H].
    alpha : float or None
        Alpha enhancement. If None, auto-estimated from [Fe/H].
    code : str
        Synthesis code: "spectrum", "moog", "turbospectrum", "synthe", "sme".
    vsini : float
        Rotational velocity in km/s.
    limb_darkening_coeff : float
        Limb darkening coefficient.
    wave_step : float
        Wavelength step in nm for the output grid.
    ispec_dir : str
        Path to iSpec installation directory.
    """

    # Class-level cache for expensive data that doesn't change between calls
    _cache = {}

    def __init__(
        self,
        wave_range_nm,
        resolution,
        teff,
        logg,
        feh,
        alpha=None,
        code="spectrum",
        vsini=1.60,
        limb_darkening_coeff=0.6,
        wave_step=0.001,
        ispec_dir=None,
    ):
        if ispec_dir is None:
            ispec_dir = os.path.expanduser("~/iSpec")
        self.ispec_dir = ispec_dir
        self.wave_range_nm = wave_range_nm
        self.resolution = resolution
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.code = code
        self.vsini = vsini
        self.limb_darkening_coeff = limb_darkening_coeff
        self.wave_step = wave_step

        # Import iSpec lazily
        self._ispec = self._get_ispec()

        # Auto-estimate alpha if not provided
        if alpha is None:
            self.alpha = self._ispec.determine_abundance_enchancements(feh)
        else:
            self.alpha = alpha

        # Load cached data
        self._load_cached_data()

    def _get_ispec(self):
        """Lazily import iSpec from ispec_dir."""
        import sys as _sys
        if self.ispec_dir not in _sys.path:
            _sys.path.insert(0, self.ispec_dir)
        import ispec as _ispec
        return _ispec

    def _load_cached_data(self):
        """Load linelist, atmospheres, etc. from cache or disk."""
        cache_key = (self.ispec_dir, self.code, self.wave_range_nm[0], self.wave_range_nm[1])
        if cache_key not in ISpecSynthSpec._cache:
            ispec = self._ispec
            ispec_dir = self.ispec_dir

            # Select atmosphere model based on code
            if self.code in ("spectrum", "moog"):
                model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/"
                solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
            else:
                model = ispec_dir + "/input/atmospheres/MARCS.GES/"
                solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"

            atomic_linelist_file = (
                ispec_dir + "/input/linelists/transitions/GESv6_atom_hfs_iso.420_920nm/atomic_lines.tsv"
            )
            isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

            atomic_linelist = ispec.read_atomic_linelist(
                atomic_linelist_file,
                wave_base=self.wave_range_nm[0],
                wave_top=self.wave_range_nm[1],
            )
            atomic_linelist = atomic_linelist[atomic_linelist["theoretical_depth"] >= 0.01]

            ISpecSynthSpec._cache[cache_key] = {
                "modeled_layers_pack": ispec.load_modeled_layers_pack(model),
                "atomic_linelist": atomic_linelist,
                "isotopes": ispec.read_isotope_data(isotope_file),
                "solar_abundances": ispec.read_solar_abundances(solar_abundances_file),
            }

        cached = ISpecSynthSpec._cache[cache_key]
        self._modeled_layers_pack = cached["modeled_layers_pack"]
        self._atomic_linelist = cached["atomic_linelist"]
        self._isotopes = cached["isotopes"]
        self._solar_abundances = cached["solar_abundances"]

    def get_spec(self):
        """
        Synthesize the spectrum and return a ModelSpec object.

        Returns
        -------
        ModelSpec
            Spectrum with wavelength in nm and normalized flux.
        """
        ispec = self._ispec
        wave_base, wave_top = self.wave_range_nm

        # Empirical estimates for broadening
        microturbulence_vel = ispec.estimate_vmic(self.teff, self.logg, self.feh)
        macroturbulence = ispec.estimate_vmac(self.teff, self.logg, self.feh)

        # Interpolate atmosphere
        atmosphere_layers = ispec.interpolate_atmosphere_layers(
            self._modeled_layers_pack,
            {"teff": self.teff, "logg": self.logg, "MH": self.feh, "alpha": self.alpha},
            code=self.code,
        )

        # Create wavelength grid and synthesize
        synth_spectrum = ispec.create_spectrum_structure(
            np.arange(wave_base, wave_top, self.wave_step)
        )
        synth_spectrum["flux"] = ispec.generate_spectrum(
            synth_spectrum["waveobs"],
            atmosphere_layers,
            self.teff, self.logg, self.feh, self.alpha,
            self._atomic_linelist, self._isotopes,
            self._solar_abundances, None,
            microturbulence_vel=microturbulence_vel,
            macroturbulence=macroturbulence,
            vsini=self.vsini,
            limb_darkening_coeff=self.limb_darkening_coeff,
            R=self.resolution,
            regions=None,
            verbose=0,
            code=self.code,
        )

        model_spec = ModelSpec(
            wave_nm=synth_spectrum["waveobs"],
            flux=synth_spectrum["flux"],
            wave_range=self.wave_range_nm,
            rv=0,
        )
        return model_spec


class ModelBinarySpec:
    """
    Reads two spectra and combines them into a single object.
    Currently only works for the synthetic spectra, because the error propagation is not implemented.
    The unnormalized spectra are not implemented as well.
    """

    def __init__(self, model_spec1: ModelSpec, model_spec2: ModelSpec, flux_ratio=1.0):
        self.model_spec1 = deepcopy(model_spec1)
        self.model_spec2 = deepcopy(model_spec2)
        assert len(model_spec1.spectrum["waveobs"]) == len(
            model_spec2.spectrum["waveobs"]
        )
        assert (
            np.abs(
                np.min(model_spec1.spectrum["waveobs"])
                - np.min(model_spec2.spectrum["waveobs"])
            )
            < 1e-6
        )
        assert (
            np.abs(
                np.max(model_spec1.spectrum["waveobs"])
                - np.max(model_spec2.spectrum["waveobs"])
            )
            < 1e-6
        )
        self.flux_ratio = flux_ratio
        self.normalized_spectrum = None
        return

    def _add_two_norm_spec(self):
        flux_fraction_1 = 1 / (1 + self.flux_ratio)
        flux_fraction_2 = self.flux_ratio / (1 + self.flux_ratio)
        self.normalized_spectrum = np.copy(self.model_spec1.normalized_spectrum)
        self.normalized_spectrum["flux"] = (
            self.model_spec1.normalized_spectrum["flux"] * flux_fraction_1
            + self.model_spec2.normalized_spectrum["flux"] * flux_fraction_2
        )
        return

    def normalize_whole_spectrum(self):
        self.model_spec1.normalize_whole_spectrum()
        self.model_spec2.normalize_whole_spectrum()
        # here I assume that the normalizations are perfect
        # so I normalize the fluxes with the flux ratio
        self._add_two_norm_spec()
        return

    def shift_spec(self, v1, v2):
        base_wave = self.model_spec1.normalized_spectrum[
            "waveobs"
        ]  # prepare for interpolation
        self.model_spec1.shift_spec(v1)
        self.model_spec2.shift_spec(v2)
        # only keep the overlapping part, for the interpolation
        wave_min = np.max(
            [
                np.min(self.model_spec1.normalized_spectrum["waveobs"]),
                np.min(self.model_spec2.normalized_spectrum["waveobs"]),
            ]
        )
        wave_max = np.min(
            [
                np.max(self.model_spec1.normalized_spectrum["waveobs"]),
                np.max(self.model_spec2.normalized_spectrum["waveobs"]),
            ]
        )
        base_wave = base_wave[(base_wave > wave_min) & (base_wave < wave_max)]
        # as the shifted wavelengths are different for the two spectra, we need to
        # 1. interpolate the spectra, note that the interpolation only affect the orginal flux
        self.model_spec1.interpolate_new_wave(base_wave)
        self.model_spec2.interpolate_new_wave(base_wave)
        self.normalize_whole_spectrum()
        # then we can add them together with the flux ratio
        self._add_two_norm_spec()
        return

    def interpolate_new_wave(self, new_wave):
        self.model_spec1.interpolate_new_wave(new_wave)
        self.model_spec2.interpolate_new_wave(new_wave)
        self.normalize_whole_spectrum()
        self._add_two_norm_spec()
        return

    def plot_normalized_spectrum(self, ax: plt.Axes, flux_sty_kw: dict = {}):
        ax.plot(
            self.normalized_spectrum["waveobs"],
            self.normalized_spectrum["flux"],
            **flux_sty_kw,
        )
        return


class SynthBinarySpec:
    def __init__(
        self,
        wave_range,
        resolution,
        teff1,
        logg1,
        feh1,
        teff2,
        logg2,
        feh2,
        flux_ratio,
    ) -> None:
        self.synthspec1 = SynthSpec(wave_range, resolution, teff1, logg1, feh1)
        self.synthspec2 = SynthSpec(wave_range, resolution, teff2, logg2, feh2)
        self.flux_ratio = flux_ratio
        return

    def get_spec(self):
        self.obs_spec1 = self.synthspec1.get_spec()
        self.obs_spec2 = self.synthspec2.get_spec()
        obs_binary_spec = ModelBinarySpec(
            self.obs_spec1, self.obs_spec2, self.flux_ratio
        )
        return obs_binary_spec
