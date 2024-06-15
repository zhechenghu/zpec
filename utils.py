import numpy as np
from zpec.path import _PHOENIX_DATA_DIR
import os
import re
import pandas as pd
import matplotlib.pyplot as plt


class SpecUtils:
    @staticmethod
    def lower_resolution(
        wave,
        flux,
        wave_range,
        r,
        padding=2,
        save_padding=1,
    ):
        """
        Convolve original template spec with given psf, psf is fixed now.

        Args:
            star_wave (array): original template wave
            star_spec (array): original template spec
            wave (array): wavelength range of this trace, saving blaze>0.05
            r (float): resolution, different at different band
            padding (float): padding of the wave range to avoid edge effect
            save_padding (float): padding of the wave range to save, should be smaller than padding

        Returns: convolved wave and spec, used x pixel and psf

        """
        # os_factor = 1.0  # over sampling factor for the simulation
        # boundaries of each orders
        # fmt: off
        mask = np.where((wave >= wave_range[0] - padding) & (wave <= wave_range[-1] + padding))[0]  # chunk index
        delta_lambda = np.median(abs(wave[mask[0 : (len(mask) - 1)]] - wave[mask[1:]]))  # samping size
        sigma_lambda = (np.median(wave[mask]) / r)  # the change in lambda of two peaks, derived from Resolution = lambda/delta_lambda
        sigma = (sigma_lambda / delta_lambda / 2.35)  # Assume that the psf is an gaussian, then 2.35 * sigma_psf = sigma_lambda
        # fmt: on
        # construct PSF
        # x_psf = np.arange(-15*3, 15*3, 1.0/os_factor)
        x_psf = np.arange(-15 * 3, 15 * 3, 1.0)
        psf = np.exp(-0.5 * (x_psf / sigma) ** 2)
        psf /= np.trapz(psf, x_psf)
        spec_convolved = np.convolve(flux[mask], psf, "same")
        # grab usable part without padding
        wave_convolved = wave[mask]
        final_mask = np.where(
            (wave_convolved >= (wave_range[0] - save_padding))
            & (wave_convolved <= wave_range[-1] + save_padding)
        )[0]
        synwave = wave_convolved[final_mask]
        synspec = spec_convolved[final_mask]

        return synwave, synspec, x_psf, psf


class PlotUtils:
    @staticmethod
    def _parse_file_name(file_name):
        pattern = r"lte(\d+)-([-+]?\d+\.\d+)([-+]\d+\.\d+)"
        match = re.search(pattern, file_name)
        if match:
            teff = int(match.group(1))
            logg = float(match.group(2))
            metallicity = float(match.group(3))
            return teff, logg, metallicity
        else:
            return None, None, None

    @staticmethod
    def _generate_params_dataframe():
        file_names = os.listdir(_PHOENIX_DATA_DIR)
        # Create a list to store the parsed data
        data = []
        for file_name in file_names:
            teff, logg, metallicity = PlotUtils._parse_file_name(file_name)
            if teff is not None and logg is not None and metallicity is not None:
                data.append((teff, logg, metallicity, file_name))

        # Create a DataFrame
        df = pd.DataFrame(data, columns=["Teff", "logg", "feh", "File Name"])
        return df

    @staticmethod
    def plot_params_range(plot_save_path):
        df = PlotUtils._generate_params_dataframe()
        # plot a corner plot
        fig, axes = plt.subplots(2, 2, figsize=(5, 5), dpi=300)
        ax1 = axes[0, 0]
        ax1.scatter(df["Teff"], df["logg"], s=1)
        ax1.set_xticks([])
        ax1.set_ylabel("logg")
        ax2 = axes[1, 0]
        ax2.scatter(df["Teff"], df["feh"], s=1)
        ax2.set_xlabel("T_eff (k)")
        ax2.set_ylabel("[Fe/H]")
        ax3 = axes[1, 1]
        ax3.scatter(df["logg"], df["feh"], s=1)
        ax3.set_yticks([])
        ax3.set_xlabel("logg")
        axes[0, 1].axis("off")
        fig.tight_layout()
        plt.savefig(plot_save_path)
