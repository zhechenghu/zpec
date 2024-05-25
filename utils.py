import numpy as np


class SpecUtils:
    @staticmethod
    def lower_reslution(
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
        mask = np.where(
            (wave >= wave_range[0] - padding) & (wave <= wave_range[-1] + padding)
        )[
            0
        ]  # chunk index
        delta_lambda = np.median(
            abs(wave[mask[0 : (len(mask) - 1)]] - wave[mask[1:]])
        )  # samping size
        sigma_lambda = (
            np.median(wave[mask]) / r
        )  # the change in lambda of two peaks, derived from Resolution = lambda/delta_lambda
        sigma = (
            sigma_lambda / delta_lambda / 2.35
        )  # Assume that the psf is an gaussian, then 2.35 * sigma_psf = sigma_lambda
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
