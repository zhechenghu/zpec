import numpy as np
import matplotlib.pyplot as plt
from zpec.spec import ObsSpec, ModelSpec
from zpec.path import _TEST_DATA_DIR
import os
from pytest import approx


class Utils:
    @staticmethod
    def assert_lenth(spec: ObsSpec):
        lenwave = len(spec.spectrum["waveobs"])
        lenflux = len(spec.spectrum["flux"])
        lenerr = len(spec.spectrum["err"])
        assert lenwave == lenflux == lenerr


class TestObsSpec:
    fits_file = os.path.join(
        _TEST_DATA_DIR,
        "spec1d_0003973582-20231104-OSIRIS-OsirisLongSlitSpectroscopy-Bernhard2_OSIRIS_20231105T055016.199.fits",
    )
    txt_file = os.path.join(
        _TEST_DATA_DIR,
        "Bernhard_2023-11-05T05:49:40.010.txt",
    )
    model_file = os.path.join(
        _TEST_DATA_DIR,
        "bernhard_template_GTC_04900-4.50+0.5.txt",
    )

    def test_txt_loading(self):
        spec = ObsSpec(
            self.txt_file,
            spectrum_format="txt",
            date_obs="2023-11-05T05:49:40.010",
            vel_corr=1,
            vel_type="heliocentric",
        )
        Utils.assert_lenth(spec)

    def test_fits_loading(self):
        spec = ObsSpec(self.fits_file)
        Utils.assert_lenth(spec)

    def test_restrict_wave_range(self):
        spec = ObsSpec(self.fits_file, wave_range=(620, 640))
        Utils.assert_lenth(spec)
        wave = spec.spectrum["waveobs"]
        wave_min = min(wave)
        wave_max = max(wave)
        assert 620 <= wave_min <= wave_max <= 640

    def test_errfac(self):
        spec = ObsSpec(self.fits_file, errfac=2.0)
        spec_origin = ObsSpec(self.fits_file, errfac=1.0)
        tot_err = np.sum(spec.spectrum["err"])
        tot_err_origin = np.sum(spec_origin.spectrum["err"])
        assert tot_err == approx(2.0 * tot_err_origin, abs=1e-6)

    def test_reset_errfac(self):
        spec = ObsSpec(self.fits_file, errfac=1.0)
        err = np.copy(spec.spectrum["err"])
        spec.reset_errfac(3.0)
        new_err = np.copy(spec.spectrum["err"])
        assert np.sum(new_err) == approx(3.0 * np.sum(err), abs=1e-6)


if __name__ == "__main__":
    obs_spec_base = ObsSpec(TestObsSpec.fits_file)
    print(obs_spec_base)
    obs_spec_base.normalize_whole_spectrum()
    result_dir_path = os.path.join(os.path.dirname(__file__), "result")

    #####################################
    ##### start test shift spectrum #####
    obs_spec = ObsSpec(
        TestObsSpec.txt_file,
        spectrum_format="txt",
        date_obs="2023-11-05T05:49:40.010",
        vel_corr=1,
        vel_type="heliocentric",
    )
    obs_spec.shift_spec(500)  # 500 km/s at 6000 angstrom is about 1 nm
    obs_spec.normalize_whole_spectrum()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    obs_spec.plot_normalized_spectrum(
        ax=ax, flux_sty_kw={"alpha": 0.2, "label": "shifted + 500 km/s"}
    )
    obs_spec_base.plot_normalized_spectrum(
        ax=ax, flux_sty_kw={"alpha": 0.2, "label": "original"}
    )
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Flux")
    ax.set_title("Test Shift spectrum (~ 1 nm from peak to peak)")
    ax.set_xlim(610, 620)
    ax.set_xticks(np.arange(610, 621, 1))
    ax.grid()
    ax.legend()
    fig.savefig(os.path.join(result_dir_path, "test_shift_spectrum.png"))
    ##### end test shift spectrum #####
    ###################################

    #########################################
    ##### start test normalize spectrum #####
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    obs_spec_base.plot_continuum_fit(ax=ax)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Flux")
    ax.set_title("Continuum fit before normalization")
    fig.savefig(os.path.join(result_dir_path, "test_normalize_spectrum.png"))
    ##### end test normalize spectrum #####
    #######################################

    #######################################
    ##### start test flatten spectrum #####
    obs_spec = ObsSpec(TestObsSpec.fits_file)
    obs_spec.flatten_spectrum((600, 620))
    obs_spec.normalize_whole_spectrum()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    obs_spec.plot_spectrum(ax=ax, flux_sty_kw={"alpha": 0.2, "label": "flattened"})
    obs_spec_base.plot_continuum_fit(
        ax=ax, flux_sty_kw={"alpha": 0.2, "label": "original"}
    )
    # froze the xlim and ylim
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    ax.vlines(600, 0, 1e9, color="red", linestyle="--")
    ax.vlines(620, 0, 1e9, color="red", linestyle="--")
    ax.set_xlim(590, 630)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Flux")
    ax.set_title("Test Flattened spectrum")
    ax.legend()
    fig.savefig(os.path.join(result_dir_path, "test_flatten_spectrum.png"))
    plt.close(fig)
    ##### end test flatten spectrum #####
    #####################################

    ###########################################
    ##### start test interpolate spectrum #####
    model_spec_base = ModelSpec(TestObsSpec.model_file, wave_range=(590, 630))
    model_spec = ModelSpec(TestObsSpec.model_file, wave_range=(590, 630))
    new_wave = model_spec_base.spectrum["waveobs"] + np.random.normal(
        0, 0.1, len(model_spec_base.spectrum["waveobs"])
    )
    new_wave_mask = (new_wave > np.min(model_spec_base.spectrum["waveobs"])) & (
        new_wave < np.max(model_spec_base.spectrum["waveobs"])
    )
    new_wave = np.sort(new_wave[new_wave_mask])
    model_spec.interpolate_new_wave(new_wave)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    model_spec.plot_spectrum(ax=ax, flux_sty_kw={"alpha": 0.2, "label": "interploated"})
    model_spec_base.plot_spectrum(
        ax=ax, flux_sty_kw={"alpha": 0.2, "label": "original"}
    )
    # froze the xlim and ylim
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    ax.set_xlim(590, 630)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Flux")
    ax.set_title("Test Flattened spectrum")
    ax.legend()
    fig.savefig(os.path.join(result_dir_path, "test_interp_spectrum.png"))
    plt.close(fig)
    ##### end test interpolate spectrum #####
    #########################################
