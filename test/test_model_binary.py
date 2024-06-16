import numpy as np
import matplotlib.pyplot as plt
from zpec.spec import SynthBinarySpec, ModelBinarySpec
from zpec.path import _TEST_DATA_DIR, _TEST_RES_DIR
import os
from pytest import approx

if __name__ == "__main__":
    teff1 = 6000
    teff2 = 5000
    logg1 = 4.5
    logg2 = 4.5
    feh1 = 0.0
    feh2 = 0.0
    flux_ratio_list = [0.1, 0.3, 0.7, 1.0]

    #################################
    ##### start test flux ratio #####
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    axes = axes.flatten()
    for flux_ratio, ax in zip(flux_ratio_list, axes):
        synth_binary = SynthBinarySpec(
            (640, 660), 5000, teff1, logg1, feh1, teff2, logg2, feh2, flux_ratio
        )
        model_binary = synth_binary.get_spec()
        model_binary.normalize_whole_spectrum()
        model_binary.plot_normalized_spectrum(
            ax=ax, flux_sty_kw={"color": "k", "alpha": 0.5}
        )
        model_binary.model_spec1.plot_normalized_spectrum(
            ax=ax, flux_sty_kw={"color": "C0", "alpha": 0.5, "label": f"Teff = {teff1}"}
        )
        model_binary.model_spec2.plot_normalized_spectrum(
            ax=ax, flux_sty_kw={"color": "C1", "alpha": 0.5, "label": f"Teff = {teff2}"}
        )
        flux_frac_1 = 1 / (1 + flux_ratio)
        ax.plot(
            model_binary.model_spec1.normalized_spectrum["waveobs"],
            model_binary.model_spec1.normalized_spectrum["flux"] * flux_frac_1,
            color="C0",
            alpha=flux_frac_1,
        )
        flux_frac_2 = flux_ratio / (1 + flux_ratio)
        ax.plot(
            model_binary.model_spec2.normalized_spectrum["waveobs"],
            model_binary.model_spec2.normalized_spectrum["flux"] * flux_frac_2,
            color="C1",
            alpha=flux_frac_2,
        )
        ax.legend()
        ax.set_title(f"flux ratio = {flux_ratio}")
    plt.tight_layout()
    plt.savefig(os.path.join(_TEST_RES_DIR, "test_flux_ratio.png"))
    ##### end test flux ratio #####
    ###############################

    #################################
    ##### start test spec shift #####
    synth_binary = SynthBinarySpec(
        (640, 660), 5000, teff1, logg1, feh1, teff2, logg2, feh2, flux_ratio
    )
    model_binary = synth_binary.get_spec()
    model_binary_base = synth_binary.get_spec()
    model_binary.normalize_whole_spectrum()
    model_binary_base.normalize_whole_spectrum()
    v1 = 200
    v2 = -300
    model_binary.shift_spec(v1, v2)

    # Create a figure with the specified layout: two small axes at the top and one large at the bottom
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ## Create the two small axes at the top
    # ax1 = plt.subplot2grid((2, 2), (0, 0))
    # ax2 = plt.subplot2grid((2, 2), (0, 1))
    ## Create the large axis at the bottom
    # ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
    ax1, ax2, ax3 = fig.subplots(3, 1)

    # for ax1 plot the model 1
    model_binary.model_spec1.plot_normalized_spectrum(
        ax=ax1, flux_sty_kw={"color": "C0", "label": f"model 1, v1 = {v1}"}
    )
    model_binary_base.model_spec1.plot_normalized_spectrum(
        ax=ax1, flux_sty_kw={"color": "C1", "label": "model 1 base"}
    )
    ax1.legend()
    ax1.set_xlim(640, 660)
    # for ax2 plot the model 2
    model_binary.model_spec2.plot_normalized_spectrum(
        ax=ax2, flux_sty_kw={"color": "C0", "label": f"model 2, v2 = {v2}"}
    )
    model_binary_base.model_spec2.plot_normalized_spectrum(
        ax=ax2, flux_sty_kw={"color": "C1", "label": "model 2 base"}
    )
    ax2.legend()
    ax2.set_xlim(640, 660)
    # for ax3 plot the combined model
    model_binary.plot_normalized_spectrum(
        ax=ax3, flux_sty_kw={"color": "k", "label": f"combined, v1 = {v1}, v2 = {v2}"}
    )
    model_binary_base.plot_normalized_spectrum(
        ax=ax3, flux_sty_kw={"color": "r", "label": "combined base"}
    )
    ax3.legend()
    ax3.set_xlim(640, 660)
    ax3.set_xlabel("Wavelength (nm)")

    plt.tight_layout()
    plt.savefig(os.path.join(_TEST_RES_DIR, "test_spec_shift.png"))
