from zpec.spec import SynthSpec
from zpec.path import _PHOENIX_DATA_DIR, _TEST_RES_DIR
import pandas as pd
import os
from zpec.utils import PlotUtils, SpecUtils
import matplotlib.pyplot as plt


class TestSynthSpec:
    high_temp_giant_params = (7777, 2.3, 0.0)
    solar_like = (5777, 4.44, 0.0)
    solar_but_high_meh = (5777, 4.44, 0.7)
    low_temp_dwarf_params = (2955, 5.0, 0.2)
    test_params_list = [
        high_temp_giant_params,
        solar_like,
        solar_but_high_meh,
        low_temp_dwarf_params,
    ]
    test_name_list = [
        "high_temp_giant",
        "solar_like",
        "solar_but_high_meh",
        "low_temp_dwarf",
    ]
    params_df = PlotUtils._generate_params_dataframe()

    @staticmethod
    def set_fig_axes():
        # Create a figure with a specific layout of subplots
        fig = plt.figure(figsize=(20, 8))
        # Define the grid for subplots
        ax1 = plt.subplot2grid((3, 4), (0, 0))
        ax2 = plt.subplot2grid((3, 4), (0, 1))
        ax3 = plt.subplot2grid((3, 4), (0, 2))
        ax4 = plt.subplot2grid((3, 4), (0, 3))
        ax5 = plt.subplot2grid((3, 4), (1, 1))
        ax6 = plt.subplot2grid((3, 4), (1, 2))
        ax7 = plt.subplot2grid((3, 4), (2, 1), colspan=2)

        # Adjust layout
        plt.tight_layout()
        return fig, [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

    @staticmethod
    def plot_temp_interp(
        ax,
        spec_list,
        temp_model_spec,
        final_model_spec,
        weight1,
        weight2,
        label1,
        label2,
    ):
        spec_list[0].plot_spectrum(
            ax, flux_sty_kw={"label": label1, "color": "C0", "alpha": weight1}
        )
        spec_list[1].plot_spectrum(
            ax, flux_sty_kw={"label": label2, "color": "C0", "alpha": weight2}
        )
        temp_model_spec.plot_spectrum(
            ax,
            flux_sty_kw={
                "label": "Temp Interpolated Spectrum",
                "color": "C1",
                "alpha": 0.5,
            },
        )
        final_model_spec.plot_spectrum(
            ax, flux_sty_kw={"label": "Final Spectrum", "color": "black"}
        )
        ax.legend(frameon=False)
        return


# Idea
# ----
# We can only use plot to check the output of the SynthSpec class.
# 1. We choose several arbitrary parameters and generate the synthetic spectrum.
# 2. We compare the output with the directly loaded neighbour spectrum.
# 3. The final plot should contain 3 subplots:
#    - Inspect the T_eff neighbour spectrum
#    - Inspect the Logg neighbour spectrum
#    - Inspect the Metallicity neighbour spectrum

if __name__ == "__main__":
    wave_range = (640, 660)
    PlotUtils.plot_params_range(os.path.join(_TEST_RES_DIR, "params_range.png"))
    for params, test_name in zip(
        TestSynthSpec.test_params_list, TestSynthSpec.test_name_list
    ):
        teff, logg, feh = params
        synth_spec = SynthSpec(
            wave_range,
            resolution=2500,
            teff=teff,
            logg=logg,
            feh=feh,
        )
        target_spec = synth_spec.get_spec()
        teff_1, logg_1, feh_1, teff_2, logg_2, feh_2 = synth_spec._get_neighbour()
        (
            weight_teff1,
            weight_teff2,
            weight_logg1,
            weight_logg2,
            weight_feh1,
            weight_feh2,
        ) = synth_spec._get_weight()
        #################################################
        # plot first plot: see if the interpolated ######
        # neighbour spectrum is the same as the #########
        # loaded neighbour spectrum. ####################
        test_synth_spec = SynthSpec(
            wave_range,
            resolution=2500,
            teff=teff_1,
            logg=logg_1,
            feh=feh_1,
        )
        test_model_spec = test_synth_spec.get_spec()
        wave_A, flux = synth_spec._read_synth_spec(teff_1, logg_1, feh_1)
        wave_nm = wave_A / 10
        low_res_wave_nm, low_res_flux, _, _ = SpecUtils.lower_resolution(
            wave_nm, flux, wave_range, 2500, padding=2, save_padding=1
        )  # the setting is the same as in the synth spec

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
        test_model_spec.plot_spectrum(
            ax, flux_sty_kw={"label": "Interpolated Spectrum"}
        )
        ax.set_ylim(ax.get_ylim())
        ax.set_xlim(*wave_range)
        ax.plot(low_res_wave_nm, low_res_flux, label="Phoenix Spectrum (low res)")
        ax.legend()
        plt.savefig(os.path.join(_TEST_RES_DIR, f"test_synth_{test_name}_1.png"))

        # 3. plot second plot:
        #    3.1 in the first row, plot 8 different neighbour spectra in 4 different axes,
        #        each axis represents same teff and logg but different metallicity. Also
        #        plot the final spectra in each axis. Also plot the interpolated spectra
        #        in this step in each axis with synth spec.
        #    3.2 in the second row, do the same thing but with different logg.
        #    3.3 in the thrid (last) row, do the same thing but with different teff.
        fig, axes = TestSynthSpec.set_fig_axes()
        teff_spec_list = []
        for i, temp_teff in enumerate([teff_1, teff_2]):
            logg_spec_list = []
            for j, temp_logg in enumerate([logg_1, logg_2]):
                feh_spec_list = []
                for k, temp_feh in enumerate([feh_1, feh_2]):
                    # 1. get neighbour spectrum
                    temp_synth_spec = SynthSpec(
                        wave_range,
                        resolution=2500,
                        teff=temp_teff,
                        logg=temp_logg,
                        feh=temp_feh,
                    )  # since we tested the interpolation perfectly reproduces the
                    # spectrum with parameters just on the grid, we simply use it here.
                    feh_spec_list.append(temp_synth_spec.get_spec())
                temp_synth_spec = SynthSpec(
                    wave_range,
                    resolution=2500,
                    teff=temp_teff,
                    logg=temp_logg,
                    feh=feh,
                )  # only the feh is finished interpolation,
                # check if the first interpolation is correct.
                temp_model_spec = temp_synth_spec.get_spec()
                logg_spec_list.append(temp_model_spec)
                TestSynthSpec.plot_temp_interp(
                    axes[i * 2 + j],
                    feh_spec_list,
                    temp_model_spec,
                    target_spec,
                    weight_feh1,
                    weight_feh2,
                    f"feh={feh_1}",
                    f"feh={feh_2}",
                )
            temp_synth_spec = SynthSpec(
                wave_range,
                resolution=2500,
                teff=temp_teff,
                logg=logg,
                feh=feh,
            )
            temp_model_spec = temp_synth_spec.get_spec()
            teff_spec_list.append(temp_model_spec)
            TestSynthSpec.plot_temp_interp(
                axes[4 + i],
                logg_spec_list,
                temp_model_spec,
                target_spec,
                weight_logg1,
                weight_logg2,
                f"logg={logg_1}",
                f"logg={logg_2}",
            )
        temp_synth_spec = SynthSpec(
            wave_range,
            resolution=2500,
            teff=teff,
            logg=logg,
            feh=feh,
        )
        temp_model_spec = temp_synth_spec.get_spec()
        TestSynthSpec.plot_temp_interp(
            axes[6],
            teff_spec_list,
            temp_model_spec,
            target_spec,
            weight_teff1,
            weight_teff2,
            f"teff={teff_1}",
            f"teff={teff_2}",
        )

        fig.suptitle(f"T_eff={teff}, logg={logg}, feh={feh}")
        fig.tight_layout()
        plt.savefig(os.path.join(_TEST_RES_DIR, f"test_synth_{test_name}_2.png"))
