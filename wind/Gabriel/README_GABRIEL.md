# Gabriel Standalone Jump-Intensity Bundle

This folder is self-contained for the two notebooks:

- `Jump_intensity_analysis.ipynb`
- `Jump_intensity_final_new.ipynb`

Run Jupyter from this folder, or open the notebooks from a parent folder that still contains this `Gabriel` directory. The notebooks locate the bundle automatically and write outputs to:

- `jump_intensity_analysis_outputs/`
- `jump_intensity_final_new_outputs/`

## Required Inputs Included

- `DayAheadPrices_2021_2025.csv`
- `enwex_GER_wind_v25_combined.csv`
- `enwex_GER_solar_v25_combined.csv`
- `Important_Data_all_included.csv`
- `entsoe_de_lu_load_extended_model.csv`
- `openmeteo_temperature_history_extended_model.csv`

No ENTSO-E API key is required for the notebooks, because the cached `entsoe_de_lu_load_extended_model.csv` file is included and auto-download is disabled.

## Python Files Included

The folder includes the intensity model modules needed by the notebooks, including the final calibration code, corrected jump-intensity code, stochastic covariance code, Wishart helpers, seasonality code, data loaders, and diagnostics helpers.

## Environment

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Then run:

```bash
jupyter lab
```

The main final model is in `Jump_intensity_final_new.ipynb`. `Jump_intensity_analysis.ipynb` is the side analysis that motivates and checks the corrected jump event definition.

## Standalone Changes

- Notebook root detection now uses the local `Gabriel` folder instead of the original repository root.
- Notebook input paths now point to CSV files stored directly in this folder.
- Notebook output paths now point to output directories inside this folder.
- The copied loader modules now default to the bundled local CSV paths.
- `entsoe-py` is no longer required just to load the cached ENTSO-E CSV.
- The copied Wishart helper no longer imports the original repo's `application` package during notebook startup.
- `np.trapz` calls were made compatible with newer NumPy versions by using `np.trapezoid` when available.
