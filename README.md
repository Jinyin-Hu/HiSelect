# HiSelect

Python package for automated seismic station selection based on SNR, waveform cross-correlation, epicentral distance, and azimuthal coverage.

Given a directory of ZRT SAC files for a single event, HiSelect scores every candidate station and iteratively picks the subset that jointly maximises data quality and azimuthal coverage — ready for moment-tensor inversion or other waveform-based analyses.

## Selection criteria

| Metric | Description |
|---|---|
| SNR | Signal-to-noise ratio over the S-wave window vs. pre-P noise |
| CC | Peak cross-correlation with the azimuth-sector waveform stack |
| Distance score | Magnitude-dependent epicentral distance weight (Scognamiglio et al. 2009) |
| Azimuthal coverage | Effective station number C (Ekström 2006, Eq. 10) |

Station 1 is chosen by the highest combined score. Each subsequent station minimises the combined rank of score and azimuthal coverage, so the selected set fills azimuthal gaps while maintaining high data quality.

## Requirements

- Python ≥ 3.8
- [ObsPy](https://docs.obspy.org/) ≥ 1.4
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)

Optional:

- [Cartopy](https://scitools.org.uk/cartopy/) — for geographic station maps (falls back to plain Matplotlib if absent)
- [pysep](https://github.com/uafgeotools/pysep) — for writing CAP-format `weights*.dat` files (falls back to a built-in writer if absent)

## Installation

```bash
git clone git@github.com:Jinyin-Hu/HiSelect.git
cd HiSelect
pip install -e .
```

## Input data

HiSelect reads three-component (Z, R, T) SAC files from a single event directory. Files must follow the naming convention:

```
{eventid}.{net}.{sta}.{loc}.{band}{comp}.sac
```

For stations with multiple channel types, the priority order BH > HH > HN is applied automatically.

## Example

See the [script](./examples/run_hiselect.py) for an example.

Output files

| File | Description |
|---|---|
| `station_file.txt` | One line per selected station: STA NET LAT LON DIST BAZ |
| `weights.dat` | CAP-format weight file |
| `weights_body.dat` | CAP-format body-wave weights |
| `weights_surf.dat` | CAP-format surface-wave weights |
| `selected/` | Copy of selected SAC files and all output files |
| `hiselect_map.png` | Station map (selected in blue, unselected in grey) |
| `hiselect_scores.png` | Per-station SNR, distance, CC, and combined score bar charts |
| `hiselect_waveforms.png` | Record-section plot of selected stations' signal windows |

## References

- Hu, J., Tkalčić, H., Phạm, T.-S. & Hejrani, B., 2026. Improved joint constraints on moment tensors, source time functions and uncertainty quantification for moderately large earthquakes. Geophys J Int, 246, ggag154. doi:10.1093/gji/ggag154
