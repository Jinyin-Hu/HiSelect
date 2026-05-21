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
git clone https://github.com/<your-username>/HiSelect.git
cd HiSelect
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[cartopy,pysep]"
```

## Input data

HiSelect reads three-component (Z, R, T) SAC files from a single event directory. Files must follow the naming convention:

```
{eventid}.{net}.{sta}.{loc}.{band}{comp}.sac
```

For stations with multiple channel types, the priority order BH > HH > HN is applied automatically.

## Velocity model

The `taup_model` argument controls how P and S arrival times are computed. Two options are supported:

**Built-in ObsPy model** (ak135, prem, iasp91, …):
```python
taup_model = 'ak135'
```

**User-supplied plain-text model file** (e.g. `Greece.txt`):
```
# H(km)  Vp(km/s)  Vs(km/s)  rho(g/cc)  Qp   Qs
  5.0     5.80      3.36      2.60       600  300
 15.0     6.20      3.58      2.75       600  300
 10.0     6.80      3.93      2.90       600  300
  0.0     8.10      4.65      3.35       600  300
```

Columns: layer thickness, Vp, Vs, and optionally density, Qp, Qs. Header lines and `#` comments are skipped automatically. Q values < 1 are treated as 1/Q attenuation (CPS convention) and inverted.

## Example

See the [script](./examples/run_hiselect.py)

### Output files

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
