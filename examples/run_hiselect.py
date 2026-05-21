"""
Example script: run HiSelect on the Greece region event.

Reads SAC files from the Greece_region event directory, selects the best 8
stations based on SNR, distance score, and azimuthal coverage, then writes
station_file.txt and weights*.dat alongside diagnostic plots.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from obspy import UTCDateTime
from hiselect import HiSelector

EVENT_DIR = '../data/20250204130414/'
ORIG_TIME = UTCDateTime('2025-02-04T13:04:14')
EVLA, EVLO = 36.5658, 25.7261
EVDP       = 10.0      # depth in km
EVMAG      = 5.0

sel = HiSelector(
    event_dir      = EVENT_DIR,
    orig_time      = ORIG_TIME,
    evla           = EVLA,
    evlo           = EVLO,
    evdp           = EVDP,
    evmag          = EVMAG,
    n_select       = 10,
    window_length  = 300,        # surface-wave window length (s)
    noise_duration = 200,        # pre-P noise window length (s)
    filter_dict    = {'fmin': 0.03, 'fmax': 0.06},
    n_clusters     = 8,          # K-means azimuth clusters for CC
    weights        = (1.0, 1.0, 1.0),   # w_snr, w_dist, w_cc
    taup_model     = 'Greece.txt',      # built-in name or path to velocity model file
    dist_range     = (30, 300),  # distance range for station selection in km
)

sel.run()
sel.write_outputs()
sel.copy_selected_data()

# figures for diagnosing the selection.
sel.plot_summary()
sel.plot_waveforms()

print('\nSelected stations:')
for m in sel.selected_stations:
    print(f'  {m["net"]}.{m["sta"]:6s}  dist={m["dist"]:7.1f} km  '
          f'azi={m["azi"]:6.1f} deg')
