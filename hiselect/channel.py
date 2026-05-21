"""
Scan SAC files in an event directory and select one channel per station
using BH > HH > HN priority.  Returns an obspy Stream of Z-component traces
(one per station) that carries the SAC header metadata needed downstream.
"""

import glob
import os
from collections import defaultdict

from obspy import Stream, read

BAND_PRIORITY = ('BH', 'HH', 'HN')


def _loc_priority(loc):
    if loc == '00': return (0, loc)
    if loc == '':   return (1, loc)
    return (2, loc)


def load_unique_stations(event_dir):
    """
    Return an obspy Stream with one Z-component trace per physical station.

    Channel priority: BH* > HH* > HN*.
    Location-code priority: '00' > '' (empty) > others (lexicographic).
    Stations without all three ZRT components are skipped.
    """
    band_files = {b: glob.glob(os.path.join(event_dir, f'*.{b}Z.sac'))
                  for b in BAND_PRIORITY}

    if not any(band_files.values()):
        raise FileNotFoundError(
            f'No BHZ / HHZ / HNZ SAC files found in {event_dir}')

    # (net, sta) → {band: [loc, ...]}
    sta_info = defaultdict(lambda: {b: [] for b in BAND_PRIORITY})
    for band, files in band_files.items():
        for fpath in files:
            parts = os.path.basename(fpath).split('.')
            net, sta, loc = parts[1], parts[2], parts[3]
            sta_info[(net, sta)][band].append(loc)

    st = Stream()
    skipped_bands = []
    for (net, sta), bands in sorted(sta_info.items()):
        chosen_band = None
        for band in BAND_PRIORITY:
            if bands[band]:
                chosen_band = band
                chosen_loc  = sorted(bands[band], key=_loc_priority)[0]
                for lower in BAND_PRIORITY[BAND_PRIORITY.index(band) + 1:]:
                    if bands[lower]:
                        skipped_bands.append(f'{net}.{sta} ({lower}→{band})')
                break
        if chosen_band is None:
            continue

        z_pat = os.path.join(event_dir,
                             f'*{net}.{sta}.{chosen_loc}.{chosen_band}Z.sac')
        z_matches = sorted(glob.glob(z_pat))
        if not z_matches:
            print(f'  WARNING: no Z file for {net}.{sta}.{chosen_loc}.{chosen_band}Z')
            continue

        all_ok = True
        for comp in ('R', 'T'):
            pat = os.path.join(event_dir,
                               f'*{net}.{sta}.{chosen_loc}.{chosen_band}{comp}.sac')
            if not glob.glob(pat):
                print(f'  SKIP {net}.{sta}: missing {chosen_band}{comp}')
                all_ok = False
                break
        if not all_ok:
            continue

        st += read(z_matches[0])[0]

    counts = {b: sum(1 for tr in st if tr.stats.channel.startswith(b))
              for b in BAND_PRIORITY}
    print('Loaded {:d} stations: {}'.format(
        len(st),
        ', '.join(f'{b}: {counts[b]}' for b in BAND_PRIORITY if counts[b])))
    if skipped_bands:
        print('  Dropped lower-priority channels: ' + ', '.join(skipped_bands))
    return st


def load_stream_for_station(event_dir, net, sta, loc, band):
    """Load Z, R, T SAC files for a single station. Returns obspy Stream."""
    st = Stream()
    for comp in ('Z', 'R', 'T'):
        pat = os.path.join(event_dir,
                           f'*{net}.{sta}.{loc}.{band}{comp}.sac')
        matches = sorted(glob.glob(pat))
        if matches:
            st += read(matches[0])[0]
    return st
