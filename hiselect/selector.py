"""
Core station-selection algorithm modified from Ekström (2006).

Selection strategy
------------------
Station 1 : highest combined score  score = w_snr*SNR_norm + w_dist*D + w_cc*CC

Stations 2…N : minimise the combined rank of
    - station score (rank_score, same formula as above)
    - azimuthal coverage (rank_azi), measured by the Ekström (2006) Eq.10
      metric C = 1 / Σ (gap_i / 2π)² — higher C means more uniform coverage.

Combined score = SNR (normalised) + D (distance) + CC (normalised within azimuth sector)
"""

import numpy as np
from obspy.core import UTCDateTime

from .channel import load_unique_stations, load_stream_for_station
from .metrics import (filter_stream, cut_signal_window, cut_noise_window,
                      compute_snr, compute_cc, distance_score,
                      azimuth_coverage)
from .io import write_station_file, write_weights, copy_selected_data


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def rank_data_score(data, tiebreak=None):
    """
    Rank 1-D array in ascending order (rank 1 = smallest value).
    Ties are broken by *tiebreak* (ascending; smallest wins).
    If *tiebreak* is None, ties are broken by position (stable, deterministic).
    """
    data = np.asarray(data, dtype=float)
    n    = len(data)
    tb   = np.asarray(tiebreak, dtype=float) if tiebreak is not None else np.arange(n, dtype=float)
    order  = np.lexsort((tb, data))
    ranks  = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    return ranks



# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HiSelector:
    """
    Select the best N stations from an event directory of SAC files.

    Parameters
    ----------
    event_dir      : str
        Directory containing ZRT SAC files named
        ``{eventid}.{net}.{sta}.{loc}.{band}{comp}.sac``.
    orig_time      : str or obspy.UTCDateTime
        Event origin time.
    evla, evlo     : float
        Event latitude and longitude (degrees).
    evdp           : float
        Event depth (km).
    evmag          : float
        Event magnitude.
    n_select       : int
        Number of stations to select (default 8).
    window_length  : float
        Surface-wave signal window length in seconds (default 300).
    noise_duration : float
        Pre-P noise window length in seconds (default 200).
    filter_dict    : dict or None
        Bandpass filter parameters: fmin, fmax, corners, zerophase.
        Pass None to skip filtering entirely (raw data used for metrics).
    n_clusters     : int
        Number of K-means azimuth clusters for CC computation (default 8).
    weights        : tuple of three floats
        (w_snr, w_dist, w_cc) weights for the combined score (default 1,1,1).
    taup_model     : str
        TauPy velocity model for P and S arrivals (default 'ak135').
        May be a built-in name or a path to a plain-text velocity model file.
    dist_range     : tuple of (float or None, float or None)
        (min_dist_km, max_dist_km) epicentral distance filter applied before
        any metric computation.  Use None for no limit on either end.
        Example: (100, 1500) keeps only stations between 100 and 1500 km.
    """

    def __init__(self, event_dir, orig_time, evla, evlo, evdp, evmag,
                 n_select=8, window_length=300, noise_duration=200,
                 filter_dict=None, n_clusters=8,
                 weights=(1.0, 1.0, 1.0), taup_model='ak135',
                 window_alignment=0.3, dist_range=(None, None)):

        self.event_dir      = event_dir
        self.orig_time      = UTCDateTime(orig_time)
        self.evla           = evla
        self.evlo           = evlo
        self.evdp           = evdp
        self.evmag          = evmag
        self.n_select       = n_select
        self.window_length  = window_length
        self.noise_duration = noise_duration
        self.filter_dict    = filter_dict   # None → no filtering
        self.n_clusters     = n_clusters
        self.w_snr, self.w_dist, self.w_cc = weights
        self.taup_model     = taup_model
        self.window_alignment = window_alignment
        self.dist_min, self.dist_max = dist_range

        # populated by run()
        self.meta           = []     # list of dicts, one per candidate station
        self.signal_streams = []     # filtered, windowed Stream per candidate
        self.snr_scores     = None   # shape (ns,)
        self.cc_scores      = None
        self.dist_scores    = None
        self.combined       = None
        self.selected_idx   = []     # indices into self.meta

    # ------------------------------------------------------------------
    def run(self):
        """
        Full pipeline:
        1. Scan SAC files, apply channel priority.
        2. Filter, cut signal and noise windows.
        3. Compute SNR, CC, and distance scores.
        4. Run iterative station selection.
        """
        print('=== HiSelect: loading stations ===')
        z_stream = load_unique_stations(self.event_dir)

        print('=== HiSelect: computing metrics ===')
        signal_streams, noise_streams, meta = [], [], []

        for tr in z_stream:
            net     = tr.stats.network
            sta     = tr.stats.station
            loc     = tr.stats.location
            band    = tr.stats.channel[:2]
            dist_km = tr.stats.sac.dist
            azi     = tr.stats.sac.az
            baz     = tr.stats.sac.baz
            stla    = tr.stats.sac.stla
            stlo    = tr.stats.sac.stlo

            if self.dist_min is not None and dist_km < self.dist_min:
                # print(f'  SKIP {net}.{sta}: dist {dist_km:.0f} km < {self.dist_min} km')
                continue
            if self.dist_max is not None and dist_km > self.dist_max:
                # print(f'  SKIP {net}.{sta}: dist {dist_km:.0f} km > {self.dist_max} km')
                continue

            full_st = load_stream_for_station(
                self.event_dir, net, sta, loc, band)
            if len(full_st) < 3:
                # print(f'  SKIP {net}.{sta}: could not load ZRT')
                continue

            filtered = (filter_stream(full_st, self.filter_dict)
                        if self.filter_dict is not None else full_st.copy())
            sig_st   = cut_signal_window(
                filtered, self.orig_time, dist_km, self.window_length,
                taup_model=self.taup_model,
                window_alignment=self.window_alignment,
                depth_km=self.evdp)
            noi_st   = cut_noise_window(
                filtered, self.orig_time, dist_km, self.evdp,
                self.noise_duration, self.taup_model)

            if len(sig_st) == 0 or len(noi_st) == 0:
                # print(f'  SKIP {net}.{sta}: empty window')
                continue

            signal_streams.append(sig_st)
            noise_streams.append(noi_st)
            meta.append(dict(net=net, sta=sta, loc=loc, band=band,
                             dist=dist_km, azi=azi, baz=baz,
                             stla=stla, stlo=stlo, tr_z=tr))

        if not meta:
            raise RuntimeError('No stations passed window-cutting checks.')

        ns      = len(meta)
        self.meta           = meta
        self.signal_streams = signal_streams
        azi_arr  = np.array([m['azi']  for m in meta])

        # --- SNR (normalised) ---
        snr_raw = np.array([compute_snr(signal_streams[s], noise_streams[s])
                            for s in range(ns)])
        snr_norm = snr_raw / np.max(snr_raw) if np.max(snr_raw) > 0 else snr_raw
        self.snr_scores = snr_norm

        # --- CC ---
        print('=== HiSelect: cross-correlation scores ===')
        cc = compute_cc(signal_streams, azi_arr,
                        n_clusters=min(self.n_clusters, ns))
        self.cc_scores = cc

        # --- Distance score ---
        d_score = np.array([distance_score(m['dist'], self.evmag) for m in meta])
        self.dist_scores = d_score

        # --- Combined ---
        self.combined = (self.w_snr  * snr_norm +
                         self.w_dist * d_score  +
                         self.w_cc   * cc)

        # --- Selection ---
        print('=== HiSelect: selecting stations ===')
        self._select()
        print(f'Selected {len(self.selected_idx)} / {ns} stations.')

    # ------------------------------------------------------------------
    def _select(self):
        """Iterative selection: first by score, then by combined score+coverage."""
        ns    = len(self.meta)
        score = self.combined
        azi   = np.array([m['azi']  for m in self.meta])
        dist  = np.array([m['dist'] for m in self.meta])

        # Station 1: highest combined score; ties broken by distance (nearest wins)
        rank_all = rank_data_score(
            1.0 / np.where(score > 0, score, 1e-10), tiebreak=dist)
        i_first  = int(np.argmin(rank_all))
        selected  = [i_first]
        remaining = list(range(ns))
        remaining.remove(i_first)
        print(f'  #1  {self.meta[i_first]["net"]}.{self.meta[i_first]["sta"]}'
              f'  score={score[i_first]:.3f}')

        # Stations 2 … n_select
        for step in range(1, self.n_select):
            if not remaining:
                break
            # effective number of stations C for each candidate (higher = better)
            az_cov   = np.array([
                azimuth_coverage(azi[selected], azi[k])
                for k in remaining])
            dist_rem = dist[remaining]
            print(f'  #{step+1}  effective N (best candidate) = '
                  f'{az_cov.max():.2f}')

            rank_azi   = rank_data_score(-az_cov, tiebreak=dist_rem)   # negate: rank 1 = highest C
            score_rem  = score[remaining]
            rank_score = rank_data_score(
                1.0 / np.where(score_rem > 0, score_rem, 1e-10), tiebreak=dist_rem)

            combined_rank = rank_azi + rank_score
            best_local    = int(np.argmin(rank_data_score(combined_rank, tiebreak=dist_rem)))
            i_next        = remaining[best_local]

            selected.append(i_next)
            remaining.remove(i_next)
            print(f'  #{step+1}  {self.meta[i_next]["net"]}.{self.meta[i_next]["sta"]}'
                  f'  score={score[i_next]:.3f}')

        self.selected_idx = selected

    # ------------------------------------------------------------------
    def write_outputs(self, path_out=None):
        """Write station_file.txt and weights*.dat."""
        if not self.selected_idx:
            raise RuntimeError('Call run() before write_outputs().')
        path_out = path_out or self.event_dir
        sel_meta = [self.meta[i] for i in self.selected_idx]
        write_station_file(sel_meta, path_out)
        write_weights(sel_meta, path_out)

    # ------------------------------------------------------------------
    def plot_summary(self, path_out=None):
        """Station map and per-station score bar charts."""
        from .plot import plot_map, plot_scores
        path_out = path_out or self.event_dir
        plot_map(self.meta, self.selected_idx,
                 self.evla, self.evlo, self.evmag, path_out)
        plot_scores(self.meta, self.snr_scores, self.dist_scores,
                    self.cc_scores, self.combined, self.selected_idx, path_out)

    # ------------------------------------------------------------------
    def plot_waveforms(self, path_out=None):
        """Record-section waveform plot for the selected stations."""
        if not self.selected_idx:
            raise RuntimeError('Call run() before plot_waveforms().')
        from .plot import plot_waveforms
        path_out = path_out or self.event_dir
        plot_waveforms(self.meta, self.selected_idx, self.signal_streams,
                       self.orig_time, path_out)

    # ------------------------------------------------------------------
    def copy_selected_data(self, path_out=None):
        """Copy selected SAC files and output files into {path_out}/selected/."""
        if not self.selected_idx:
            raise RuntimeError('Call run() before copy_selected_data().')
        path_out = path_out or self.event_dir
        sel_meta = [self.meta[i] for i in self.selected_idx]
        copy_selected_data(sel_meta, self.event_dir, path_out)

    # ------------------------------------------------------------------
    @property
    def selected_stations(self):
        """List of metadata dicts for the selected stations."""
        return [self.meta[i] for i in self.selected_idx]
