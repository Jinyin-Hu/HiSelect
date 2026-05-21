"""
Per-station quality metrics: SNR, cross-correlation (CC), and distance score.

SNR  — ratio of signal RMS (surface-wave window) to pre-P noise RMS (dB).
CC   — normalised peak cross-correlation with the azimuth-sector stack.
D    — magnitude-based epicentral distance score (Scognamiglio et al. 2009).

score = w_snr * SNR_norm + w_dist * D + w_cc * CC
"""

import os
import tempfile
import warnings
import numpy as np
from sklearn.cluster import KMeans

_taup_model_cache = {}   # model name or file path → TauPyModel
_velmodel_nd_cache     = {}   # CPS file path → converted .nd temp file path


def _velmodel_to_nd(model_path):
    """
    Convert a plain-text velocity model to an ObsPy nd temp file.

    Expected columns (whitespace- or comma-separated):
        H(km)  Vp(km/s)  Vs(km/s)  [rho(g/cc)  [Qp  [Qs]]]

    Lines that cannot be parsed as numbers (headers, comments) are skipped
    automatically, so any text file with those columns works regardless of
    format or origin.

    Q columns are auto-detected:
      - value >= 1  → treated as Q directly
      - 0 < value < 1 → treated as 1/Q attenuation and inverted
      - value == 0  → mapped to Q = 9999 (effectively no attenuation)

    The resulting .nd file is cached so conversion happens only once per path.
    """
    if model_path in _velmodel_nd_cache:
        return _velmodel_nd_cache[model_path]

    layers = []
    with open(model_path) as fh:
        for line in fh:
            # strip inline comments and whitespace
            stripped = line.split('#')[0].strip().replace(',', ' ')
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                h   = float(parts[0])
                vp  = float(parts[1])
                vs  = float(parts[2])
                rho = float(parts[3]) if len(parts) > 3 else 2.7

                def _to_Q(val_str):
                    v = float(val_str)
                    if v == 0:
                        return 9999.0
                    return (1.0 / v) if v < 1.0 else v

                qp = _to_Q(parts[4]) if len(parts) > 4 else 9999.0
                qs = _to_Q(parts[5]) if len(parts) > 5 else 9999.0
                layers.append((h, vp, vs, rho, qp, qs))
            except ValueError:
                continue   # header or non-numeric line — skip silently

    if not layers:
        raise ValueError(
            f'No numeric velocity layers found in model file: {model_path}\n'
            f'Expected columns: H(km)  Vp(km/s)  Vs(km/s)  [rho  [Qp  [Qs]]]')

    nd_lines = []
    depth = 0.0
    for h, vp, vs, rho, qp, qs in layers:
        nd_lines.append(
            f'{depth:.4f}  {vp:.4f}  {vs:.4f}  {rho:.4f}  {qp:.2f}  {qs:.2f}')
        if h > 0:
            depth += h
            nd_lines.append(
                f'{depth:.4f}  {vp:.4f}  {vs:.4f}  {rho:.4f}  {qp:.2f}  {qs:.2f}')

    tmp = tempfile.NamedTemporaryFile(suffix='.nd', delete=False, mode='w')
    tmp.write('\n'.join(nd_lines) + '\n')
    tmp.close()
    _velmodel_nd_cache[model_path] = tmp.name
    return tmp.name


def _get_taup(model):
    """
    Return a cached TauPyModel.

    *model* may be:
    - a built-in name ('ak135', 'prem', 'iasp91', …)
    - a path to a plain-text velocity model file (converted to nd, then
      compiled to npz by build_taup_model before loading)
    """
    if model not in _taup_model_cache:
        from obspy.taup import TauPyModel
        if os.path.isfile(model):
            nd_path  = _velmodel_to_nd(model)
            from obspy.taup.taup_create import build_taup_model
            tmpdir   = tempfile.mkdtemp()
            build_taup_model(nd_path, output_folder=tmpdir)
            stem     = os.path.splitext(os.path.basename(nd_path))[0]
            npz_path = os.path.join(tmpdir, stem + '.npz')
            _taup_model_cache[model] = TauPyModel(model=npz_path)
        else:
            _taup_model_cache[model] = TauPyModel(model=model)
    return _taup_model_cache[model]


def _get_phase_arrival(arrivals, phases):
    """
    Return travel time (s) of the first phase in *phases* found in *arrivals*.

    *phases* is tried in order; raises ValueError if none are found.
    """
    phase_names = [a.phase.name for a in arrivals]
    for ph in phases:
        if ph in phase_names:
            return float(arrivals[phase_names.index(ph)].time)
    raise ValueError(f'None of {phases} found; available: {phase_names}')


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_stream(st, filter_dict):
    """Return a bandpass-filtered copy of *st*."""
    out = st.copy()
    out.detrend('demean')
    out.detrend('linear')
    out.taper(max_percentage=0.05, type='cosine')
    out.filter('bandpass',
               freqmin=filter_dict['fmin'],
               freqmax=filter_dict['fmax'],
               corners=filter_dict.get('corners', 4),
               zerophase=filter_dict.get('zerophase', False))
    out.taper(max_percentage=0.05, type='cosine')
    out.detrend('demean')
    out.detrend('linear')
    return out


# ---------------------------------------------------------------------------
# Window cutting
# ---------------------------------------------------------------------------

def cut_signal_window(st, orig_time, dist_km, window_length,
                      taup_model=None, window_alignment=0.3, depth_km=None):
    """
    Cut the surface-wave signal window using the S-wave arrival from TauPy.

    The S-wave arrival (phases tried in order: s, S, Sn, Sg) is used as the
    window reference.  *taup_model* may be a built-in name ('ak135', 'prem',
    …) or a path to a plain-text velocity model file.

    Window layout
    -------------
    t_start = S_arrival - window_alignment * window_length
    t_end   = t_start + window_length
    """
    if taup_model is None or depth_km is None:
        raise ValueError(
            'cut_signal_window requires taup_model and depth_km.')

    from obspy.geodetics.base import kilometers2degrees
    dist_deg = kilometers2degrees(dist_km)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        arrivals = _get_taup(taup_model).get_travel_times(
            source_depth_in_km=depth_km,
            distance_in_degree=dist_deg,
            phase_list=['p', 's', 'P', 'S'])
    arr_S = _get_phase_arrival(arrivals, ['s', 'S'])

    t_start = orig_time + arr_S - window_alignment * window_length
    t_end   = t_start + window_length
    return st.slice(t_start, t_end)


def cut_noise_window(st, orig_time, dist_km, depth_km,
                     noise_duration=200, taup_model='ak135'):
    """
    Cut pre-P-arrival noise window of *noise_duration* seconds.

    TauPy (phases tried in order: p, P, Pn, Pg) is used to estimate the
    P arrival.  *taup_model* may be a built-in name or a CPS .mod file path.
    Falls back to dist_km / 8 if TauPy fails.
    """
    try:
        from obspy.geodetics.base import kilometers2degrees
        dist_deg = kilometers2degrees(dist_km)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            arrivals = _get_taup(taup_model).get_travel_times(
                source_depth_in_km=depth_km,
                distance_in_degree=dist_deg,
                phase_list=['p', 's', 'P', 'S'])
        arr_P = _get_phase_arrival(arrivals, ['p', 'P'])
    except Exception:
        arr_P = dist_km / 8.0

    t_end   = orig_time + arr_P - 10.0
    t_start = t_end - noise_duration
    return st.slice(t_start, t_end)


# ---------------------------------------------------------------------------
# Signal-to-noise Ratio (SNR)
# ---------------------------------------------------------------------------

def compute_snr(signal_st, noise_st):
    """
    Return mean SNR (dB) over available Z/R/T components.

    SNR = 20 * log10( std(signal) / std(noise) )
    """
    snrs = []
    for comp in ('Z', 'R', 'T'):
        sig = signal_st.select(component=comp)
        noi = noise_st.select(component=comp)
        if not sig or not noi:
            continue
        As = np.std(sig[0].data, ddof=0)
        An = np.std(noi[0].data, ddof=0)
        if As > 0 and An > 0:
            snrs.append(20.0 * np.log10(As / An))
    return float(np.mean(snrs)) if snrs else 0.0


# ---------------------------------------------------------------------------
# Cross-correlation (CC)
# ---------------------------------------------------------------------------

def compute_cc(signal_streams, azi_deg, n_clusters=8):
    """
    Cross-correlation score for each station via K-means azimuth clustering.

    Within each azimuth cluster:
      1. Align all traces to the first station using the cross-correlation lag.
      2. Amplitude-normalise each trace.
      3. Build a linear stack (reference waveform).
      4. CC[s] = max |correlate(stack, original_trace)| for the each component.
      5. Normalise CC within the cluster to [0, 1].
      6. Average CC over components to get a single score per station.

    Returns array of shape (ns,).

    Parameters
    ----------
    signal_streams : list of obspy.Stream, one per station (ZRT components)
    azi_deg        : array-like of azimuths in degrees, length ns
    n_clusters     : number of K-means azimuth clusters (default 8)
    """
    ns = len(signal_streams)
    if ns == 0:
        return np.array([])
    n_clusters = min(n_clusters, ns)

    # determine common time-series length from the Z component
    nt = None
    for st in signal_streams:
        tr = st.select(component='Z')
        if tr:
            nt = len(tr[0].data)
            break
    if nt is None:
        return np.zeros(ns)

    comps = ('Z', 'R', 'T')
    nc = len(comps)
    data = np.zeros((ns, nc, nt))
    for s, st in enumerate(signal_streams):
        for ic, comp in enumerate(comps):
            tr = st.select(component=comp)
            if tr:
                d = tr[0].data
                n = min(len(d), nt)
                data[s, ic, :n] = d[:n]

    data_copy = data.copy()
    cc = np.zeros((ns, nc))

    azi_arr  = np.array(azi_deg).reshape(ns, 1)
    kmeans   = KMeans(n_clusters=n_clusters, random_state=0,
                      n_init='auto').fit(azi_arr)

    for ig in range(n_clusters):
        indices = np.where(kmeans.labels_ == ig)[0]
        if len(indices) == 0:
            continue

        # align all traces in this cluster to the first member
        ib = indices[0]
        for ik in indices:
            for ic in range(nc):
                corr  = np.correlate(data_copy[ib, ic],
                                     data_copy[ik, ic], mode='full')
                shift = np.argmax(corr) - nt
                data_copy[ik, ic] = np.roll(data_copy[ik, ic], shift)
                amp = np.max(np.abs(data_copy[ik, ic]))
                if amp > 0:
                    data_copy[ik, ic] /= amp

        # linear stack, then measure peak correlation with original trace
        for ic in range(nc):
            stack = np.mean(data_copy[np.ix_(indices, [ic])].squeeze(axis=1),
                            axis=0)
            for ik in indices:
                corr        = np.correlate(stack, data[ik, ic], mode='full')
                cc[ik, ic]  = np.max(np.abs(corr))

        # normalise within group
        grp_max = np.max(cc[indices])
        if grp_max > 0:
            cc[indices] /= grp_max

    # mean over components → scalar per station
    return np.mean(cc, axis=1)


# ---------------------------------------------------------------------------
# Distance score
# ---------------------------------------------------------------------------

def distance_score(dist_km, evmag=5.0):
    """
    Magnitude-based epicentral distance score (Scognamiglio et al. 2009).

    The thresholds are tuned for the given magnitude range and return a
    value in {0.25, 0.50, 0.75, 1.00}.
    """
    if evmag >= 5.5:
        if dist_km <= 50:   return 0.25
        if dist_km <= 130:  return 0.50
        if dist_km <= 220:  return 0.75
        return 1.0
    elif evmag >= 4.3 and evmag < 5.5:
        if dist_km <= 50:   return 0.25
        if dist_km <= 130:  return 0.75
        if dist_km <= 220:  return 1.00
        return 0.50
    elif evmag >= 3.6 and evmag < 4.3:
        # smaller events: favour closer stations
        if dist_km <= 50:   return 0.50
        if dist_km <= 130:  return 1.00
        if dist_km <= 220:  return 0.75
        return 0.25
    else:
        raise ValueError(f'Unsupported magnitude {evmag:.1f} for distance score')


# ---------------------------------------------------------------------------
# Azimuthal coverage (Ekström 2006, Eq. 10)
# ---------------------------------------------------------------------------

def azimuth_coverage(selected_azi_deg, candidate_azi_deg):
    """
    Effective number of stations C for the selected set plus one candidate.

    C = 1 / Σ_i ( gap_i / 2π )²

    where the gaps are the sorted azimuth differences (in radians) between
    consecutive azimuths, including the wrap-around gap.

    C = 1   → all stations in one direction (worst)
    C = N   → N stations uniformly distributed (best)

    Higher C is better.  Equivalent to Ekström (2006) Eq.10 written as
    the reciprocal of the raw coverage measure.
    """
    azimuths = np.deg2rad(
        np.array(list(selected_azi_deg) + [candidate_azi_deg]))
    azimuths = np.sort(azimuths % (2 * np.pi))
    n        = len(azimuths)
    raw = sum((azimuths[i + 1] - azimuths[i]) ** 2 for i in range(n - 1))
    raw += (2 * np.pi + azimuths[0] - azimuths[-1]) ** 2
    raw /= (2 * np.pi) ** 2
    return 1.0 / raw
