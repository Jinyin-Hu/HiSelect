"""
Diagnostic plots produced by HiSelect.

plot_map    — regional map: all candidates (grey) and selected stations (blue).
plot_scores — bar charts of SNR, distance, CC, and combined scores.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Station map
# ---------------------------------------------------------------------------

def plot_map(meta, selected_idx, evla, evlo, evmag, path_out):
    try:
        import cartopy.crs as ccrs          # noqa: F401
        _plot_map_cartopy(meta, selected_idx, evla, evlo, evmag, path_out)
    except ImportError:
        _plot_map_mpl(meta, selected_idx, evla, evlo, evmag, path_out)


def _plot_map_cartopy(meta, selected_idx, evla, evlo, evmag, path_out):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    sel_set = set(selected_idx)
    lats = [m['stla'] for m in meta]
    lons = [m['stlo'] for m in meta]
    pad  = 2.0
    extent = [min(lons + [evlo]) - pad, max(lons + [evlo]) + pad,
              min(lats + [evla]) - pad, max(lats + [evla]) + pad]

    fig  = plt.figure(figsize=(10, 8))
    proj = ccrs.PlateCarree()
    ax   = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND,      facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN,     facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle='--')
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      linestyle='--', color='gray')
    gl.top_labels = gl.right_labels = False

    for i, m in enumerate(meta):
        if i in sel_set:
            continue
        ax.scatter(m['stlo'], m['stla'], marker='v', s=70, color='lightgray',
                   edgecolors='#E69F00', linewidths=0.4, zorder=5, transform=proj)

    for i, m in enumerate(meta):
        if i not in sel_set:
            continue
        ax.scatter(m['stlo'], m['stla'], marker='v', s=70, color='steelblue',
                   edgecolors='#E69F00', linewidths=0.4, zorder=7, transform=proj)
        ax.text(m['stlo'] + 0.05, m['stla'] + 0.05,
                f'{m["net"]}.{m["sta"]}', fontsize=4,
                transform=proj, zorder=8)

    ax.scatter([evlo], [evla], marker='*', s=300, color='red',
               edgecolors='k', linewidths=0.5, zorder=7, transform=proj)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='v', color='w', markerfacecolor='steelblue',
               markeredgecolor='#E69F00', markersize=8, label='Selected'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='lightgray',
               markeredgecolor='#E69F00', markersize=8, label='Unselected'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markeredgecolor='k', markersize=10, label=f'Event M{evmag}'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)
    ax.set_title('HiSelect: Station Map')
    out = os.path.join(path_out, 'hiselect_map.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    # print(f'Map saved: {out}')


def _plot_map_mpl(meta, selected_idx, evla, evlo, evmag, path_out):
    """Fallback map without cartopy."""
    sel_set = set(selected_idx)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, m in enumerate(meta):
        if i in sel_set:
            continue
        ax.scatter(m['stlo'], m['stla'], marker='v', s=70, color='lightgray',
                   edgecolors='#E69F00', linewidths=0.4, zorder=5)

    for i, m in enumerate(meta):
        if i not in sel_set:
            continue
        ax.scatter(m['stlo'], m['stla'], marker='v', s=70, color='steelblue',
                   edgecolors='#E69F00', linewidths=0.4, zorder=7)
        ax.text(m['stlo'] + 0.05, m['stla'] + 0.05,
                f'{m["net"]}.{m["sta"]}', fontsize=4, zorder=8)
    ax.scatter([evlo], [evla], marker='*', s=300, color='red',
               edgecolors='k', linewidths=0.5, zorder=7)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('HiSelect: Station Map')

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='v', color='w', markerfacecolor='steelblue',
               markeredgecolor='#E69F00', markersize=8, label='Selected'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='lightgray',
               markeredgecolor='#E69F00', markersize=8, label='Unselected'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markeredgecolor='k', markersize=10, label=f'Event M{evmag}'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.3)
    out = os.path.join(path_out, 'hiselect_map.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    # print(f'Map saved: {out}')


# ---------------------------------------------------------------------------
# Waveform record section
# ---------------------------------------------------------------------------

def plot_waveforms(meta, selected_idx, signal_streams, orig_time, path_out):
    """
    Record-section plot of the selected stations' filtered signal windows.

    Three panels (Z, R, T), one row per selected station sorted by distance.
    Each trace is normalised by its peak absolute amplitude and offset
    vertically so rows do not overlap.  The x-axis is time relative to the
    event origin.
    """
    # Sort by decreasing distance so row=0 sits at y=0 (bottom) and the
    # nearest station (last row) sits at the largest y value (top).
    # matplotlib renders larger y values higher, so nearest ends up at the top
    # with correct waveform polarity — no axis inversion needed.
    entries = sorted(
        [(meta[i]['dist'], i) for i in selected_idx], key=lambda x: x[0],
        reverse=True)
    n_sta = len(entries)
    comps = ('Z', 'R', 'T')

    fig, axes = plt.subplots(1, 3, figsize=(15, max(6, n_sta * 0.8)),
                             sharey=True)
    fig.subplots_adjust(wspace=0.05)

    yticks, ylabels = [], []
    for row, (dist_km, idx) in enumerate(entries):
        m  = meta[idx]
        st = signal_streams[idx]
        yticks.append(row)
        ylabels.append(f'{m["net"]}.{m["sta"]}  ({dist_km:.0f} km)')

        # peak across all three components for this station
        station_peak = max(
            (np.max(np.abs(st.select(component=c)[0].data))
             if st.select(component=c) else 0.0)
            for c in comps)

        for ax, comp in zip(axes, comps):
            tr_sel = st.select(component=comp)
            if not tr_sel:
                continue
            tr   = tr_sel[0]
            t    = (float(tr.stats.starttime) - float(orig_time) +
                    np.arange(tr.stats.npts) * tr.stats.delta)
            data = tr.data.copy()
            peak = np.max(np.abs(data))
            if station_peak > 0:
                data /= station_peak
            ax.plot(t, data + row, color='k', linewidth=0.5)
            # amplitude label just above the trace band
            ax.text(t[0], row + 0.45, f'{peak:.2e}',
                    fontsize=5, va='bottom', ha='left', color='dimgray')

    for ax, comp in zip(axes, comps):
        ax.set_xlabel('Time relative to origin (s)', fontsize=9)
        ax.set_title(comp, fontsize=10)
        ax.axhline(y=-0.5, color='none')          # padding at bottom
        ax.axhline(y=n_sta - 0.5, color='none')   # padding at top
        ax.grid(True, axis='x', linestyle='--', linewidth=0.3, alpha=0.5)

    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(ylabels, fontsize=7)

    fig.suptitle('HiSelect: selected station waveforms (signal window)',
                 fontsize=12, y=0.98)
    out = os.path.join(path_out, 'hiselect_waveforms.png')
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    # print(f'Waveform plot saved: {out}')


# ---------------------------------------------------------------------------
# Score bar charts
# ---------------------------------------------------------------------------

def plot_scores(meta, snr, dist_sc, cc, combined, selected_idx, path_out):
    """
    Four horizontal panels showing per-station SNR, distance, CC, and
    combined scores, sorted by descending combined score.
    Selected stations are shown in blue; others in light grey.
    """
    sel_set = set(selected_idx)
    ns      = len(meta)
    labels  = [f'{m["net"]}.{m["sta"]}' for m in meta]
    order   = np.argsort(combined)[::-1]   # descending combined score

    # effective number of stations C for the selected set
    sel_azi = np.sort(np.deg2rad(
        np.array([meta[i]['azi'] for i in selected_idx])) % (2 * np.pi))
    n_sel = len(sel_azi)
    raw = sum((sel_azi[i + 1] - sel_azi[i]) ** 2 for i in range(n_sel - 1))
    raw += (2 * np.pi + sel_azi[0] - sel_azi[-1]) ** 2
    raw /= (2 * np.pi) ** 2
    eff_C = 1.0 / raw if raw > 0 else 0.0

    panels  = [
        (snr,      'SNR (normalised)'),
        (dist_sc,  'Distance score D'),
        (cc,       'CC score'),
        (combined, 'Combined score'),
    ]
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(max(12, ns * 0.35), 10),
                             sharex=True)
    x = np.arange(ns)
    for ax, (values, title) in zip(axes, panels):
        colors = ['steelblue' if order[i] in sel_set else 'lightgray'
                  for i in range(ns)]
        ax.bar(x, values[order], color=colors, edgecolor='k', linewidth=0.3)
        ax.set_ylabel(title, fontsize=10)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_xlim(-0.5, ns - 0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([labels[i] for i in order],
                              rotation=90, fontsize=6)
    axes[0].set_title(
        f'HiSelect: per-station scores  (blue = selected, {n_sel} stations)'
        f'  |  Effective N = {eff_C:.2f}',
        fontsize=12)
    fig.tight_layout()
    out = os.path.join(path_out, 'hiselect_scores.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    # print(f'Score plot saved: {out}')
