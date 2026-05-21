"""
Output writers for HiSelect results.

Writes:
  station_file.txt  — one line per selected station, sorted by distance
  weights.dat       — CAP weights file (via pysep; fallback writer if pysep absent)
  weights_body.dat  — same format, body-wave weights
  weights_surf.dat  — same format, surface-wave weights
"""

import glob
import os
import shutil

from obspy import Stream


def write_station_file(selected_meta, path_out):
    """
    Write station_file.txt with columns:
        STA   NET   LAT   LON   DIST_KM   BAZ_DEG
    Rows are sorted by epicentral distance.
    """
    rows = sorted(selected_meta, key=lambda m: m['dist'])
    out  = os.path.join(path_out, 'station_file.txt')
    with open(out, 'w') as fh:
        for m in rows:
            fh.write(
                f'{m["sta"]:<6} {m["net"]:<4} '
                f'{m["stla"]:9.4f} {m["stlo"]:10.4f} '
                f'{m["dist"]:9.3f} {m["baz"]:7.2f}\n')
    # print(f'Written {out}  ({len(rows)} stations)')


def write_weights(selected_meta, path_out):
    """
    Write weights*.dat files.  Delegates to pysep when available;
    falls back to a minimal writer otherwise.
    """
    try:
        from pysep.utils.cap_sac import write_cap_weights_files
        st = Stream()
        for m in selected_meta:
            st += m['tr_z']
        write_cap_weights_files(st, path_out=path_out, order_by='dist')
        # print(f'Written weights*.dat  ({len(selected_meta)} stations, via pysep)')
    except Exception as exc:
        print(f'WARNING: pysep unavailable ({exc}); using fallback writer.')
        _write_weights_fallback(selected_meta, path_out)


def _write_weights_fallback(selected_meta, path_out):
    """Minimal fallback that writes the three weights files without pysep."""
    rows     = sorted(selected_meta, key=lambda m: m['dist'])
    event_id = os.path.basename(os.path.normpath(path_out))
    for fname in ('weights.dat', 'weights_body.dat', 'weights_surf.dat'):
        out = os.path.join(path_out, fname)
        with open(out, 'w') as fh:
            for m in rows:
                # token format matches pysep convention: eventid00.NET.STA.LOC.BAND
                token = f'{event_id}00.{m["net"]}.{m["sta"]}.{m["loc"]}.{m["band"]}'
                fh.write(
                    f'    {token:<44} {m["dist"]:8.2f}'
                    f'   1 1    1 1 1     0.00 0     0.00 0    0\n')
    # print(f'Written weights*.dat (fallback, {len(rows)} stations)')


def copy_selected_data(selected_meta, event_dir, path_out):
    """
    Copy the selected stations' SAC files and output files into
    ``{path_out}/selected/``.

    SAC files (Z, R, T) are sourced from *event_dir*.
    Output files (station_file.txt, weights*.dat) are sourced from *path_out*;
    a warning is printed for any that do not yet exist (call write_outputs()
    first to avoid this).
    """
    dest = os.path.join(path_out, 'selected')
    os.makedirs(dest, exist_ok=True)

    # --- SAC files ---
    n_copied = 0
    for m in selected_meta:
        for comp in ('Z', 'R', 'T'):
            pattern = os.path.join(
                event_dir,
                f'*{m["net"]}.{m["sta"]}.{m["loc"]}.{m["band"]}{comp}.sac')
            matches = glob.glob(pattern)
            if not matches:
                print(f'  WARNING: SAC file not found for '
                      f'{m["net"]}.{m["sta"]} {m["band"]}{comp}')
                continue
            shutil.copy2(matches[0], dest)
            n_copied += 1
    # print(f'Copied {n_copied} SAC files → {dest}')

    # --- output files ---
    output_files = (
        'station_file.txt',
        'weights.dat',
        'weights_body.dat',
        'weights_surf.dat',
    )
    for fname in output_files:
        src = os.path.join(path_out, fname)
        if os.path.exists(src):
            shutil.copy2(src, dest)
        else:
            print(f'  WARNING: {fname} not found in {path_out} '
                  f'— run write_outputs() first')
    # print(f'Copied output files → {dest}')
