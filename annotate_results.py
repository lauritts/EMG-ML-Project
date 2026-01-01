#!/usr/bin/env python3
"""
Annotate individual results CSVs by prepending metadata extracted from
`results_ninapro_experiments/experiment_summary.csv` or `experiment_summary.csv` in that folder.

This script will look for CSV files in `results_ninapro_experiments/` and for each one,
try to find a matching row in the summary by filename or by a contained timestamp/identifier.
If found, it will prepend a metadata block to the CSV (lines beginning with '#') and
save the file back.
"""

import csv
import os
import re
from datetime import datetime

SUMMARY_PATH = './results_ninapro_experiments/experiment_summary.csv'
RESULTS_DIR = './results_ninapro_experiments'


def load_summary(path):
    if not os.path.exists(path):
        print(f"Summary not found: {path}")
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Try to parse timestamp into a datetime for fuzzy matching
            ts = r.get('timestamp', '')
            try:
                from datetime import datetime as _dt
                r['_dt'] = _dt.fromisoformat(ts) if ts else None
            except Exception:
                r['_dt'] = None
            rows.append(r)
    return rows


def make_metadata(row):
    # Build a compact metadata string from summary row
    items = []
    keys = ['features','window_size','overlap','train_ratio','pca','select_k','models','name']
    for k in keys:
        if k in row and row[k] and row[k] != 'None':
            items.append(f"{k}={row[k]}")
    items.append(f"timestamp={row.get('timestamp', '')}")
    return '# ' + ' | '.join(items) + '\n'


def match_summary_to_file(summary_rows, filename):
    # Try to find a matching row for a result CSV filename.
    # Matching strategies:
    # 1) If filename contains known run name (e.g., "results_ninapro_full_LDA_random_20251227_122431.csv"),
    #    extract the timestamp-like pattern YYYYMMDD_HHMMSS and match with summary timestamp
    # 2) Otherwise, perform substring match against 'name' or 'model' fields

    # 1) extract timestamp in filename like 20251227_144709 and match to nearest summary timestamp
    ts_match = re.search(r"(20\d{6}_\d{6})", filename)
    if ts_match:
        ts = ts_match.group(1)
        # convert to iso-like string: 20251227_144709 -> 2025-12-27T14:47:09
        try:
            dt = None
            y = int(ts[0:4]); m = int(ts[4:6]); d = int(ts[6:8])
            hh = int(ts[9:11]); mm = int(ts[11:13]); ss = int(ts[13:15])
            from datetime import datetime as _dt
            dt = _dt(y, m, d, hh, mm, ss)
        except Exception:
            dt = None

        if dt is not None:
            # find the summary row with the nearest timestamp (if available)
            best = None
            best_diff = None
            for r in summary_rows:
                if r.get('_dt') is None:
                    continue
                diff = abs((r['_dt'] - dt).total_seconds())
                if best is None or diff < best_diff:
                    best = r
                    best_diff = diff
            # accept match if within 2 hours (7200s)
            if best is not None and best_diff is not None and best_diff <= 7200:
                return best

    # 2) substring matches
    fname_noext = os.path.splitext(os.path.basename(filename))[0]
    for r in summary_rows:
        # match 'name' field
        if 'name' in r and r['name'] and r['name'] in fname_noext:
            return r
        # match model/run strings
        if 'models' in r and r['models'] and r['models'] in fname_noext:
            return r
        # match any known piece like 'full' 'basic' 'extended'
        if r.get('features') and r['features'] in fname_noext:
            return r

    return None


def prepend_metadata_to_csv(path, meta_lines):
    # Read original content
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Strip existing leading comment block (lines starting with '#')
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith('#'):
        i += 1
    rest = ''.join(lines[i:])

    new_content = ''.join(meta_lines) + '\n' + rest
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True


def main():
    summary_rows = load_summary(SUMMARY_PATH)
    if not summary_rows:
        print('No summary rows found; exiting')
        return

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
    updated = []
    not_matched = []

    for fname in files:
        fpath = os.path.join(RESULTS_DIR, fname)
        # skip the main summary file
        if fname == 'experiment_summary.csv' or fname.startswith('OPTIMAL_'):
            continue

        matched = match_summary_to_file(summary_rows, fname)
        if matched:
            meta = make_metadata(matched)
            # Also add a header line with the CSV filename and a human message
            meta_lines = [f"# file={fname}\n", meta, f"# added={datetime.now().isoformat()}\n"]
            changed = prepend_metadata_to_csv(fpath, meta_lines)
            if changed:
                updated.append(fname)
        else:
            not_matched.append(fname)

    print(f"Updated {len(updated)} files:")
    for u in updated[:50]:
        print('  ', u)
    if not_matched:
        print(f"{len(not_matched)} files not matched. Examples:")
        for n in not_matched[:10]:
            print('  ', n)

if __name__ == '__main__':
    main()
