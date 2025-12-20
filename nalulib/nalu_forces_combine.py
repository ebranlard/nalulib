#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd


def find_pairs(pattern="*forces*.csv", verbose=False):
    import glob
    import os
    files = set(glob.glob(pattern))
    if len(files)==0:
        raise FileNotFoundError('No file matching pattern: ', pattern)
    pairs = []
    for f_pp in files:
        if "_pp_" not in f_pp:
            continue
        f_base = f_pp.replace("_pp_", "_", 1)
        if f_base in files:
            pairs.append((f_base, f_pp))
            if verbose:
                print('Found pair:', f_pp)
        else:
            print('[WARN] Missing paired csv: ', f_base)
    return pairs


def combine_files(f1, f2, fout):
    df1 = pd.read_csv(f1, delim_whitespace=True)
    df2 = pd.read_csv(f2, delim_whitespace=True)

    if not df1.columns.equals(df2.columns):
        raise ValueError(f"Column mismatch between {f1} and {f2}")
    df_sum = df1.copy()
    df_sum.iloc[:, 1:] += df2.iloc[:, 1:]  # keep Time, sum rest
    df_sum.to_csv(fout, sep=" ", index=False, float_format="%.8g")

def nalu_forces_combine(pattern="*forces*.csv", dry_run=False, verbose=False):
    pairs = find_pairs(pattern, verbose=verbose)

    if not pairs:
        print("No matching force file pairs found.")
        return

    fouts=[]
    for f_base, f_pp in pairs:
        fout = os.path.join(os.path.dirname(f_base), '_'+os.path.basename(f_base))
        if dry_run:
            if verbose:
#             print(f"{f_base} + {f_pp} -> {fout}")
                print(f"Combined(dry): {fout}")
        else:
            if verbose:
                print(f"Combined: {fout}")
            combine_files(f_base, f_pp, fout)
            fouts.append(fout)
    return fouts

def nalu_forces_combine_CLI():
    parser = argparse.ArgumentParser(description="Combine Nalu forces CSV files (forces + forces_pp)")
    parser.add_argument( "--dry-run", action="store_true", help="Show detected pairs only")
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    nalu_forces_combine(pattern, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
