import os
import pandas as pd
import numpy as np
import glob
import re
import argparse

import nalulib.pyplot as plt
from nalulib.weio.csv_file import CSVFile
from nalulib.nalu_input import NALUInputFile



def extract_aoa(filename):
    filename= filename.lower().replace('=', '').replace('_','')
    match = re.search(r'aoa([-\d\.]+)', filename)
    if match:
        aoa_str = match.group(1).rstrip('.')
        try:
            return float(aoa_str)
        except ValueError:
            return None
    return None

def reference_force(chord=1, rho=1.2, nu=9e-6, U0=None, dz=1, yaml_file='input.yaml', auto=True, dimensionless=True, verbose=False):
    # --- Infer velocity and density from input file
    if os.path.exists(yaml_file) and auto:
        print('[INFO] Reading velocity and density from input file:', yaml_file)
        yml = NALUInputFile(yaml_file)
        if verbose:
            print(yml)
        if U0 is None:
            vel = yml.velocity
            U0 = np.linalg.norm(vel)
        if rho is None:
            rho = yml.density
        if nu is None:
            nu = yml.viscosity

    Fin = 1
    if dimensionless:
        if (U0 is None or rho is None or nu is None):
            raise ValueError('For dimensionless forces, U0, rho and nu must be provided.')
        # Dimensionless forces
        Fin = 0.5 * rho * chord *dz * U0**2
    return Fin, U0, rho, nu


def read_forces(input_file='forces.csv', tmin=None, tmax=None, verbose=False, Fin=None):
    df0 = CSVFile(input_file).toDataFrame()
    df = df0.copy()

    if tmin is not None:
        df = df[df['Time']>tmin]
    if tmax is not None:
        df = df[df['Time']<tmax]

    if verbose:
        print('nSteps:',len(df), len(df0))

    df['Fx'] = ( df['Fpx'].values + df['Fvx'].values)
    df['Fy'] = ( df['Fpy'].values + df['Fvy'].values)
    df['Cx'] = df['Fx'] / Fin
    df['Cy'] = df['Fy'] / Fin   

    return df


def plot_forces(input_files='forces.csv', tmin=None, tmax=None, 
                chord=1, rho=None, nu=None, U0=None, dz=1, 
                yaml_file='input.yaml', 
                polar_ref=None,
                polar_out='polar.csv',
                auto=True,
                dimensionless=True,
                var='xy',
                verbose=False,
                ):

    # --- Are we dealing with multiple files or one file?
    if isinstance(input_files, str):
            input_files = [input_files]
    patterns = input_files 
    input_files = []
    for pattern in patterns:
        if "*" in pattern:
            input_files_loc = glob.glob(pattern)
            if len(input_files_loc) == 0:
                raise FileNotFoundError(f"No files matching pattern: {pattern}")
            input_files.extend(input_files_loc)
        else:
            input_files.extend([pattern])
    # unique and sorted
    input_files = list(set(input_files))
    input_files = sorted(input_files)
    print('[INFO] Input files provided ({}):'.format(len(input_files)), input_files[0], '...', input_files[-1])

    # --- Get reference force
    Fin, U0, rho, nu = reference_force(chord=chord, rho=rho, nu=nu, U0=U0, dz=dz, yaml_file=yaml_file, auto=auto, dimensionless=dimensionless, verbose=verbose)
    FC, label= 'F', 'Forces [N]'
    if dimensionless:
        FC, label='C' , 'Coefficient [-]'

    if verbose:
        print('chord: {:9.3f}  '.format(chord))
        print('dz   : {:9.3f}  '.format(dz))
        print('U0   : {:9.3f}  '.format(U0))
        print('rho  : {:9.3f}  '.format(rho))
        print('nu   : {:9.3e}  '.format(nu))
        print('Re   : {:9.3f} Million'.format(U0 * chord /(nu*1e6)))
        print('Fref : {:9.3f} [N]'.format(Fin))

    if len(input_files) == 1:
        for input_file in input_files:
            df = read_forces(input_file, tmin=tmin, tmax=tmax, verbose=verbose, Fin=Fin)
            if verbose:
                print("Filename:", input_file)
                print('Final values: {}x[-1]={:.3f}, {}y[-1]={:.3f}'.format(FC, df['Cx'].values[-1], FC, df['Cy'].values[-1]))

            # --- Plot time series
            fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,5.8))
            fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
            if 'x' in var:
                plt.plot(df['Time'].values, df['Cx'].values, label=FC+'x')
            if 'y' in var:
                plt.plot(df['Time'].values, df['Cy'].values, label=FC+'y')
            ax.set_ylabel(label)
            ax.set_xlabel('Time [s]')
            ax.legend()

    if polar_ref is None and len(input_files) == 1:
        # If no polar reference is provided, we only plot the time series
        return

    # ---
    results = []
    for input_file in input_files:
        if verbose:
            print("Filename:", input_file)
        try:
            df = read_forces(input_file, tmin=tmin, tmax=tmax, verbose=verbose, Fin=Fin)
        except Exception as e:
            print(f"[FAIL] Error reading {input_file}: {e}")
            continue
        aoa = extract_aoa(input_file)
        results.append({'alpha': aoa, 'Cl': df['Cy'].values[-1], 'Cd': df['Cx'].values[-1]})

    df = pd.DataFrame(results)
    # Sort by AoA
    df = df.sort_values(by='alpha')
    # Create a DataFrame for results
    df.to_csv(polar_out, index=False)

    # --- Plot polar
    fig,axes = plt.subplots(1, 2, sharey=False, figsize=(12.8,5.8))
    if polar_ref is not None and dimensionless:
        df_exp = CSVFile(polar_ref).toDataFrame()
        def sanitize(c):
            c = c.lower().replace('c_', 'c').replace('_', ' ').split()[0]
            if c=='aoa':
                c='alpha'
            return c
        new_cols = [sanitize(c) for c in df_exp.columns.values]
        df_exp.columns = new_cols
        axes[0].plot(df_exp['alpha'], df_exp['cl'], 'ko', label='Cl-exp')
        axes[0].plot(df_exp['alpha'], df_exp['cd'], 'ko', label='Cd-exp')
        axes[1].plot(df_exp['cd']   , df_exp['cl'], 'ko', label='exp')
    sty = 'ro' if len(df) == 1 else '-'
    axes[0].plot(df['alpha'], df['Cl'], sty, label='Cl')
    axes[0].plot(df['alpha'], df['Cd'], sty, label='Cd')
    axes[1].plot(df['Cd']   , df['Cl'], sty, label='Cl')
    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('Coefficient')
    axes[1].set_ylabel('Coefficient')
    axes[0].legend()
    axes[0].set_title('Cl and Cd vs Angle of Attack')
    axes[0].grid(True)
    axes[1].grid(True)


def nalu_forces_CLI():
    parser = argparse.ArgumentParser(description="Plot Nalu forces from a CSV file, optionally dimensionless.")
    parser.add_argument('input', type=str, nargs='+', default='forces.csv', help='Force file(s) from nalu-wind, or glob ("forces_aoa*.csv")')
    parser.add_argument('--tmin', type=float, default=None, help='Minimum time')
    parser.add_argument('--tmax', type=float, default=None, help='Maximum time')
    parser.add_argument('--chord', type=float, default=1.0, help='Airfoil chord')
    parser.add_argument('--rho', type=float, default=None, help='Density (overrides YAML)')
    parser.add_argument('--nu', type=float, default=None, help='Kinematic viscosity (overrides YAML)')
    parser.add_argument('--U0', type=float, default=None, help='Reference velocity (overrides YAML)')
    parser.add_argument('--dz', type=float, default=1.0, help='Spanwise thickness')
    parser.add_argument('--yaml', type=str, default='input.yaml', help='NALU input YAML file for auto properties')
    parser.add_argument('--polar-ref', type=str, default=None, help='CSV file with reference polar data, alpha, Cl, Cd, Cm')
    parser.add_argument('--polar-out', type=str, default=None, help='CSV file with output polar if multiple files provied.')
    parser.add_argument('--no-auto', dest='auto', action='store_false', help='Do not auto-read velocity, density, etc from YAML')
    parser.add_argument('--dimensionless', action='store_false', help='Plot dimensional forces')
    parser.add_argument('--var', type=str, default='xy', help='Which force components to plot (x, y, or xy)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    plot_forces(
        input_files=args.input,
        tmin=args.tmin,
        tmax=args.tmax,
        chord=args.chord,
        rho=args.rho,
        nu=args.nu,
        U0=args.U0,
        dz=args.dz,
        yaml_file=args.yaml,
        polar_ref=args.polar_ref,
        auto=args.auto,
        dimensionless=args.dimensionless,
        var=args.var,
        verbose=args.verbose,
    )
    plt.show()


if __name__ == '__main__':
    #plot_forces('forces.csv', verbose=True)
    nalu_forces_CLI()
    plt.show()
