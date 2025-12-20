import os
import pandas as pd
import numpy as np
import glob
import re
import argparse

import numpy as np

# Local
import nalulib.pyplot as plt
from nalulib.weio.csv_file import CSVFile
from nalulib.nalu_input import NALUInputFile
from nalulib.tools.steady_state import *


def standardize_polar_df(df):
    """ Make sure columns have the proper name """
    def sanitize(c):
        c = c.lower().replace('c_', 'c').replace('_', ' ').strip()
        c = c.replace('[deg]','').replace('[-]','').replace('(-)','').replace('(deg)','').strip()
        #.split()[0]
        if c.startswith('aoa'):
            c='Alpha'
        c = c.capitalize()
        return c
    new_cols = [sanitize(c) for c in df.columns.values]
    df.columns = new_cols
    #print('Cols', df.columns.values)
    return df

def plot_polar_df(axes, df, sty, label=None, c=None):
    axes[0].plot(df['Alpha'], df['Cl'], sty, c=c, label=(label if label is not None else ''))
    axes[0].plot(df['Alpha'], df['Cd'], sty, c=c, label=None)
#     if sty=='ko' or sty=='kd':
#     else:
#         axes[0].plot(df['Alpha'], df['Cl'], sty, label='Cl '+ (label if label is not None else ''))
#         axes[0].plot(df['Alpha'], df['Cd'], sty, label='Cd '+ (label if label is not None else ''))
    axes[1].plot(df['Cd']   , df['Cl'], sty, c=c, label=label)



def plot_polars(polars, verbose=False, 
                cfd_sty='-', cfd_c='k'
                ):
    fig,axes = plt.subplots(1, 2, sharey=False, figsize=(12.8,5.8))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.11, hspace=0.20, wspace=0.20)

    if polars is not None:
        cfd_done=False
        exp_done=False

        COLRS=plt.rcParams['axes.prop_cycle'].by_key()['color']
        kcol=-1
        for k,pol in polars.items():
            sty='-'
            if pol is None:
                continue
            if isinstance(pol, str):
                if verbose:
                    print(f'[INFO] Polar {k:15s}: ', pol)
                df2 = CSVFile(pol).toDataFrame()
                df2 = standardize_polar_df(df2)
            elif isinstance(pol, pd.DataFrame):
#                 if verbose:
#                     print(f'[INFO] Polar {k:15s}: ', pol.keys(), len(pol))
                if len(pol)==0:
                    continue
                df2 = standardize_polar_df(pol)
            else:
                raise NotImplementedError(type(pol))
            # --- TODO ugly logic here
            if 'exp' in k.lower() and not exp_done:
                sty='o'; c='k'
                exp_done=True
            elif 'grit' in k.lower():
                sty='d'; c='k'
            elif k.lower()=='cfd':
                sty=cfd_sty; c=cfd_c;
            elif 'cfd3d' in k.lower():
                sty='+-'
                kcol+=1
                c=COLRS[kcol]
            elif 'cfd' in k.lower():
                sty='--'
                kcol+=1
                c=COLRS[kcol]
            else:
                kcol+=1
                c=COLRS[kcol]
            if len(pol)==1:
                sty='o'
            if verbose:
                print(f'[INFO] Plot Polar: label={k:15s}: ', len(pol), sty, c)
            plot_polar_df(axes, df2, sty=sty, label=k, c=c)


    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('Coefficient [-]')
    axes[1].set_xlabel('Cd [-]')
    axes[1].set_ylabel('Cl [-]')
    axes[0].legend()
    axes[1].legend()
    #axes[0].set_title('Cl and Cd vs Angle of Attack')
    axes[0].grid(True)
    axes[1].grid(True)
    return fig



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


def reference_force_yml(yml, Aref, dimensionless=True):
    if isinstance(yml, str):
        yml = NALUInputFile(yml)
    #if verbose:
    #    print(yml)
    vel = yml.velocity
    U0  = np.linalg.norm(vel)
    rho = yml.density
    nu  = yml.viscosity
    Fin = 1
    if dimensionless:
        if (U0 is None or rho is None or nu is None):
            raise ValueError('For dimensionless forces, U0, rho and nu must be provided.')
        # Dimensionless forces
        Fin = 0.5 * rho * Aref * U0**2
    return Fin, U0, rho, nu, yml

def reference_force(Aref=1, rho=1.2, nu=9e-6, U0=None, yaml_file='input.yaml', auto=True, dimensionless=True, verbose=False, yml=None):
    # --- Infer velocity and density from input file
    if yml is not None:
        Fin, U0, rho, nu, yml = reference_force_yml(yml, Aref, dimensionless)
    elif os.path.exists(yaml_file) and auto:
        print('[INFO] Reading velocity and density from input file:', yaml_file)
        yml = NALUInputFile(yaml_file)
        Fin, U0, rho, nu, yml = reference_force_yml(yml, Aref, dimensionless)
    else:
        yml = None
        Fin = 1
        U0 = U0
        rho = rho
        nu = nu
        if dimensionless:
            raise ValueError('For dimensionless forces, U0, rho and nu must be provided.')
    return Fin, U0, rho, nu, yml




def read_forces_csv(input_file='forces.csv', tmin=None, tmax=None, verbose=False, Fin=None):
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



def read_forces_yaml(yaml_file, Aref=None, dimensionless=True, verbose=False, tmin=None, tmax=None):
    """ 
    Read forces based on a yaml file, concatenate them if necessary
    """
    yml = None
    Fin = 1.0
    _, ext = os.path.splitext(yaml_file)
    if ext.lower() not in ['.yaml', '.yml', '.i']:
        raise Exception('Input should be yaml', yaml_file)
    yml = NALUInputFile(yaml_file)
    Fin, U0, rho, nu, _ = reference_force_yml(yml, Aref, dimensionless=dimensionless)
    csv_files = []
    basedir = os.path.dirname(yaml_file)
    for realm in yml.data['realms']: 
        if 'post_processing' in realm:
            for pp in realm['post_processing']:
                force_file = os.path.join(basedir, pp['output_file_name'])
                print(force_file)
                csv_files.append(force_file)

    for i, csv_file in enumerate(csv_files):
        df = read_forces_csv(input_file=csv_file, tmin=tmin, tmax=tmax, verbose=verbose, Fin=Fin)
        if i==0:
            df_prev = df
        else:
            # Add all except 'time'
            df.loc[:, df.columns != 'Time'] = df.drop(columns='Time') + df_prev.drop(columns='Time')
            df['Time'] = (df['Time'] + df_prev['Time']) / 2

    return df, Fin, yml, U0, rho, nu


def input_files_from_patterns(patterns):
    if not isinstance(patterns, list):
        patterns=[patterns]
    input_files = []
    if len(patterns)==0:
        raise Exception('Nalu-forces: Empty list of patterns provided')
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

    # Sort by aoa
    aoas = [extract_aoa(input_file) for input_file in input_files]
    I = np.argsort(aoas)
    input_files = np.asarray(input_files)[I]

    # --- Are we dealing with csv or yaml files
    _, ext = os.path.splitext(input_files[0])
    input_is_csv = ext.lower() not in ['.yaml', '.yml', '.i']

    return input_files, input_is_csv



def polar_postpro(input_pattern, yaml_file=None, tmin=None, chord=1, dz=1, verbose=False, polar_out=None, 
                  use_ss=False,
                  polars=None, plot=False, cfd_sty='-', cfd_c='k'):
    """ 
    INPUTS:
     - input_pattern: either:
          - a pattern to find nalu wind input files, in which case the force files are inferred by reading the yaml files
        OR- a pattern to find nalu wind force files, in which case yaml_file needs to be provided to find the ref force
    """
    dfp = None
    fig = None
    Aref = chord*dz

    # --- Input files
    input_files, input_is_csv = input_files_from_patterns(input_pattern)
    if verbose:
        print('[INFO] Input files provided ({}):'.format(len(input_files)), input_files[0], '...', input_files[-1])

    # Open Yaml file to get reference values
    if input_is_csv:
        if yaml_file is None:
            raise Exception('Provide yaml file if input are force files.')
        Fin, U0, rho, nu, yml = reference_force_yml(yaml_file, Aref, dimensionless=True)

    # --- Loop on files (csv or yaml)
    #d_sorted = dict(sorted(d.items(), key=lambda x: x[0]))
    dfs = []
    sss = []
    for input_file in input_files:
        aoa = extract_aoa(input_file)
        if verbose:
            print(f' {input_file} ')
        try:
            if input_is_csv:
                df = read_forces_csv(input_file, verbose=verbose, Fin=Fin)
            else:
                df, yml, Fin, U0, rho, nu = read_forces_yaml(input_file, Aref=Aref, dimensionless=True, verbose=verbose)
        except Exception as e:
            print(f"[FAIL] Error reading {input_file}: {e}")
            continue
        dfs.append(df)
        d = {'Alpha': aoa, 'Nt':len(df)}
        if use_ss:
            ss = analyze_steady_state(df['Time'].values, df['Fpy'].values)
            d.update(ss)
        sss.append(d)
    dfss = pd.DataFrame(sss)

    # --- 
    if use_ss:
        t_onset_default = (dfss['t_trans1'].mean()+ dfss['t_trans2'].mean())/2
        if np.isnan(t_onset_default):
            print('[WARN] No steady-state ever detected, setting t_onset to None')
            t_onset_default=None
            use_ss =None
    else:
        t_onset_default = None

    # --- Create Polar dataframe
    results = []
    for i, (df, ss) in enumerate(zip(dfs, sss)):
        period = np.nan
        if use_ss:
            if ss['type'] in ['steady_periodic' , 'constant']:
                t_onset = ss['t_trans2']
                df = df[df['Time']>t_onset]
            else:
                if df['Time'].values[-1]<t_onset_default:
                    print(f"[WARN] Simulation file for aoa {ss['Alpha']} has insufficient datapoints")
                    df = df.iloc[-1:]
                else:
                    df = df[df['Time']>t_onset_default]
            period = ss['period']
        else:
            n = int(0.3*len(df))
            df = df.iloc[-n:]
        results.append({'Alpha': ss['Alpha'], 
                        'Cl': df['Cy'].mean(), 'Cd': df['Cx'].mean(), 
                        'Cl_std': df['Cy'].std(), 'Cd_std': df['Cx'].std(), 
                        'Cl_min': df['Cy'].min(), 'Cd_min': df['Cx'].min(), 
                        'Cl_max': df['Cy'].max(), 'Cd_max': df['Cx'].max(), 
                        'N': len(df),
                        'T': period
                        })


    dfp = pd.DataFrame(results)
    #dfp = standardize_polar_df(dfp)
    if verbose:
        print(dfp)
        print(dfss)

    # Create a DataFrame for results
    if polar_out:
        print('[INFO] Output polar: ', polar_out)
        dfp.to_csv(polar_out, index=False)

    # --- Plot polar
    if plot:
        if polars is None:
            polars={}
        polars = {'cfd': dfp, **polars}
        fig = plot_polars(polars, cfd_sty=cfd_sty, cfd_c=cfd_c)
    return dfp, dfss, fig




def nalu_forces(input_files='forces.csv', tmin=None, tmax=None, 
                chord=1, rho=None, nu=None, U0=None, dz=1, 
                yaml_file='input.yaml', 
                polar_ref=None,
                polar_exp=None,
                polar_out='polar.csv',
                auto=True,
                dimensionless=True,
                var='xy',
                verbose=False,
                plot=True
                ):


    # --- Derived inputs
    FC, label= 'F', 'Forces [N]'
    if dimensionless:
        FC, label='C' , 'Coefficient [-]'
    Aref = chord*dz

    # --- Are we dealing with multiple files or one file?
    if isinstance(input_files, str):
        input_files = [input_files]
    patterns = input_files 
    input_files, input_is_csv = input_files_from_patterns(patterns)
    print('[INFO] Input files provided ({}):'.format(len(input_files)), input_files[0], '...', input_files[-1])

    # --- Are we dealing with csv or yaml files
    if not input_is_csv:
        print('[INFO] Input files are yaml files instead of csv files, overridding yaml_file')
        yaml_file = input_files [0]

    dft = None
    dfp = None
    figt = None
    figp = None

    # --- One input_file
    if len(input_files) == 1:
        for input_file in input_files:
            if input_is_csv:
                Fin, U0, rho, nu, yml = reference_force(Aref=Aref, rho=rho, nu=nu, U0=U0, yaml_file=yaml_file, auto=auto, dimensionless=dimensionless, verbose=verbose)
                dft, yml = read_forces(input_file, tmin=tmin, tmax=tmax, verbose=verbose, Fin=Fin)
            else:
                dft, yml, Fin, U0, rho, nu = read_forces_yaml(input_file, Aref=Aref, dimensionless=dimensionless, verbose=verbose, tmin=tmin, tmax=tmax)

            if verbose:
                print('chord: {:9.3f}  '.format(chord))
                print('dz   : {:9.3f}  '.format(dz))
                print('U0   : {:9.3f}  '.format(U0))
                print('rho  : {:9.3f}  '.format(rho))
                print('nu   : {:9.3e}  '.format(nu))
                print('Re   : {:9.3f} Million'.format(U0 * chord /(nu*1e6)))
                print('Fref : {:9.3f} [N]'.format(Fin))

            if verbose:
                print("Filename:", input_file)
                print('Final values: {}x[-1]={:.3f}, {}y[-1]={:.3f}'.format(FC, dft['Cx'].values[-1], FC, dft['Cy'].values[-1]))

            # --- Plot time series
            if plot:
                figt, ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,5.8))
                figt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
                if 'x' in var:
                    plt.plot(dft['Time'].values, dft['Cx'].values, label=FC+'x')
                if 'y' in var:
                    plt.plot(dft['Time'].values, dft['Cy'].values, label=FC+'y')
                ax.set_ylabel(label)
                ax.set_xlabel('Time [s]')
                ax.legend()

    if polar_ref is None and len(input_files) == 1:
        # If no polar reference is provided, we only plot the time series
        return

    # ---
    dfp, dfss, figp = polar_postpro(yaml_file, patterns, tmin=tmin, chord=chord, dz=dz, verbose=verbose, polar_out=polar_out, polars={'ref':polar_ref, 'exp':polar_exp}, plot=plot)

    return dfp, figp, dft, figt


def nalu_forces_CLI():
    parser = argparse.ArgumentParser(description="Plot Nalu forces from a CSV file, optionally dimensionless.")
    parser.add_argument('input', type=str, nargs='*', default='forces*.csv', help='Force file(s) from nalu-wind, or glob ("forces_aoa*.csv")')
    parser.add_argument('--tmin', type=float, default=None, help='Minimum time')
    parser.add_argument('--tmax', type=float, default=None, help='Maximum time')
    parser.add_argument('--chord', type=float, default=1.0, help='Airfoil chord')
    parser.add_argument('--rho', type=float, default=None, help='Density (overrides YAML)')
    parser.add_argument('--nu', type=float, default=None, help='Kinematic viscosity (overrides YAML)')
    parser.add_argument('--U0', type=float, default=None, help='Reference velocity (overrides YAML)')
    parser.add_argument('--dz', type=float, default=1.0, help='Spanwise thickness')
    parser.add_argument('--yaml', type=str, default='input.yaml', help='NALU input YAML file for auto properties')
    parser.add_argument('--polar-exp', type=str, default=None, help='CSV file with experimental polar data, alpha, Cl, Cd, Cm')
    parser.add_argument('--polar-ref', type=str, default=None, help='CSV file with reference polar data, alpha, Cl, Cd, Cm')
    parser.add_argument('--polar-out', type=str, default=None, help='CSV file with output polar if multiple files provied.')
    parser.add_argument('--no-auto', dest='auto', action='store_false', help='Do not auto-read velocity, density, etc from YAML')
    parser.add_argument('--forces', dest='dimensionless', action='store_false', help='Plot forces instead of coefficients')
    parser.add_argument('--var', type=str, default='xy', help='Which force components to plot (x, y, or xy)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-plot', action='store_true', help='Do not plot results')
    args = parser.parse_args()

    nalu_forces(
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
        polar_exp=args.polar_exp,
        polar_out=args.polar_out,
        auto=args.auto,
        dimensionless=args.dimensionless,
        var=args.var,
        verbose=args.verbose,
        plot=not args.no_plot,
    )
    plt.show()


if __name__ == '__main__':
    #plot_forces('forces.csv', verbose=True)
    nalu_forces_CLI()
    plt.show()
