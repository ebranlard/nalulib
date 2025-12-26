import os
import pandas as pd
import numpy as np
import glob
import re
import argparse

import numpy as np

# Local
import nalulib.pyplot as plt
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_output import read_forces_csv
from nalulib.tools.steady_state import analyze_steady_state
from nalulib.weio.csv_file import CSVFile



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

def plot_polar_df(axes, df, ls='-', label=None, c=None, marker=None):
    axes[0].plot(df['Alpha'], df['Cl'], ls=ls, marker=marker, c=c, label=(label if label is not None else ''))
    axes[0].plot(df['Alpha'], df['Cd'], ls=ls, marker=marker, c=c, label=None)
#     if sty=='ko' or sty=='kd':
#     else:
#         axes[0].plot(df['Alpha'], df['Cl'], sty, label='Cl '+ (label if label is not None else ''))
#         axes[0].plot(df['Alpha'], df['Cd'], sty, label='Cd '+ (label if label is not None else ''))
    axes[1].plot(df['Cd']   , df['Cl'], ls=ls, marker=marker, c=c, label=label)




def plot_polars(polars, verbose=False, 
                cfd_ls='-', cfd_c='k', cfd_m=None,
                ):
    fig,axes = plt.subplots(1, 2, sharey=False, figsize=(12.8,5.8))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.11, hspace=0.20, wspace=0.20)

    if polars is not None:
        cfd_done=False
        exp_done=False

        try:
            COLRS=plt.rcParams['axes.prop_cycle'].by_key()['color']
        except:
            COLRS=['b','r','g','k']
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
            ls='-'
            m=None
            if 'exp' in k.lower() and not exp_done:
                m='o'; c='k'; ls=''
                exp_done=True
            elif 'grit' in k.lower():
                m='d'; c='k'; ls=''
            elif k.lower()=='cfd':
                m=cfd_m; c=cfd_c; ls=cfd_ls;

            elif 'cfd3d' in k.lower():
                m='+'; ls='-'
                kcol+=1
                c=COLRS[kcol]
            elif 'cfd' in k.lower():
                ls='--'
                kcol+=1
                c=COLRS[kcol]
            else:
                kcol+=1
                c=COLRS[kcol]
            if len(pol)==1:
                m='o'; ls=''

            if verbose:
                print(f'[INFO] Plot Polar n=len(pol), label={k:15s}, ls={ls}, m={m} c={c}')
            plot_polar_df(axes, df2, marker=m, label=k, c=c, ls=ls)


    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('Coefficient [-]')
    axes[1].set_xlabel('Cd [-]')
    axes[1].set_ylabel('Cl [-]')
    axes[0].legend()
    axes[1].legend()
    #axes[0].set_title('Cl and Cd vs Angle of Attack')
    axes[0].grid(True)
    axes[1].grid(True)

    ymin, ymax = axes[0].get_ylim()
    axes[0].set_ylim(*np.clip(axes[0].get_ylim(), -3, 3))
    axes[1].set_ylim(*np.clip(axes[1].get_ylim(), -3, 3))
    axes[1].set_xlim(*np.clip(axes[1].get_xlim(), 0, 3))

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


def read_forces_yaml(yaml_file, chord=1, span=1, dimensionless=True, verbose=False, tmin=None, tmax=None):
    """ 
    Read forces based on a yaml file, concatenate them if necessary
    """
    yml = None
    _, ext = os.path.splitext(yaml_file)
    if ext.lower() not in ['.yaml', '.yml', '.i']:
        raise Exception('Input should be yaml', yaml_file)
    yml = NALUInputFile(yaml_file, chord=chord, span=span)

    if dimensionless:
        df, Fref, csv_files = yml.read_surface_forces(tmin=tmin, tmax=tmax, verbose=verbose, Fref=1)
    else:
        df, Fref, csv_files = yml.read_surface_forces()

    return df, Fref, yml, csv_files


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


def polar_postpro(input_pattern, yaml_file=None, tmin=None, chord=1, span=None, verbose=False, polar_out=None, 
                  use_ss=False,
                  polars=None, plot=False, cfd_m = '', cfd_ls='-', cfd_c='k'):
    """ 
    INPUTS:
     - input_pattern: either:
          - a pattern to find nalu wind input files, in which case the force files are inferred by reading the yaml files
        OR- a pattern to find nalu wind force files, in which case yaml_file needs to be provided to find the ref force
    """
    dfp = None
    fig = None
    if span is None:
        span=1 # TODO automatic from yaml
    Aref = chord*span

    # --- Input files
    input_files, input_is_csv = input_files_from_patterns(input_pattern)
    if verbose:
        if len(input_files)==1:
            print('[INFO] Input file provided ({}):'.format(len(input_files)), input_files[0])
        else:
            print('[INFO] Input files provided ({}):'.format(len(input_files)), input_files[0], '...', input_files[-1])

    # Open Yaml file to get reference values
    if input_is_csv:
        if yaml_file is None:
            raise Exception('Provide yaml file if input are force files.')
        yml = NALUInputFile(yaml_file, chord=chord, span=span)
        Fref = yml.reference_force

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
                df = read_forces_csv(input_file, verbose=verbose, Fref=Fref)
            else:
                df, Fref, yml, csv_files = read_forces_yaml(input_file, chord=chord, span=span, dimensionless=True, verbose=verbose)
        except Exception as e:
            print(f"[FAIL] Error reading {input_file}: {e}")
            continue
        dfs.append(df)
        d = {'Alpha': aoa, 'Nt':len(df)}
        if use_ss:
            if len(df)<20:
                print('[WARN] Few data in file, not analyzing SS: '+input_file)
            else:
                ss = analyze_steady_state(df['Time'].values, df['Fpy'].values)
                d.update(ss)
        sss.append(d)
    dfss = pd.DataFrame(sss)

    # --- 
    if use_ss and 't_trans1' in dfss:
        t_onset_default = (dfss['t_trans1'].mean()+ dfss['t_trans2'].mean())/2
        if np.isnan(t_onset_default):
            print('[WARN] No steady-state ever detected, setting t_onset to None')
            t_onset_default=None
            use_ss =None
    else:
        use_ss =False
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
        fig = plot_polars(polars, cfd_ls=cfd_ls, cfd_c=cfd_c, cfd_m=cfd_m, verbose=verbose)
    return dfp, dfss, fig




def nalu_forces(input_files='forces.csv', tmin=None, tmax=None, 
                chord=1, rho=None, nu=None, U0=None, span=None, 
                yaml_file=None,
                polar_ref=None,
                polar_exp=None,
                polar_out='polar.csv',
                auto=True,
                dimensionless=True,
                var='xy',
                verbose=False,
                plot=True, cfd_ls='-'
                ):


    # --- Derived inputs
    FC, label= 'F', 'Forces [N]'
    if dimensionless:
        FC, label='C' , 'Coefficient [-]'

    if span is None:
        span=1  # TODO

    # --- Are we dealing with multiple files or one file?
    if isinstance(input_files, str):
        input_files = [input_files]
    patterns = input_files 
    input_files, input_is_csv = input_files_from_patterns(patterns)
    if len(input_files)==1:
        print('[INFO] Input file provided ({}):'.format(len(input_files)), input_files[0])
    else:
        print('[INFO] Input files provided ({}):'.format(len(input_files)), input_files[0], '...', input_files[-1])

    # --- Are we dealing with csv or yaml files
    if not input_is_csv:
        if yaml_file is not None:
            print('[INFO] Input files are yaml files instead of CSV files, overidding yaml_file')
        yaml_file = input_files [0]

    dft = None
    dfp = None
    figt = None
    figp = None

    # --- One input_file
    if len(input_files) == 1:
        for input_file in input_files:
            yml = None
            if input_is_csv:
                if dimensionless:
                    if auto:
                        # Read from yaml_file
                        if not os.path.exists(yaml_file):
                            raise Exception(f'Please provide a yaml file for dimensionless forces or set auto to False. yaml file not found: {yaml_file}')
                        if verbose:
                            print('[INFO] Reading velocity and density from input file:', yaml_file)
                        yml = NALUInputFile(yaml_file, chord=chord, span=span)
                        Fref = yml.reference_force
                    else:
                        if any(x is None for x in [chord, rho, nu, U0, span] ):
                            raise ValueError('For dimensionless forces, U0, rho and nu must be provided.')
                        Fref = 1/2 * rho * chord * span * U0**2 

                dft = read_forces_csv(input_file, tmin=tmin, tmax=tmax, verbose=verbose, Fref=Fref)
                sInfo =  'YML  : {}\n'.format(yaml_file)
                sInfo += 'CSV  : {}'.format(input_file)
            else:
                dft, Fref, yml, csv_files = read_forces_yaml(input_file, chord=chord, span=span, dimensionless=dimensionless, verbose=verbose, tmin=tmin, tmax=tmax)
                sInfo  = 'YML  : {}\n'.format(input_file)
                sInfo += 'CSV  : {}'.format(csv_files)

            if verbose:
                print(sInfo)
                if yml is not None:
                    U0  = np.linalg.norm(yml.velocity)
                    rho = yml.density
                    nu  = yml.viscosity
                if dimensionless:
                    print('chord: {:9.3f}  '.format(chord))
                    print('span : {:9.3f}  '.format(span))
                    print('U0   : {:9.3f}  '.format(U0))
                    print('rho  : {:9.3f}  '.format(rho))
                    print('nu   : {:9.3e}  '.format(nu))
                print('Re   : {:9.3f} Million'.format(U0 * chord /(nu*1e6)))
                print('Fref : {:9.3f} [N]'.format(Fref))
                print('nStep: {:d}'.format(len(dft)))
                print('Last : Time[-1]={:.3f}, {}x[-1]={:.3f}, {}y[-1]={:.3f}'.format(dft['Time'].values[-1], FC, dft['Cx'].values[-1], FC, dft['Cy'].values[-1]))

            # --- Plot time series
            if plot:
                figt, ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,5.8))
                figt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
                if 'x' in var:
                    ax.plot(dft['Time'].values, dft['Cx'].values, label=FC+'x')
                if 'y' in var:
                    ax.plot(dft['Time'].values, dft['Cy'].values, label=FC+'y')
                ax.set_ylabel(label)
                ax.set_xlabel('Time [s]')
                ax.legend()



    if polar_ref is None and len(input_files) == 1:
        # If no polar reference is provided, we only plot the time series
        return

    # ---
    dfp, dfss, figp = polar_postpro(patterns, yaml_file, tmin=tmin, chord=chord, span=span, verbose=verbose, polar_out=polar_out, polars={'ref':polar_ref, 'exp':polar_exp}, plot=plot, cfd_ls=cfd_ls)

    return dfp, figp, dft, figt


def nalu_forces_CLI():
    parser = argparse.ArgumentParser(description="Plot Nalu forces from a CSV file, optionally dimensionless.")
    parser.add_argument('input', type=str, nargs='*', default='*.yaml', help='Force file(s) or yaml input file from nalu-wind, or glob ("forces_aoa*.csv" or "input*.yaml"). Default is *.yaml')
    parser.add_argument('--tmin', type=float, default=None, help='Minimum time')
    parser.add_argument('--tmax', type=float, default=None, help='Maximum time')
    parser.add_argument('--chord', type=float, default=1.0, help='Airfoil chord')
    parser.add_argument('--rho', type=float, default=None, help='Density (overrides YAML)')
    parser.add_argument('--nu', type=float, default=None, help='Kinematic viscosity (overrides YAML)')
    parser.add_argument('--U0', type=float, default=None, help='Reference velocity (overrides YAML)')
    parser.add_argument('--dz', type=float, default=None, help='Spanwise thickness. Default is None (auto, 1 or 4 if "pp")')
    parser.add_argument('--scatter', action='store_true', help='Plot scatter instead of lines')
    parser.add_argument('--yaml', type=str, default='input.yaml', help='NALU input YAML file for auto properties')
    parser.add_argument('--polar-exp', type=str, default=None, help='CSV file with experimental polar data, alpha, Cl, Cd, Cm')
    parser.add_argument('--polar-ref', type=str, default=None, help='CSV file with reference polar data, alpha, Cl, Cd, Cm')
    parser.add_argument('--polar-out', type=str, default=None, help='CSV file with output polar if multiple files provied.')
    parser.add_argument('--no-auto', dest='auto', action='store_false', help='Do not auto-read velocity, density, etc from YAML')
    parser.add_argument('--forces', dest='dimensionless', action='store_false', help='Plot forces instead of coefficients')
    parser.add_argument('--var', type=str, default='xy', help='Which force components to plot (x, y, or xy)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-plot', action='store_true', help='Do not plot results')
    args = parser.parse_args()

    if args.scatter:
        ls=''
    else:
        ls='-'

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
        cfd_ls = ls,
    )
    plt.show()


if __name__ == '__main__':
    #plot_forces('forces.csv', verbose=True)
    nalu_forces_CLI()
    plt.show()
