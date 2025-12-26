import numpy as np
import pandas as pd
from nalulib.weio.csv_file import CSVFile



def read_forces_csv(input_file='forces.csv', tmin=None, tmax=None, verbose=False, Fref=None):
    df0 = CSVFile(input_file).toDataFrame()
    df = pd.concat([pd.DataFrame([[0]*len(df0.columns)], columns=df0.columns), df0], ignore_index=True)

    if tmin is not None:
        df = df[df['Time']>tmin]
    if tmax is not None:
        df = df[df['Time']<tmax]

    if verbose:
        print('nSteps:',len(df), len(df0))

    df['Fx'] = ( df['Fpx'].values + df['Fvx'].values)
    df['Fy'] = ( df['Fpy'].values + df['Fvy'].values)

    if Fref is not None:
        df['Cx'] = df['Fx'] / Fref
        df['Cy'] = df['Fy'] / Fref   

    return df
