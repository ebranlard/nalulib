""" Deal with nalu input file"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import ruamel.yaml

class RuamelYamlEditor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.yaml = ruamel.yaml.YAML()
        self.yaml.preserve_quotes = True
        with open(filepath, 'r', encoding='utf-8') as f:
            f.seek(0)
            self.data = self.yaml.load(f)

    @property
    def lines(self):
        buf = StringIO()
        self.yaml.dump(self.data, buf)
        return buf.getvalue().splitlines(keepends=True)

    def save(self, outpath=None):
        with open(outpath or self.filepath, 'w', encoding='utf-8') as f:
            f.writelines(self.lines)

    def print(self, context=True, line=True):
        for i, l in enumerate(self.lines):
            c = ""
            if line:
                l = "{:50s}".format(l.rstrip())
            else:
                l = ""
            if context:
                c = '# ' + str(self.get_context(i))
            print(f"{i}: {l} {c}")


    def extract_mesh_motion(self, plot=False, csv_file=None, export=False):
        """ Extract mesh motion from the YAML file and optionally plot or export it """
        data = self.data

        # --- Locate mesh_motion block
        mesh_motion = None
        if 'mesh_motion' in data:
            mesh_motion = data['mesh_motion']
            print('Found mesh_motion line')
        else:
            for realm in data.get('realms', []):
                if 'mesh_motion' in realm:
                    mesh_motion = realm['mesh_motion']
                    print('Found mesh_motion line in realms') 
                    break
        if mesh_motion is None:
            raise ValueError("No 'mesh_motion' block found in the YAML file.")
        

        # --- Store all motions in a DataFrame
        motions = mesh_motion[0]['motion']
        print(len(motions))
        records = []
        for motion in motions:
            dx, dy, dz = motion.get('displacement', [None, None, None])
            ax, ay, az = motion.get('axis', [None, None, None])
            ox, oy, oz = motion.get('origin', [None, None, None])
            if az is not None: 
                if np.abs(az)!=1.0 or np.abs(ax)>0 or np.abs(ay)>0:
                    raise Exception("Axis 0, 0, 1.0 for rotation")
            if oz is not None and np.abs(oz)>0.0:
                raise Exception("Origin z must be 0.0")
            records.append({
                'type': motion.get('type', 'unknown'),
                'dth': float(motion.get('angle', np.nan)),
                'start_time': float(motion.get('start_time', np.nan)),
                'end_time': float(motion.get('end_time', np.nan)),
                'ox': ox,
                'oy': oy,   
                'dx': dx,
                'dy': dy,   
                'dz': dz
            })
        df = pd.DataFrame.from_records(records)

        # Separate into rotation and translation DataFrames
        df_rot = df[df['type'] == 'rotation'].copy()
        df_tra = df[df['type'] == 'translation'].copy()

        # Compute cumulative values for each
        df_rot['Time [s]'] = df_rot['end_time'].cummax()
        df_tra['Time [s]'] = df_tra['end_time'].cummax()

        df_rot['dt'] = df_rot['end_time']-df_rot['start_time']
        df_tra['dt'] = df_tra['end_time']-df_rot['start_time']
        # Compute difference between last end time and current start time, offset by 1 index
        df_rot['dt_prev'] = df_rot['start_time'].shift(-1)-df_rot['end_time']
        df_tra['dt_prev'] = df_tra['start_time'].shift(-1)-df_tra['end_time']


        df_rot['ox'] = df_rot['ox'].cumsum()
        df_rot['oy'] = df_rot['oy'].cumsum()

        df_rot['angle']    = df_rot['dth'].cumsum()
        df_tra['x'] = df_tra['dx'].cumsum()
        df_tra['y'] = df_tra['dy'].cumsum()
        df_tra['z'] = df_tra['dz'].cumsum()

        print('df_rot:')
        print(df_rot)
        print('df_tra:')
        print(df_tra)

        # Drop unused columns
        df_rot = df_rot[['Time [s]', 'angle', 'ox', 'oy']]
        df_tra = df_tra[['Time [s]', 'x', 'y', 'z']]

        # Merge/interpolate on Time (outer join, then interpolate missing values)
        df = pd.merge(df_rot, df_tra, on='Time [s]', how='outer').sort_values('Time [s]').reset_index(drop=True)
        df = df.interpolate(method='linear', limit_direction='both')

        # --- Debug prints for DataFrames
        print('df:')
        print(df)

        if export: 
            if csv_file is None:
                csv_file = os.path.splitext(self.filepath)[0]+'_motion.csv'
            df.to_csv(csv_file, index=False)
            print('Motion data saved to:', csv_file)
        
        if plot:
            plt.figure()
            plt.plot(df['Time [s]'], df['angle'], "g", label="theta [deg]")
            plt.plot(df['Time [s]'], df['x'], "r", label="x [m]")
            plt.plot(df['Time [s]'], df['y'], "b", label="y [m]")
            plt.plot(df['Time [s]'], df['ox'], "--", label="ox [m]")
            plt.plot(df['Time [s]'], df['oy'], "--", label="oy [m]")
            plt.xlabel("t")
            plt.ylabel("Oscillation")
            plt.ylim([-15, 15])
            plt.grid()
            plt.legend()
            plt.show()


        #return types, angles, start_times, end_times, axes, origins, theta_deg_array, time_array
        return time_array, x, y, theta_deg_array




NALUInputFile = RuamelYamlEditor



if __name__ == '__main__':

    #yml = NALUInputFile('input2.yaml')
    #yml = NALUInputFile('_mesh_motion/input_restart_simpler.yaml')
    yml = NALUInputFile('_mesh_motion/input_restart.yaml')
    yml.extract_mesh_motion(plot=True, export=True)

    pass
