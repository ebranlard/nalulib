""" Deal with nalu input file"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

import yaml
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

# Local
from nalulib.exodus_info import exodus_get_names
# -----------------------------------------------------------------------------------
# --- Helper functions for sorting YAML keys
# -----------------------------------------------------------------------------------
# Recursively sort keys in mappings
# NOTE: recursivity will loose anchors and cause expansion of the mapping
def sort_keys_recursive(obj):
    if isinstance(obj, dict):
        sorted_map = CommentedMap()
        # Preserve anchor if present
        if hasattr(obj, 'anchor') and obj.anchor.value is not None:
            sorted_map.yaml_set_anchor(obj.anchor.value, always_dump=True)
        for k in sorted(obj):
            sorted_map[k] = sort_keys(obj[k])
            # preserve comments for each key
            if hasattr(obj, 'ca') and k in obj.ca.items:
                sorted_map.ca.items[k] = obj.ca.items[k]
        # preserve overall comments
        if hasattr(obj, 'ca'):
            sorted_map.ca.comment = obj.ca.comment
        return sorted_map
    elif isinstance(obj, list):
        return [sort_keys(i) for i in obj]
    else:
        return obj

def sort_keys_to_depth(obj, depth=1):
    if isinstance(obj, CommentedMap) and depth > 0:
        sorted_map = CommentedMap()
        # Copy overall comment
        if hasattr(obj, 'ca') and getattr(obj.ca, 'comment', None):
            sorted_map.ca.comment = obj.ca.comment
        keys = list(obj.keys())
        # Sort all keys except 'mesh_motion'
        keys_sorted = sorted([k for k in keys if k != 'mesh_motion'])
        if 'mesh_motion' in keys:
            keys_sorted.append('mesh_motion')
        for k in keys_sorted:
            v = obj[k]
            # Recursively sort one level deeper if possible
            if isinstance(v, CommentedMap):
                sorted_map[k] = sort_keys_to_depth(v, depth=depth-1)
            elif isinstance(v, list):
                sorted_map[k] = [
                    sort_keys_to_depth(i, depth=depth-1) if isinstance(i, CommentedMap) else i
                    for i in v
                ]
            else:
                sorted_map[k] = v
            # Copy comments for each key
            if hasattr(obj, 'ca') and k in obj.ca.items:
                sorted_map.ca.items[k] = obj.ca.items[k]
        return sorted_map
    elif isinstance(obj, list):
        return [sort_keys_to_depth(i, depth=depth) for i in obj]
    else:
        return obj

class YamlEditor:
    """ 
    Some problems:
      - yaml: sort keys, and discard comments
      - ruamel: perserve comments, can sort keys, but looses id and anchor information
    """
    def __init__(self, filename, reader='ruamel', sort=False):
        #if sort:
        #    reader = 'yaml'
        self.filename = filename
        self.reader=reader

        if self.reader=='ruamel':
            self.yaml = ruamel.yaml.YAML()
            self.yaml.preserve_quotes = True
            with open(filename, 'r', encoding='utf-8') as fid:
                fid.seek(0)
                self.data = self.yaml.load(fid)
        else:
            with open(filename, 'r', encoding='utf-8') as fid:
                self.data = yaml.safe_load(fid)

    def sort(self, inplace=False, depth=2):
        data = self.data
        if self.reader == 'ruamel':
            # Note: recurisive sort will loose anchors and cause expansion of the mapping
            # We can sort the first and second level for a NALU input file
            data = sort_keys_to_depth(data, depth=2) 
        if inplace:
            self.data = data
        else:
            return data

    def save(self, outpath=None, sort=False):
        if sort: 
            data = self.sort()
        else:
            data = self.data
        with open(outpath or self.filename, 'w', encoding='utf-8') as fid:
            if self.reader=='ruamel':
                self.yaml.dump(data, fid)
            else:
                yaml.dump(data, fid, default_flow_style=False)

    def lines(self):
        buf = StringIO()
        self.yaml.dump(self.data, buf)
        return buf.getvalue().splitlines(keepends=True)

    def print(self, context=True, line=True):
        """ Convenient function to show context"""
        for i, l in enumerate(self.lines()):
            c = ""
            if line:
                l = "{:50s}".format(l.rstrip())
            else:
                l = ""
            #if context:
            #    c = '# ' + str(self.get_context(i))
            print(f"{i}: {l} {c}")


class NALUInputFile(YamlEditor):

    def __init__(self, filename, reader='ruamel'):
        """ Initialize NALUInputFile with a YAML file path """
        super().__init__(filename=filename, reader=reader)

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
                csv_file = os.path.splitext(self.filename)[0]+'_motion.csv'
            df.to_csv(csv_file, index=False)
            print('Motion data saved to:', csv_file)
        
        if plot:
            plt.figure()
            plt.plot(df['Time [s]'], df['angle'], "-", label="theta [deg]")
            plt.plot(df['Time [s]'], df['x'], "-", label="x [m]")
            plt.plot(df['Time [s]'], df['y'], "-", label="y [m]")
            plt.plot(df['Time [s]'], df['ox'], "--", label="ox [m]")
            plt.plot(df['Time [s]'], df['oy'], "--", label="oy [m]")
            plt.xlabel("t")
            plt.ylabel("Oscillation")
            plt.ylim([-15, 15])
            plt.grid()
            plt.legend()
            plt.show()


        #return types, angles, start_times, end_times, axes, origins, theta_deg_array, time_array
        return df

    def check(self, verbose=True):
        """
        Checks that all 'meshes' keys exist and that BC domain names are present in the exodus file.
        If exo_file is provided, checks BC domains against exodus side-sets and block names.
        """
        data = self.data
        errors = []
        # Check for 'meshes' key at top level or in realms
        for i, realm in enumerate(data['realms']):
            mesh = realm.get('mesh', None)
            if verbose:
                print(f" - realm name: {realm['name']}")
                print(f"   mesh  file: {mesh}")
            if not os.path.exists(mesh):
                errors.append(f"Mesh file '{mesh}' does not exist.")
                continue

            names = exodus_get_names(mesh, lower=True)
            block_names = names.get('element_blocks', [])   
            if verbose:
                print('   side_sets :', names['side_sets'])
                print('   blocks    :', names['blocks'])
            # Check BC domains
            for bc in realm['boundary_conditions']:
                allowed = names['side_sets']
                if 'overset_user_data' in bc:
                    allowed = names['blocks']
                    domain_names = []
                    for mg in bc['overset_user_data']['mesh_group']:
                        domain_names += mg['mesh_parts'] if isinstance(mg['mesh_parts'], list) else [mg['mesh_parts']]
                else:
                    domain_names = bc['target_name'] if isinstance(bc['target_name'], list) else [bc['target_name']]
                    if verbose:
                        print('   bc domains:', domain_names)
                for domain in domain_names:
                    if domain.lower() not in allowed:
                        errors.append(f"BC domain '{domain}' in realm {i} not found in exodus file (allowed: {allowed}).")
            # Check for 'initial_conditions' key
            for ic in realm['initial_conditions']:
                domain_names = ic['target_name'] if isinstance(ic['target_name'], list) else [ic['target_name']]
                if verbose:
                    print('   ic domains:', domain_names)
                for domain in domain_names:
                    if domain.lower() not in names['blocks']:
                        errors.append(f"IC domain '{domain}' in realm {i} not found in exodus file (allowed: {names['blocks']}).")
            # Check for 'material_properties' key
            mat_names = realm['material_properties']['target_name']
            mat_names = mat_names if isinstance(mat_names, list) else [mat_names]
            if verbose:
                print('   materials :', mat_names)
            for mat_name in mat_names:
                if mat_name.lower() not in names['blocks']:
                    errors.append(f"Material '{mat_name}' in realm {i} not found in exodus file (allowed: {names['blocks']}).")

            # Check for 'mesh_transformation' key
            for mt in realm.get('mesh_transformation', []):
                for mp in mt['mesh_parts']:
                    if mp not in names['blocks']:
                        errors.append(f"Mesh part '{mp}' in realm {i} not found in exodus file (allowed: {names['blocks']}).")
            # Check for turbulence averaging
            if 'turbulence_averaging' in realm:
                if 'specifications' in realm['turbulence_averaging']:
                    for spec in realm['turbulence_averaging']['specifications']:
                        target_names = spec['target_name'] if 'target_name' in spec else []
                        if isinstance(target_names, str):
                            target_names = [target_names]
                        if verbose:
                            print('   turb targ :', target_names)
                        for target in target_names:
                            if target.lower() not in names['blocks']:
                                errors.append(f"Turbulence averaging target '{target}' in realm {i} not found in exodus file (allowed: {names['blocks']}).")

            # Check for mesh motion
            if 'mesh_motion' in realm:
                for mm in realm['mesh_motion']:
                    if 'mesh_parts' in mm:
                        mesh_parts = mm['mesh_parts']
                        if isinstance(mesh_parts, str):
                            mesh_parts = [mesh_parts]
                        if verbose:
                            print('   motion    :', mesh_parts)
                        for mp in mesh_parts:
                            if mp.lower() not in names['blocks']:
                                errors.append(f"Mesh motion part '{mp}' in realm {i} not found in exodus file (allowed: {names['blocks']}).")




        if errors:
            raise Exception("Issues found in input file {}:\n -".format(self.filename) + '\n -'.join(errors))
        else:
            print("[ OK ] All checks passed.")
        return errors

def nalu_input(input_file='input.yaml', sort=False, overwrite=False, check=False, reader='ruamel', verbose=False):
    """
    Main function to handle NALU input file operations.
    :param input_file: Path to the NALU YAML input file.
    :param sort: Whether to sort the YAML file.
    :param overwrite: Whether to overwrite the original file when sorting.
    :param check: Whether to check the YAML file for required keys and BC domains.
    :param exo_file: Exodus file for BC/domain checks.
    :param verbose: Whether to print verbose output.
    """
    if verbose:
        print('NALUInputFile: Reading', input_file)
    yml = NALUInputFile(input_file, reader=reader)
    if check:
        yml.check(verbose=verbose)
    if sort:
        if overwrite:
            yml.save(sort=True)
            if verbose:
                print(f"[INFO] File {input_file} sorted and overwritten.")
        else:
            outpath = input_file.replace(".yaml", "_sorted.yaml")
            yml.save(outpath=outpath, sort=True)
            if verbose:
                print(f"[INFO] Sorted file written to {outpath} (original not overwritten).")
                print(f"       yaml reader: {yml.reader}")


def nalu_input_CLI():
    import argparse
    parser = argparse.ArgumentParser(description="NALU input file utility")
    parser.add_argument("-i", "--input", type=str, help="Input NALU YAML file", default="input.yaml")
    parser.add_argument("--sort", action="store_true", help="Standardize/sort the YAML file")
    parser.add_argument("-overwrite", action="store_true", help="Overwrite input file (default: False)")
    parser.add_argument("--no-check", action="store_true", help="Do not check YAML file for existing files and domains")
    parser.add_argument("--reader", default='ruamel', help="Specify reader: 'yaml' or 'ruamel' (default: ruamel). Note: yaml sorts but looses comments. Ruamel sorts the first two levels only.", choices=['yaml', 'ruamel'])
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    nalu_input(input_file=args.input, sort=args.sort, overwrite=args.overwrite, check=not args.no_check, verbose=args.verbose, reader=args.reader)  

if __name__ == '__main__':

    ##yml = NALUInputFile('input2.yaml')
    #yml = NALUInputFile('_mesh_motion/input_restart_simpler.yaml')
    ##yml = NALUInputFile('_mesh_motion/input_restart.yaml')
    ##yml.extract_mesh_motion(plot=True, export=True)
    #yml.print()
    #pass

    nalu_input_CLI()
