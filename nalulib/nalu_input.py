""" Deal with nalu input file"""
import os
import numpy as np
import pandas as pd
import nalulib.pyplot as plt
from io import StringIO

import yaml
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

# Local
from nalulib.essentials import *
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

def find_key_recursive(obj, key):
    """ Recursively find a key in a nested structure """
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for k, v in obj.items():
            result = find_key_recursive(v, key)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_key_recursive(item, key)
            if result is not None:
                return result
    return None


class YamlEditor:
    """ 
    Some problems:
      - yaml: sort keys, and discard comments
      - ruamel: perserve comments, can sort keys, but looses id and anchor information
    """
    def __init__(self, filename=None, reader='ruamel', sort=False, profiler=False):
        #if sort:
        #    reader = 'yaml'
        self.filename = filename
        self.reader=reader

        if filename is not None:
            with Timer('Reading yaml file', silent = not profiler, writeBefore=True):
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


    def write(self, *args, **kwargs):
        self.save(*args, **kwargs)

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



def bc_target_names(realm):
    """ Returns a list of all target names for a given realm"""
    target_names = []
    for bc in realm['boundary_conditions']:
        if 'target_name' in bc:
            if isinstance(bc['target_name'], list):
                target_names.extend(bc['target_name'])
            else:
                target_names.append(bc['target_name'])
    return target_names

def bc_types(realm):
    """ Returns a list of all target names for a given realm"""
    bc_types = []
    for bc in realm['boundary_conditions']:
        if 'inflow_boundary_condition' in bc:
            bc_types.append('inflow')
        elif 'open_boundary_condition' in bc:
            bc_types.append('open')
        elif 'wall_boundary_condition' in bc:
            bc_types.append('wall')
        elif 'symmetry_boundary_condition' in bc:
            bc_types.append('symmetry')
        elif 'periodic_boundary_condition' in bc:
            bc_types.append('periodic')
        elif 'overset_boundary_condition' in bc:
            bc_types.append('overset')
        else:
            bc_types.append('unknown')
    return bc_types

def bc_summary(realm):
    """ Returns a list of all target names for a given realm"""
    bcs = []
    for bc in realm['boundary_conditions']:
        if 'inflow_boundary_condition' in bc:
            key = 'inflow'
        elif 'open_boundary_condition' in bc:
            key = 'open'
        elif 'wall_boundary_condition' in bc:
            key = 'wall'
        elif 'symmetry_boundary_condition' in bc:   
            key = 'symmetry'
        elif 'periodic_boundary_condition' in bc:
            key = 'periodic'
        elif 'overset_boundary_condition' in bc:
            key = 'overset'
        else:
            key = 'unknown'
        if 'target_name' in bc:
            target_names = bc['target_name']
        else:
            try:
                target_names = bc['overset_user_data'] ['mesh_group'][0]['mesh_parts']
            except KeyError:
                target_names = []
        bcs.append((key, target_names if isinstance(target_names, list) else [target_names]))

    return bcs

def time_dict(yml):
    """ return a dictionary with time information from the YAML file """
    try:
        restart = yml['realms'][0]['restart']
        it_start = restart['restart_start']
    except:
        it_start =0

    try:
        ti = yml['Time_Integrators'][0]['StandardTimeIntegrator']
        dt = ti['time_step']
        nt_max = ti['termination_step_count']
    except:
        dt = np.nan
        nt_max = np.nan
    tstart = it_start * dt
    return {'tstart': tstart, 'dt':dt, 'tmax':nt_max}

def solvers_list(yml):
    """ return a dictionary with solvers information from the YAML file """
    #try:
    solvers = []
    try:
        for lins in yml['linear_solvers']:
            solvers.append({'name': lins['name'], 'type': lins['type'], 'method': lins['method']})
    except:
        print('[WARN] No linear solvers found in YAML file')

    return solvers
#  equation_systems:
#    max_iterations: 4
#    name: theEqSys
#    solver_system_specification:
#      ndtw: solve_elliptic
#      pressure: solve_elliptic
#      specific_dissipation_rate: solve_scalar
#      turbulent_ke: solve_scalar
#      velocity: solve_mom
#    systems:
#    - WallDistance:
#        convergence_tolerance: 1e-8
#        max_iterations: 1
#        name: myNDTW
#    - LowMachEOM:
#        convergence_tolerance: 1e-8
#        max_iterations: 1
#        name: myLowMach
#    - ShearStressTransport:
#        convergence_tolerance: 1e-8
#        max_iterations: 1
#        name: mySST

def equation_systems_specs(realm):
    sys = realm['equation_systems']
    niter = sys['max_iterations'] if 'max_iterations' in sys else np.nan
    specs = sys['solver_system_specification']
    return specs, niter





def process_motion_list(motions):
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

    # Drop unused columns
    df_rot = df_rot[['Time [s]', 'angle', 'ox', 'oy']]
    df_tra = df_tra[['Time [s]', 'x', 'y', 'z']]

    # Merge/interpolate on Time (outer join, then interpolate missing values)
    df = pd.merge(df_rot, df_tra, on='Time [s]', how='outer').sort_values('Time [s]').reset_index(drop=True)
    df = df.interpolate(method='linear', limit_direction='both')

    # --- Debug prints for DataFrames
    #print('df_rot:')
    #print(df_rot)
    #print('df_tra:')
    #print(df_tra)
    #print('df:')
    #print(df)
    return df, df_rot, df_tra



class NALUInputFile(YamlEditor):

    def __init__(self, filename=None, reader='yaml', profiler=False):
        """ Initialize NALUInputFile with a YAML file path """
        super().__init__(filename=filename, reader=reader, profiler=profiler)

    def copy(self):
        new = NALUInputFile()
        new.reader = self.reader
        new.data = self.data.copy()
        new.filename = None
        return new

    def __repr__(self):
            s = f"<{type(self).__name__} object read with {self.reader}>\n"
            s += f" - filename: {self.filename}\n"
            realms = self.data.get('realms', [])
            s += f" * realms: {self.realm_names} ({len(realms)})\n"
            try:
                for i, realm in enumerate(realms):
                    s += f"   - realm {i}: name={realm.get('name', 'N/A')}\n"
                    mesh = realm.get('mesh', realm.get('meshes', 'N/A'))
                    s += f"     - mesh: {mesh}\n"
                    if 'material_properties' in realm:
                        mats = str(realm['material_properties'].get('target_name', 'N/A')).replace("'","").replace("[","").replace("]","")
                        s += f"     - material: {mats}\n"
                    if 'boundary_conditions' in realm:
                        bcs = str(bc_summary(realm)).replace("'","").replace(", [",":").replace("]","").replace("[","").replace("), ("," | ").replace("(", "").replace(")","")
                        s += f"     - boundary_conditions: {bcs}\n"
                        
                    if 'initial_conditions' in realm:
                        s += f"     - initial_conditions: {len(realm['initial_conditions'])}\n"
                        
                    if 'mesh_motion' in realm:
                        s += f"     - mesh_motion: present, size:{len(realm['mesh_motion'])}\n"
                        for mesh_motion in realm['mesh_motion']:
                            motions_list = mesh_motion['motion']
                            df, _, _ = process_motion_list(motions_list)
                            s += f"          - size:{len(df)} tmin:{df['Time [s]'].min():.3f} tmax:{df['Time [s]'].max():.3f} xmax:{df['x'].max():.3f} ymax:{df['y'].max():.3f} thmax:{df['angle'].max():.3f}\n"
                specs, niter = equation_systems_specs(realms[0])
                s += "     - equations_system: iterations:{},  specs:\n".format(niter)
            except Exception as e:
                s += f" [Error reading realms: {e}]\n"
            for k,v in specs.items():
                s += "         - {:30s}: {:15s}\n".format(k,v)
            try: 
                for i, sol in  enumerate(solvers_list(self.data)):
                    s += f" * solver[{i}] : name:{sol['name']:15s} type:{sol['type']:10s} method:{sol['method']:15s}\n"
            except Exception:
                s += f" * solvers : [not found]\n"

            s += f" * turbulence_model : {self.turbulence_model}\n" 
            s += f" * transition_model : {self.transition_model}\n" 

            # Computed properties
            nu  = self.viscosity
            vel = self.velocity
            rho = self.density
            U0  = np.linalg.norm(vel)
            Re  = rho*U0*1/nu/ 1e6
            time_dict = self.time_dict
            dt = time_dict['dt']
            dt_rec = 0.02 * 1  / U0
            s += f" * time      : {time_dict}  (dt_rec~{dt_rec:.4f} if chord=1)\n"
            s += f" * velocity  : {vel}\n"
            s += f" * density   : {rho}\n"
            s += f" * viscosity : {nu}  (nu)\n"
            s += f" * Reynolds  ~ {Re:.2f}M (if chord=1)\n"

            s += "methods:\n"
            s += " - sort, extract_mesh_motion, check, print"
            return s
    @property
    def realm_names(self):
        """Returns the name of the first realm, or None if not found."""
        return [r.get('name') for r in self.data['realms']]

    @property
    def velocity(self):
        for realm in self.data['realms']:
            for bc in realm['boundary_conditions']:
                #if 'inflow_boundary_condition' in bc:
                if 'inflow_user_data' in bc:
                    return bc['inflow_user_data']['velocity']
            for ic in realm['initial_conditions']:
                if 'value' in ic:
                    if 'velocity' in ic['value']:
                        return ic['value']['velocity']
        #raise Exception('Unable to extract velocity from yaml file')
        return np.nan

    @velocity.setter
    def velocity(self, new_velocity):
        for realm in self.data['realms']:
            for bc in realm['boundary_conditions']:
                if 'inflow_user_data' in bc:
                    bc['inflow_user_data']['velocity'] = new_velocity
            for ic in realm['initial_conditions']:
                if 'value' in ic and 'velocity' in ic['value']:
                    ic['value']['velocity'] = new_velocity


    @property
    def density(self):
        for realm in self.data['realms']:
            #if 'material_properties' in realm:
            for specs in realm['material_properties']['specifications']:
                if specs['name'] == 'density':
                    return specs['value']
        #raise Exception('Unable to extract density from yaml file')
        return np.nan

    @density.setter
    def density(self, new_density):
        for realm in self.data['realms']:
            for specs in realm['material_properties']['specifications']:
                if specs['name'] == 'density':
                    specs['value'] = new_density

    @property
    def viscosity(self):
        """ kinematic viscosity , nu = mu/rho """
        for realm in self.data['realms']:
            #if 'material_properties' in realm:
            for specs in realm['material_properties']['specifications']:
                if specs['name'] == 'viscosity':
                    return specs['value']
        #raise Exception('Unable to extract density from yaml file')
        return np.nan

    @viscosity.setter
    def viscosity(self, new_viscosity):
        for realm in self.data['realms']:
            for specs in realm['material_properties']['specifications']:
                if specs['name'] == 'viscosity':
                    specs['value'] = new_viscosity

    @property
    def turbulence_model(self):
        model = find_key_recursive(self.data, 'turbulence_model')
        if model is None:
            return 'unknown'
            #raise Exception('Unable to extract turbulence_model from yaml file')
        return model

    @property
    def transition_model(self):
        transition = find_key_recursive(self.data, 'transition_model')
        if transition is None:
            return 'unknown'
            #raise Exception('Unable to extract transition_model from yaml file')
        return transition
    
    @property
    def bc_target_names(self):
        """ Returns a list of all target names for boundary conditions """
        target_names = []
        for realm in self.data['realms']:
            for bc in realm['boundary_conditions']:
                if 'target_name' in bc:
                    if isinstance(bc['target_name'], list):
                        target_names.extend(bc['target_name'])
                    else:
                        target_names.append(bc['target_name'])
        return target_names

    @property
    def time_dict(self):
        """ Returns [tstart, dt, tmax] """
        return time_dict(self.data)

    @property
    def solvers_dict(self):
        """ Returns [tstart, dt, tmax] """
        return solvers_dict(self.data)

    def extract_mesh_motion(self, plot=False, csv_file=None, export=False):
        """ Extract mesh motion from the YAML file and optionally plot or export it """
        data = self.data

        # --- Locate mesh_motion block
        mesh_motion = None
        if 'mesh_motion' in data:
            mesh_motion = data['mesh_motion']
            print('Found mesh_motion line')
        else:
            for irealm, realm in enumerate(data.get('realms', [])):
                if 'mesh_motion' in realm:
                    mesh_motion = realm['mesh_motion']
                    print('Found mesh_motion line in realms', irealm) 
                    break
        if mesh_motion is None:
            raise ValueError("No 'mesh_motion' block found in the YAML file.")
        

        # --- Store all motions in a DataFrame
        motions = mesh_motion[0]['motion']
        print(len(motions))

        df, df_rot, df_tra = process_motion_list(motions)

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



    def set_sine_motion(self, A, f, n_periods, t_steady=0, dt=None, DOF='pitch', plot=False, irealm=0):
        if dt is None:
            dt = self.time_dict['dt']
        t, x, y, theta = sine_motion(A, f, n_periods, t_steady, dt, DOF=DOF)
        self.set_motion(t, x, y, theta, plot=plot, irealm=irealm)
        return t, x, y, theta


    def set_motion(self, t, x, y, theta, plot=False, irealm=0):
        mesh_motion = generate_mesh_motion(t, x, y, theta, plot=plot)
        self.data['realms'][irealm]['mesh_motion'] = [mesh_motion]
        if plot:
            self.extract_mesh_motion(plot=plot)

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
                    if mp.lower() not in names['blocks']:
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


            # Check for time step consistency
            vel = self.velocity
            time_dict = self.time_dict
            dt = time_dict['dt']
            dt_rec = 0.02 * 1  / np.linalg.norm(vel)
            if (np.abs(dt-dt_rec)/dt_rec) > 0.5:
                errors.append(f"Time step dt={dt:.4f} is inconsistent with expected dt_rec={dt_rec:.4f} (if chord=1).")


        if errors:
            raise Exception("Issues found in input file {}:\n -".format(self.filename) + '\n -'.join(errors))
        else:
            print("[ OK ] All checks passed.")
        return errors


def sine_motion(A, f, n_periods, t_steady, dt, DOF='pitch'):

    def sine_with_steady(A, f, n_periods, t_steady, dt):
        num_iter_steady=int(round(t_steady / dt))
        #------------------------------- Steady state part --------------------------
        x_st = np.linspace(0.0, 0.0, num_iter_steady + 1)
        time_st = np.linspace(0.0, t_steady, num_iter_steady + 1)

        #----------------------------- Dynamic part ---------------------------
        T = 1 / f
        sin_duration = n_periods * T
        sin_num_iter = int(round(sin_duration / (dt)))
        sin_duration = int(round(sin_duration / (dt)))*dt

        time_sin = np.linspace(0.0, sin_duration, sin_num_iter + 1)
        x_sine   = A * np.sin(2 * np.pi * f * time_sin)
        t = np.concatenate((time_st[:-1], time_st[-1] + time_sin))
        vel = 0*t
        x_t = np.concatenate((x_st, x_sine[:-1]))
        return x_t,  t, vel
    
    if DOF.lower() in ['pitch','theta']:
        theta, t, vel = sine_with_steady(A, f, n_periods, t_steady, dt)
        x=t*0
        y=t*0
    elif DOF.lower() in ['x','flap']:
        x, t, vel = sine_with_steady(A, f, n_periods, t_steady, dt)
        theta=t*0
        y=t*0
    elif DOF.lower() in ['y','edge']:
        y, t, vel = generate_sine(A, f, n_periods, t_steady, dt)
        x=t*0
        theta=t*0
    return t, x, y, theta

def generate_mesh_motion(t, x, y, theta, plot=False):

    delta_theta = [j - i for i, j in zip(theta[: -1], theta[1 :])]
    delta_x= [x2-x1 for x1, x2 in zip(x[: -1], x[1 :])]
    delta_y= [y2-y1 for y1, y2 in zip(y[: -1], y[1 :])]

    mesh_motion = {
                'name': 'arbitrary_motion_airfoil',
                'mesh_parts': ['fluid-HEX'],
                'frame': 'non_inertial',
                'motion': []
            }

    if np.any(theta != 0):
        for i in range(len(t) - 1):
            motion_block = {
                'type': 'rotation',
                'angle': float(delta_theta[i]),
                #-------------------Sheryas' code---------
                'start_time': float(t[i] + 1e-6),
                'end_time': float(t[i + 1]),
                #-------------------Sheryas' code---------
                'axis': [0.0, 0.0, 1.0],
                'origin': [float(delta_x[i]), float(delta_y[i]), 0.0]
            }
            mesh_motion['motion'].append(motion_block)
        
    elif np.any(x != 0) or np.any(y != 0):
        for i in range(len(t) - 1):
            motion_block = {
                'type': 'translation',
                'start_time': float(t[i] + 1e-6),
                'end_time': float(t[i + 1]),
                'displacement': [float(delta_x[i]), float(delta_y[i]), 0.0]
                #'velocity'
            }
            mesh_motion['motion'].append(motion_block)
    return mesh_motion



def nalu_input(input_file='input.yaml', sort=False, overwrite=False, check=False, reader='yaml', 
               plot_motion=False, verbose=False, profiler=False):
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
    yml = NALUInputFile(input_file, reader=reader, profiler=profiler)

    # --- Print class to file
    print(yml)

    if check:
        yml.check(verbose=verbose)
    if sort:
        if overwrite:
            yml.save(sort=True)
            print(f"[INFO] Sorted file written to: {input_file}   (overwritten)")
        else:
            outpath = input_file.replace(".yaml", "_sorted.yaml")
            yml.save(outpath=outpath, sort=True)
            print(f"[INFO] Sorted file written to: {outpath}   (otherwise use --overwrite).")
            #print(f"       yaml reader: {yml.reader}")
    if plot_motion:
        yml.extract_mesh_motion(plot=True)


def nalu_input_CLI():
    import argparse
    parser = argparse.ArgumentParser(description="NALU input file utility")
    parser.add_argument("-i", "--input", type=str, help="Input NALU YAML file", default="input.yaml")
    parser.add_argument("--sort", action="store_true", help="Standardize/sort the YAML file")
    parser.add_argument("-overwrite", action="store_true", help="Overwrite input file (default: False)")
    parser.add_argument("--no-check", action="store_true", help="Do not check YAML file for existing files and domains")
    parser.add_argument("--reader", default='yaml', help="Specify reader: 'yaml' or 'ruamel' (default: ruamel). Note: yaml sorts but looses comments. Ruamel sorts the first two levels only.", choices=['yaml', 'ruamel'])
    parser.add_argument("--profiler", action="store_true", help="Enable profiling with timers.")
    parser.add_argument("--plot-motion", action="store_true", help="Extract and Plot mesh motion")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    nalu_input(input_file=args.input, sort=args.sort, overwrite=args.overwrite, check=not args.no_check, verbose=args.verbose, reader=args.reader, profiler=args.profiler, plot_motion=args.plot_motion)  

if __name__ == '__main__':

    ##yml = NALUInputFile('input2.yaml')
    yml = NALUInputFile('_mesh_motion/input_restart_simpler.yaml')
    yml.set_sine_motion(A=10, f=1, n_periods=1, t_steady=20, DOF='pitch', plot=True, irealm=1)
    yml.write('_DUMMY.yml')
    ##yml = NALUInputFile('_mesh_motion/input_restart.yaml')
    #yml.extract_mesh_motion(plot=True, export=True)
    #yml.print()
    #pass

    nalu_input_CLI()
