import re
import os
import numpy as np
import glob
# Local
from nalulib.essentials import *
from nalulib.nalu_input import NALUInputFile
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus import exo_get_times

def myprint(s1, s2, s3=None):
    if s3 is None:
        print('{:30s}: '.format(s1) + '{:20s}'.format(str(s2)))
    else:
        print('{:30s}: '.format(s1) + '{:20s}'.format(str(s2))+' -> '+'{:20s}'.format(str(s3)) )

def nalu_restart(yaml_file, it=None, output_file=None, verbose=False, debug=False, nt_max_tol=10, nrun=None, batch_file=None, cluster='unity'):
    myprint('Input YAML file', yaml_file)

    yml = NALUInputFile(yaml_file, reader='ruamel') # NOTE: using ruamel to keep comments
    #    start_time: 0
    #    termination_step_count: 3760
    #    time_step: 0.000266667
    #    time_step_count: 0
    #    time_stepping_type: fixed
    ti = yml.data['Time_Integrators'][0]['StandardTimeIntegrator']
    if debug:
        print('Time_Integrators', ti)

    dt = ti['time_step']
    nt_max = ti['termination_step_count']
    if verbose:
        myprint('dt',dt)
        myprint('nt_max',nt_max)


    # --- Infer iRestart
    print('--- Inferring iRestart')
    realms = yml.data['realms']
    if it is None:
        print('Trying to infer iRestart from restart files')
        iRestarts = []
        for i in range(len(realms)):
            if debug:
                print('Realm', i, realms[i]['name'])
            # --- Try to figure out last restart
            try:
                restart = realms[i]['restart']
            except:
                raise Exception('Cannot do reload without a restart')

            # --- Infer number of time steps from last restart
            restart_mesh = restart['restart_data_base_name'].replace('\\', '/')
            pattern = restart_mesh + '.*.*'  # Matches result/output.rst.16.00, result/output.rst.0001.0000, etc.
            myprint('Pattern to look for restart', pattern)
            candidates = glob.glob(pattern)

            # Only keep files that end with .digits.digits
            restart_files = [f for f in candidates if re.search(r'\.\d+\.\d+$', f)]
            if len(restart_files) == 0:
                raise Exception('Cannot infer restart time from restart files.\nNo restart files found with pattern: {}.\nMake sure simulations were run, or,  explicitely set the restart time with the flag --it NEW_TIME_STEP .'.format(pattern))

            myprint('Restart file', restart_files[0])
            times = exo_get_times(restart_files[0])
            last_time = times[-1]
            iRestart = int(last_time/dt)+1
            if iRestart*dt> (times[-1]+dt/2):
                iRestart -= 1
            iRestarts.append(iRestart)
        iRestart = min(iRestarts)
    else:
        iRestart = it
    tRestart = iRestart*dt
    myprint('iRestart', iRestart)
    myprint('tRestart', iRestart*dt)
    if tRestart > (nt_max-nt_max_tol)*dt:
        print('[WARN] iRestart is larger or close to nt_max, doubling nt_max')
        nt_max_new = 2*nt_max
        ti['termination_step_count'] = nt_max_new
        myprint('nt_max', nt_max, nt_max_new)

    print('--- Editing YAML file for restart')
    # --- Edit yaml file
    for i in range(len(realms)):
        if debug:
            print('Realm', i, realms[i]['name'])
        try:
            restart = realms[i]['restart']
        except:
            raise Exception('Cannot do reload without a restrt')
        if debug:
            print('Restart', restart)

        start_old = restart['restart_start'] if 'restart_start' in restart else None
        time_old = restart['restart_time'] if 'restart_time' in restart else None

        restart['restart_start'] = iRestart
        restart['restart_time'] = tRestart
        myprint('restart_start', start_old, restart['restart_start'])
        myprint('restart_time',  time_old, restart['restart_time'])

        # --- Change mesh
        mesh_ori = realms[i]['mesh']
        mesh_new = restart['restart_data_base_name']
        realms[i]['mesh'] = mesh_new
        myprint('Mesh', mesh_ori, mesh_new)
        # --- Change mesh
        if 'automatic_decomposition_type' in realms[i]:
            try:
                myprint('automatic_decomposition:', realms[i]['automatic_decomposition_type'], 'DELETED')
                del   realms[i]['automatic_decomposition_type']
            except:
                print('[FAIL] Failed to delete automatic_decomposition_type')
                pass
        else:
            print('[INFO] automatic_decomposition_type absent')
        
        # --- Rebalance not needed
        if 'rebalance_mesh' in realms[i]:
            try:
                myprint('rebalance_mesh:', realms[i]['rebalance_mesh'], 'DELETED')
                del   realms[i]['rebalance_mesh']
            except:
                print('[FAIL] Failed to delete rebalance_mesh')
                pass
        if 'stk_rebalance_method' in realms[i]:
            try:
                myprint('stk_rebalance_method:', realms[i]['stk_rebalance_method'], 'DELETED')
                del   realms[i]['stk_rebalance_method']
            except:
                print('[FAIL] Failed to delete stk_rebalance_method')
                pass

        # --- Change outputs
        if 'output' in realms[i]:
            if 'output_start' in realms[i]['output']:
                output_start_old = realms[i]['output']['output_start']
            else:
                output_start_old = None
            realms[i]['output']['output_start'] = iRestart
            myprint('output_start: ', output_start_old, realms[i]['output']['output_start']) 
        else:
            print('[INFO] No output in realm', i)
        # --- Change postprocessing
        if 'post_processing' in realms[i]:
            for j in range(len(realms[i]['post_processing'])):
                postpro = realms[i]['post_processing'][j]
                outfile = postpro['output_file_name']
                print('Postprocessing', j, outfile)
                base, ext = os.path.splitext(outfile)
                sp = base.split('_')
                if nrun is None:
                    # --- Infer n restart from the file name
                    # Check if the last _2 is a number, if it is increment it, otherwise, add _2
                    #print('>>> sp' ,sp, sp[-1].isdigit())
                    if sp[-1].isdigit():
                        nrun = int(sp[-1])+1
                        base= '_'.join(sp[:-1])
                    else:
                        nrun = 2
                myprint('[INFO] Number of run', nrun)
                base  += '_' + str(nrun)    
                postpro['output_file_name'] = base + ext
                new_outfile = base + ext
                # --- Change mesh
                myprint('output_file_name: ', outfile, new_outfile)
        else:
            print('[INFO] No postprocessing in realm', i)


    try:
        myprint('hypre_config: ', 'misc' , 'DELETED')
        del yml.data['hypre_config']
    except:
        pass

    # --- 
    if output_file is None:
        base, ext = os.path.splitext(yaml_file)
        sp = yaml_file.split('_run')
        if len(sp) > 1:
            base = '_'.join(sp[:-1])
        output_file = base + '_run{}.yaml'.format(nrun)
    yml.save(output_file)
    myprint('Written YAML File    ', output_file)



    # --- Create batch file
    print('--- Creating batch file')
    new_batch = nalu_batch(batch_file, output_file, cluster=cluster, verbose=verbose, jobname=None)
    myprint('Written Batch File   ', new_batch)



def nalu_restart_CLI():
    """
    Command-line entry point for preparing a Nalu-Wind restart YAML and batch file.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Nalu-Wind restart YAML and batch file.")
    parser.add_argument("yaml_file", default='input.yaml', nargs="?", help="Input YAML file")
    parser.add_argument("--it", type=int, default=None, help="Restart integer time step (optional, inferred from restart if not given)")
    parser.add_argument("-o", "--output_file", default=None, help="Output YAML file (optional, default: input_runNRUN.yaml)")
    parser.add_argument("-b", "--batch_file", default=None, help="Batch file template (optional)")
    parser.add_argument("--cluster", default='unity', choices=["unity", "kestrel"], help="Cluster type")
    parser.add_argument("-n", "--nrun", type=int, default=None, help="Run index for output naming (optional, inferred from output filename)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args() 
    if args.verbose:
        print('Arguments:', args)

    nalu_restart(
        yaml_file=args.yaml_file,
        it=args.it,
        output_file=args.output_file,
        verbose=args.verbose,
        debug=args.debug,
        nrun=args.nrun,
        batch_file=args.batch_file,
        cluster=args.cluster
    )

if __name__ == "__main__":
    nalu_restart_CLI()
    #test_yaml = "input.yaml"
    #nalu_prepare_restart(test_yaml, it=None, output_file=None, batch_file=None, cluster='unity', nrun=None)
