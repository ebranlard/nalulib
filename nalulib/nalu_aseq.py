"""
Takes a template yamls file assumed to be at zero angle of attack and gnerate the input files for a sequence of angles of attack.
"""
import re
import os
import ruamel.yaml
import numpy as np
from io import StringIO
import glob
# Local
from nalulib.essentials import *
from nalulib.nalu_input import *
from nalulib.nalu_batch import nalu_batch
from nalulib.exodus_rotate import rotate_exodus



def nalu_prepare_aseq(input_file, aseq=None, verbose=False, debug=False, batch_file=None, cluster='unity', aoa_ori=None, jobname=''):
    myprint('Input YAML file', input_file)
    # Basename
    base_dir = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    base, ext = os.path.splitext(base_name)

    if aseq is None:
        aseq = np.arange(-15, 15+3/2, 3)
    else:
        if len(aseq) == 3:
            aseq = np.arange(aseq[0], aseq[1]+aseq[2]/2, aseq[2])
        else:
            pass
    myprint('AoA sequence', aseq)


    # --- Read YAML file
    yml_ori = NALUInputFile(input_file)
    yml = NALUInputFile(input_file)
    realms_ori = yml_ori.data['realms']
    realms = yml.data['realms']
    if len(realms) > 1:
        print('[INFO] Multiple realms found, using the second one')
        realm  = realms[1]
    else:
        realm  = realms[0]


    mesh_ori = realm['mesh']
    myprint('Mesh', mesh_ori)

    if aoa_ori is None:
        aoa_ori = extract_aoa(mesh_ori)
    if aoa_ori is None:
        aoa_ori=0
        print('[INFO] No AoA found in mesh name, assuming 0. Change it with --aoa_ori.')
    
    # --- Mesh creation
    print('-------------------------------------------------------------------------------------')
    mesh_dir = os.path.join(base_dir, 'meshes')
    myprint('Creating meshes in : ', mesh_dir)
    center = (0.0, 0.0)
    # creating a directory "meshes" if it does not exist
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    angles = aseq - aoa_ori
    mesh_files = [os.path.join(mesh_dir, base+'_mesh_aoa{:.1f}.exo'.format(alpha)) for alpha in aseq]
    # 
    rotate_exodus(input_file=mesh_ori, output_file=mesh_files, angle=angles, center=center, verbose=verbose)
    print('-------------------------------------------------------------------------------------')


    # --- Create YAML files
    nalu_files = [os.path.join(base_dir, base+'_aoa{:.1f}.yaml'.format(alpha)) for alpha in aseq]
        
    for alpha, nalu_file, mesh_file in zip(aseq, nalu_files, mesh_files):
        sAOA = '_aoa{:.1f}'.format(alpha)
        # --- Change mesh
        mesh_rel = os.path.relpath(mesh_files[0], os.path.dirname(nalu_file))
        realm['mesh'] = mesh_rel.replace('\\','/')

        # --- Change output filenames.
        for i in range(len(realms)):
            realm_ori = realms_ori[i]
            realm = realms[i]
            if 'output' in realm:
                outbase, ext = os.path.splitext(realm_ori['output']['output_data_base_name'])
                realm['output']['output_data_base_name']  =  outbase + sAOA + ext
            if 'restart' in realm:
                resbase, ext = os.path.splitext(realm_ori['restart']['restart_data_base_name'])
                realm['restart']['restart_data_base_name']  =  resbase + sAOA + ext
            # --- Change postprocessing
            if 'post_processing' in realm:
                for j in range(len(realm['post_processing'])):
                    outbase, ext = os.path.splitext(realm_ori['post_processing'][j]['output_file_name'])
                    realm['post_processing'][j]['output_file_name'] =  outbase + sAOA + ext
            if 'sideset_writers' in realm:
                for j in range(len(realm['sideset_writers'])):
                    outbase, ext = os.path.splitext(realm_ori['sideset_writers'][j]['output_data_base_name'])
                    realm['sideset_writers'][j]['output_data_base_name'] =  outbase + sAOA + ext

        yml.save(nalu_file)
        myprint('Written NALU File', nalu_file)








    # --- Create BATCH files
    for alpha, nalu_file  in zip(aseq, nalu_files):
        jobname += 'A{:.1f}'.format(alpha)
        new_batch = nalu_batch(batch_file_template=batch_file, nalu_input_file=nalu_file, cluster=cluster, verbose=verbose, jobname=jobname)
        myprint('Written Batch File', new_batch)




#
#    print('--- Editing YAML file for restart')
#    # --- Edit yaml file
#    for i in range(len(realms)):
#        if debug:
#            print('Realm', i, realms[i]['name'])
#        try:
#            restart = realms[i]['restart']
#        except:
#            raise Exception('Cannot do reload without a restrt')
#        if debug:
#            print('Restart', restart)
#
#        start_old = restart['restart_start'] if 'restart_start' in restart else None
#        time_old = restart['restart_time'] if 'restart_time' in restart else None
#
#        restart['restart_start'] = iRestart
#        restart['restart_time'] = tRestart
#        myprint('restart_start', start_old, restart['restart_start'])
#        myprint('restart_time',  time_old, restart['restart_time'])
#
#        # --- Change mesh
#        mesh_ori = realms[i]['mesh']
#        mesh_new = restart['restart_data_base_name']
#        realms[i]['mesh'] = mesh_new
#        myprint('Mesh', mesh_ori, mesh_new)
#        # --- Change mesh
#        if 'automatic_decomposition_type' in realms[i]:
#            try:
#                myprint('automatic_decomposition:', realms[i]['automatic_decomposition_type'], 'DELETED')
#                del   realms[i]['automatic_decomposition_type']
#            except:
#                print('[FAIL] Failed to delete automatic_decomposition_type')
#                pass
#        else:
#            print('[INFO] automatic_decomposition_type absent')
#        
#        # --- Rebalance not needed
#        if 'rebalance_mesh' in realms[i]:
#            try:
#                myprint('rebalance_mesh:', realms[i]['rebalance_mesh'], 'DELETED')
#                del   realms[i]['rebalance_mesh']
#            except:
#                print('[FAIL] Failed to delete rebalance_mesh')
#                pass
#        if 'stk_rebalance_method' in realms[i]:
#            try:
#                myprint('stk_rebalance_method:', realms[i]['stk_rebalance_method'], 'DELETED')
#                del   realms[i]['stk_rebalance_method']
#            except:
#                print('[FAIL] Failed to delete stk_rebalance_method')
#                pass
#
#        # --- Change outputs
#        if 'output' in realms[i]:
#            if 'output_start' in realms[i]['output']:
#                output_start_old = realms[i]['output']['output_start']
#            else:
#                output_start_old = None
#            realms[i]['output']['output_start'] = iRestart
#            myprint('output_start: ', output_start_old, realms[i]['output']['output_start']) 
#        else:
#            print('[INFO] No output in realm', i)
#        # --- Change postprocessing
#        if 'post_processing' in realms[i]:
#            for j in range(len(realms[i]['post_processing'])):
#                postpro = realms[i]['post_processing'][j]
#                outfile = postpro['output_file_name']
#                print('Postprocessing', j, outfile)
#                base, ext = os.path.splitext(outfile)
#                sp = base.split('_')
#                if nrun is None:
#                    # --- Infer n restart from the file name
#                    # Check if the last _2 is a number, if it is increment it, otherwise, add _2
#                    print('>>> sp' ,sp, sp[-1].isdigit())
#                    if sp[-1].isdigit():
#                        nrun = int(sp[-1])+1
#                        base= '_'.join(sp[:-1])
#                    else:
#                        nrun = 2
#                myprint('[INFO] Number of run', nrun)
#                base  += '_' + str(nrun)    
#                postpro['output_file_name'] = base + ext
#                new_outfile = base + ext
#                # --- Change mesh
#                myprint('output_file_name: ', outfile, new_outfile)
#        else:
#            print('[INFO] No postprocessing in realm', i)
#
#
#    try:
#        myprint('hypre_config: ', 'misc' , 'DELETED')
#        del yml.data['hypre_config']
#    except:
#        pass
#
#    # --- 
#    if output_file is None:
#        base, ext = os.path.splitext(input_file)
#        sp = input_file.split('_run')
#        if len(sp) > 1:
#            base = '_'.join(sp[:-1])
#        output_file = base + '_run{}.yaml'.format(nrun)
#    yml.save(output_file)
#    myprint('Written              ', output_file)



def nalu_aseq():
    """
    Command-line entry point for preparing a Nalu-Wind aseq
    """
    import argparse

    parser = argparse.ArgumentParser(description="Setup Nalu-Wind for multiple AoA simulations (YAML and batch files).")
    parser.add_argument("input_file", default='input.yaml', nargs="?", help="Input YAML file")
#    parser.add_argument("-o", "--output_file", default=None, help="Output YAML file (optional, default: input_runNRUN.yaml)")
    parser.add_argument("-a", type=float, nargs=3, default=None, help="Alpha sequence (start, stop, step) (optional, default: -15 to 15 in steps of 3)")
    parser.add_argument("-b", "--batch_file", default=None, help="Batch file template (optional)")
    parser.add_argument("-j", "--jobname", default=None, help="Jobname prefix (optional)")
    parser.add_argument("--cluster", default='unity', choices=["unity", "kestrel"], help="Cluster type")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args() 
    if args.verbose:
        print('Arguments:', args)

    nalu_prepare_aseq(
        input_file=args.input_file,
        aseq=args.a,
#        output_file=args.output_file,
        verbose=args.verbose,
        batch_file=args.batch_file,
        cluster=args.cluster,
        jobname=args.jobname
    )

if __name__ == "__main__":
    nalu_aseq()
    #test_yaml = "input.yaml"
    #nalu_prepare_restart(test_yaml, it=None, output_file=None, batch_file=None, cluster='unity', nrun=None)
