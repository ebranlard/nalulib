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



def nalu_prepare_aseq(input_file, aseq=None, verbose=False, debug=False, batch_file=None, cluster='unity', aoa_ori=None, jobname='',
        inlet_name='inlet', outlet_name='outlet', block_base='fluid'):
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
    print('----------------------------------------------------------------------------')
    mesh_dir = os.path.join(base_dir, 'meshes')
    myprint('Creating meshes in : ', mesh_dir)
    center = (0.0, 0.0)
    # creating a directory "meshes" if it does not exist
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    angles = aseq - aoa_ori
    mesh_files = [os.path.join(mesh_dir, base+'_mesh_aoa{:.1f}.exo'.format(alpha)) for alpha in aseq]
    # 
    rotate_exodus(input_file=mesh_ori, output_file=mesh_files, angle=angles, center=center, verbose=verbose, inlet_name=inlet_name, outlet_name=outlet_name)
    print('----------------------------------------------------------------------------')


    # --- Create YAML files
    nalu_files = [os.path.join(base_dir, base+'_aoa{:.1f}.yaml'.format(alpha)) for alpha in aseq]
        
    for alpha, nalu_file, mesh_file in zip(aseq, nalu_files, mesh_files):
        sAOA = '_aoa{:.1f}'.format(alpha)
        # --- Change mesh
        mesh_rel = os.path.relpath(mesh_file, os.path.dirname(nalu_file))
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
        jobname_case = jobname + 'A{:.1f}'.format(alpha)
        new_batch = nalu_batch(batch_file_template=batch_file, nalu_input_file=nalu_file, cluster=cluster, verbose=verbose, jobname=jobname_case)
        myprint('Written Batch File', new_batch, jobname_case)


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
    parser.add_argument("-j", "--jobname", default='', help="Jobname prefix (optional)")
    parser.add_argument("--cluster", default='unity', choices=["unity", "kestrel"], help="Cluster type")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("--inlet-name", type=str, default="inlet", help="Name for the inlet sideset (default: 'inlet'. alternative: 'inflow').")
    parser.add_argument("--outlet-name", type=str, default="outlet", help="Name for the outlet sideset (default: 'outlet'. alternative: 'outflow').")
    #parser.add_argument("--block-base", type=str, default="fluid", help="Base name for the block (default: 'fluid', alternative: 'Flow').")

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
        jobname=args.jobname,
        inlet_name=args.inlet_name,
        outlet_name=args.outlet_name,
        #block_base=args.block_base,
    )

if __name__ == "__main__":
    nalu_aseq()
    #test_yaml = "input.yaml"
    #nalu_prepare_restart(test_yaml, it=None, output_file=None, batch_file=None, cluster='unity', nrun=None)
