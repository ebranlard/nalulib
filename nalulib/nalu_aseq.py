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
from nalulib.exodus_rotate import exo_rotate



def nalu_aseq(input_file, aseq=None, 
              verbose=False, debug=False, 
              mesh_file=None,
              sim_dir=None,
              batch_file=None, 
              cluster=None, 
              aoa_ori=None, 
              inlet_name='inlet', outlet_name='outlet', submit=False, center=None, keep_io_side_set=False, raiseError=True, 
              jobname='', hours=None, nodes=None, ntasks=None, mem=None,
              ):
    myprint('Input YAML file', input_file)
    # Basename
    base_dir = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    base, ext = os.path.splitext(base_name)
    if sim_dir is not None:
        base_dir = sim_dir

    if aseq is None:
        aseq = np.arange(-15, 15+3/2, 3)
    else:
        if len(aseq) == 3:
            aseq = np.arange(aseq[0], aseq[1]+aseq[2]/2, aseq[2])
        else:
            pass
    myprint('AoA sequence', aseq)


    # --- Read YAML file
    yml_ori = NALUInputFile(input_file, reader='ruamel') # NOTE: using ruamel to keep comments
    yml = yml_ori.copy()
    realms_ori = yml_ori.data['realms']
    realms = yml.data['realms']
    if len(realms) > 1:
        print('[INFO] Multiple realms found, using the second one')
        realm  = realms[1]
    else:
        realm  = realms[0]

    if mesh_file is None:
        mesh_ori = realm['mesh']
    else:
        realms_ori[-1]['mesh'] = mesh_file
        mesh_ori = mesh_file

    mesh_ori_abs = os.path.join(os.path.dirname(input_file), mesh_ori)
    myprint('Mesh abs', mesh_ori_abs)
    myprint('Mesh    ', mesh_ori)

    # --- Check if the input file is valid
    yml_ori.check(verbose=verbose, raiseError=raiseError)

    if aoa_ori is None:
        aoa_ori = extract_aoa(mesh_ori)
    if aoa_ori is None:
        aoa_ori=0
        print('[INFO] No AoA found in mesh name, assuming 0. Change it with --aoa_ori.')
    
    # --- Mesh creation
    print('----------------------------------------------------------------------------')
    mesh_dir = os.path.join(base_dir, 'meshes')
    myprint('Creating meshes in : ', mesh_dir)
    # creating a directory "meshes" if it does not exist
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    angles = aseq - aoa_ori
    mesh_files = [os.path.join(mesh_dir, base+'_mesh_aoa{:04.1f}.exo'.format(alpha)) for alpha in aseq]
    # 
    exo_rotate(input_file=mesh_ori_abs, output_file=mesh_files, angle=angles, center=center, verbose=verbose, inlet_name=inlet_name, outlet_name=outlet_name, keep_io_side_set=keep_io_side_set)
    print('----------------------------------------------------------------------------')


    # --- Create YAML files
    nalu_files = [os.path.join(base_dir, base+'_aoa{:04.1f}.yaml'.format(alpha)) for alpha in aseq]
        
    for ia, (alpha, nalu_file, mesh_file) in enumerate(zip(aseq, nalu_files, mesh_files)):
        sAOA = '_aoa{:04.1f}'.format(alpha)
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
        if ia in [0, len(aseq)-1]:
            myprint('Written NALU File', nalu_file)
        print('   ...') if ia==1 else None
    print('----------------------------------------------------------------------------')
    batch_files = []

    # --- Create BATCH files
    if cluster == 'local':
        base_dir = os.path.dirname(input_file)
        base_name = os.path.basename(input_file)
        base, ext = os.path.splitext(base_name)
        run_batch_file = os.path.join(base_dir, 'submit-'+base+'.sh') 
        with open(run_batch_file, 'w', encoding='utf-8') as f:
            for nalu_file in nalu_files:
                f.write('naluX -i {}\n'.format(nalu_file))
        myprint('[INFO] Commands written to:', run_batch_file)
    else:
        batch_files = []
        for ia, (alpha, nalu_file) in enumerate(zip(aseq, nalu_files)):
            jobname_case = jobname + 'A{:04.1f}'.format(alpha)
            new_batch = nalu_batch(batch_file_template=batch_file, nalu_input_file=nalu_file, cluster=cluster, verbose=verbose, jobname=jobname_case, mail=ia==len(aseq)-1, 
                                   hours=hours, nodes=nodes, ntasks=ntasks, mem=mem,
                                   sim_dir=base_dir)
            batch_files.append(new_batch)
            if ia in [0, len(aseq)-1]:
                myprint('Written Batch File', new_batch, jobname_case)
            print('   ...') if ia==1 else None
        print('----------------------------------------------------------------------------')

        # --- Submit BATCH files
        if submit:
            cmd = 'sbatch'
            print('[INFO] Submitting {} batch files using {}'.format(len(batch_files), cmd))
            for batch_file in batch_files:
                os.system(cmd + ' ' + batch_file)
    return nalu_files, batch_files 


def nalu_aseq_CLI():
    """
    Command-line entry point for preparing a Nalu-Wind aseq
    """
    import argparse

    parser = argparse.ArgumentParser(description="Setup Nalu-Wind for multiple AoA simulations (YAML and batch files).")
    parser.add_argument("input_file", default='input.yaml', nargs="?", help="Input YAML file")
#    parser.add_argument("-o", "--output_file", default=None, help="Output YAML file (optional, default: input_runNRUN.yaml)")
    parser.add_argument("-a", type=float, nargs=3, default=None, help="Alpha sequence (start, stop, step) (optional, default: -15 to 15 in steps of 3)")
    parser.add_argument("-c", "--center", type=float, nargs=2, default=(0.25, 0.0), help="Center of rotation (x, y). Default is (0.25, 0.0).")
    parser.add_argument("-b", "--batch_file", default=None, help="Batch file template (optional)")
    parser.add_argument("--keep-io-side-set", action="store_true", help="Keep inlet and outlet side sets unchanged, otherwise, they are adapted.")
    parser.add_argument("-j", "--jobname", default='', help="Jobname prefix (optional)")
    parser.add_argument("--cluster", default='unity', choices=["unity", "kestrel", "local"], help="Cluster type. If local, only one batch file is created.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--submit"     , action="store_true",  help="Submit generated batch file to cluster scheduler (`sbatch`).")
    parser.add_argument("--no-check"   , action="store_true",  help="Do not raise errors if issues are found in the input file.")

    parser.add_argument("--inlet-name", type=str, default="inlet", help="Name for the inlet sideset (default: 'inlet'. alternative: 'inflow').")
    parser.add_argument("--outlet-name", type=str, default="outlet", help="Name for the outlet sideset (default: 'outlet'. alternative: 'outflow').")

    args = parser.parse_args() 
    if args.verbose:
        print('Arguments:', args)

    nalu_aseq(
        input_file=args.input_file,
        aseq=args.a,
#        output_file=args.output_file,
        center=tuple(args.center),
        keep_io_side_set=args.keep_io_side_set,
        verbose=args.verbose,
        batch_file=args.batch_file,
        cluster=args.cluster,
        jobname=args.jobname,
        inlet_name=args.inlet_name,
        outlet_name=args.outlet_name,
        submit=args.submit, 
        raiseError=not args.no_check
    )

if __name__ == "__main__":
    nalu_aseq()
    #test_yaml = "input.yaml"
    #nalu_prepare_restart(test_yaml, it=None, output_file=None, batch_file=None, cluster='unity', nrun=None)
