import sys, json;
import argparse
import numpy as np
import os
from nalulib.plot3D_plot3D2exo import plt3d2exo
from nalulib.weio.plot3d_file import read_plot3d
from nalulib.airfoil_shapes_io import convert_airfoil



_DEFAULT_OPTIONS = {
    # ---    Input Parameters
    "fileType": "PLOT3D",
    "unattachedEdgesAreSymmetry": False,
    "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
    "outerFaceBC": "farfield",
    "families": "wall",
    "autoConnect": True,
    # --- Grid Parameters
    "N": 129,
    "s0": 1e-6,
    "marchDist": 100.0,
    # --- Pseudo Grid Parameters
    "ps0": -1.0,
    "pGridRatio": -1.0,
    "cMax": 3.0,
    # --- Smoothing parameters
    "epsE": 1.0,
    "epsI": 2.0,
    "theta": 3.0,
#     "volCoef": 0.25, # Coefficient used in point-Jacobi local volume smoothing algorithm. The value should be between 0 and 1. Larger values will result in a more uniform cell volume distribution.
#     "volBlend": 0.0001, # The global volume blending coefficient. This value will typically be very small, especially if you have widely varying cell sizes.
#     "volSmoothIter": 100, #The number of point-Jacobi local volume smoothing iterations to perform at each level. More iterations will result in a more uniform cell volume distribution.
    "volCoef": [[0, 0.25], [0.1, 0.5], [1.0, 0.8]], 
    "volBlend": [[0, 1e-7], [0.1, 1e-6], [1.0, 1e-3]],
    "volSmoothIter": [[0, 0], [0.1, 500], [1.0, 1000]],
    # ---------------------------
    #   Solution parameters
    # ---------------------------
    "kspRelTol": 1e-10,
    "kspMaxIts": 1500,
    "kspSubspaceSize": 50,

}

def pyhyp_run(options, output_file_fmt=None):
    from pyhyp import pyHyp
    hyp = pyHyp(options=options)
    hyp.run()
    if output_file_fmt is not None:
        hyp.writePlot3D(output_file_fmt);    
    return hyp # todo return coords & connectivity


def get_s0(reynolds, yplus=1, nu=1.46071e-5):
    """
    Compute the first layer height based on Reynolds number and desired y+ using a turbulent flat plate approximation
    """
    U_inf = reynolds * nu
    Cf = 0.027 / reynolds ** (1 / 7)
    u_tau = np.sqrt(0.5 * Cf * U_inf**2)
    s0 = yplus * nu / u_tau

    return s0



def pyhyp_write_pyscript(options, filename=None, output_file=None):
    # use a temporary file if no filename is provided
    if filename is None:
        import tempfile
        filename = tempfile.NamedTemporaryFile(delete=False, suffix='.py').name
        print('>>> temp file', filename)
    
    with open(filename, 'w') as f:
        # make sure filename is unicdoe escaped
        filename2 = filename.encode('unicode_escape').decode('utf-8')
        f.write("#!/usr/bin/env python\n")
        f.write("import sys, json\n")
        f.write("import pyhyp\n")
        f.write("from pyhyp import pyHyp\n")
        f.write("import numpy as np\n") # just in case we get some np.float
        f.write("print('pyhypScript: lib    :', pyhyp.__file__)\n")
        f.write("print('pyhypScript: script : {}')\n".format(filename2))
        f.write("options = {}\n")
        for key, value in options.items():
            f.write(f"options['{key}'] = {repr(value)}\n")

        f.write("for key, value in options.items():\n")
        f.write("    print('pyhypScript: options: ', key, value)\n")

        f.write("hyp = pyHyp(options=options)\n")
        f.write("hyp.run()\n")
        if output_file is not None:
            output_file = output_file.encode('unicode_escape').decode('utf-8')
            #print("Output file: {}".format(output_file))
            f.write("hyp.writePlot3D('{}')\n".format(output_file))
            f.write("print('pyhypScript: output : {}')\n".format(output_file))

    return filename

def winpath2wsl(path):
    """Convert a Windows path to a WSL path."""
    if not path:
        return path
    if path[1:2] == ':':
        letter = path[0:1].lower()
        path = '/mnt/' + letter + path[2:]
    return path.replace('\\', '/')

def pyhyp_run_WSL(options, output_file_fmt='output.fmt', input_file=None, verbose=False):
    import subprocess
    import json
    import platform

    # --- Input sanitization
    if options is None:
        options = _DEFAULT_OPTIONS.copy()
    if os.path.exists(output_file_fmt):
        os.remove(output_file_fmt)
    if input_file is not None:
        options['inputFile'] = input_file
    if 'inputFile' not in options.keys(): 
        raise Exception('inputFile need to be in options')
    if not os.path.exists(options['inputFile']):
        raise FileNotFoundError("Input file not found: {}".format( options['inputFile']))
    
    input_file_win = options['inputFile']
    options['inputFile'] = winpath2wsl(os.path.abspath(options['inputFile']))
    if verbose:
        print('>>> inputFile', options['inputFile'])

    # --- Writing python script file wrapping pyhyp
    script_file = None
    #script_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_TEMP.py')
    script_file = os.path.join(os.path.dirname(os.path.abspath(input_file_win)), '_TEMP.py')
    if verbose:
        print('>>> script_file', script_file)

    output_file_wsl = winpath2wsl(os.path.abspath(output_file_fmt))

    script_file = pyhyp_write_pyscript(options, filename=script_file, output_file=output_file_wsl)
    script_file_wsl = winpath2wsl(script_file)

    # --- Setting up WSL command
    PYCMD = "python {}".format(script_file_wsl)
    bash_script = (
"source ~/libs/pyenv/bin/activate && "
"export CGNS_HOME=\\${HOME}/libs/CGNS/opt/ && "
"export PATH=$PATH:\\$CGNS_HOME/bin && "
"export LD_LIBRARY_PATH=\\$LD_LIBRARY_PATH:\\$CGNS_HOME/lib && "
"echo \\$PATH && "
"echo \\$LD_LIBRARY_PATH &&"
"source ~/.bashrc && "
"" + PYCMD)

    # --- Running WSL command
    wsl_command = ['wsl', 'bash', '-c', bash_script]
    print('>>>> Running WSL')
    if verbose:
        print('>>>> WSL command: ', wsl_command)
        print('   ')
    process = subprocess.Popen(
        wsl_command,
        stdin=subprocess.PIPE,
        stdout=None,   # Inherit parent's stdout (prints live)
        stderr=None,   # Inherit parent's stderr (prints live)
    )
    process.wait()  # Wait for the process to complete
    if verbose:
        print('   ')
        print('>>> WSL DONE')

    # --- Finalization
    if not os.path.exists(output_file_fmt):
        raise Exception('>>> Output file not generated', output_file_fmt)
    if script_file is not None and os.path.exists(script_file):
        os.remove(script_file)
        print('>>> Removed temporary script file:', script_file)    


def pyhyp_cmdline_CLI():

    parser = argparse.ArgumentParser(description="Run pyHyp with input and output files.")
    parser.add_argument('-i', '--input', required=True, help='Input file path')
    parser.add_argument('-o', '--output', required=False, help='Output file path', default='_OUTPUT.exo')
    parser.add_argument('-n', '--N', type=int, help='Grid parameter N')
    parser.add_argument('--re', type=float, default=None, help='For s0 - Reynolds number used to compute s0 if provided')
    parser.add_argument("--nu", type=float, default=1.46071e-5, help='For s0 - Kinematic viscosity (nu). Only used if Re is provided')
    parser.add_argument("--yplus", type=float, default=1, help='For s0 - Yplus value, only used if Re is provided')
    parser.add_argument('--s0', type=float, help='Grid parameter s0 (not used if Re is provided)')
    parser.add_argument('--marchDist', type=float, help='Grid parameter marchDist', default=_DEFAULT_OPTIONS['marchDist'])
    parser.add_argument('--cMax', type=float, help='Pseudo grid parameter cMax', default=_DEFAULT_OPTIONS['cMax'])
    parser.add_argument('--epsE', type=float, help='Smoothing parameter epsE', default=_DEFAULT_OPTIONS['epsE'])
    parser.add_argument('--epsI', type=float, help='Smoothing parameter epsI', default=_DEFAULT_OPTIONS['epsI'])
    parser.add_argument('--theta', type=float, help='Smoothing parameter theta', default=_DEFAULT_OPTIONS['theta'])
    parser.add_argument('--volCoef', type=float, help='Smoothing parameter volCoef')
    parser.add_argument('--volBlend', type=float, help='Smoothing parameter volBlend')
    parser.add_argument('--volSmoothIter', type=int, help='Smoothing parameter volSmoothIter')
    parser.add_argument("--no-zpos-check", action="store_true", help="Do not check and enforce the orientation of quads about Z axis (default: check).")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not delete intermediary files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    parser.add_argument("--inlet-name", type=str, default="inlet", help="Name for the inlet sideset (default: 'inlet'. alternative: 'inflow').")
    parser.add_argument("--outlet-name", type=str, default="outlet", help="Name for the outlet sideset (default: 'outlet'. alternative: 'outflow').")
    parser.add_argument("--block-base", type=str, default="fluid", help="Base name for the block (default: 'fluid', alternative: 'Flow').")

    args = parser.parse_args()

    # Prepare options dictionary
    options = _DEFAULT_OPTIONS.copy()
    if args.N is not None:
        options['N'] = args.N
    if args.re is not None:
        s0 = get_s0(reynolds=args.re, yplus=args.yplus, nu=args.nu)
        options['s0'] = float(s0) # we don't want np.float
        print('S0 computed from Reynolds number as:', s0, '(Re={}, yplus={}, nu={})'.format(args.re, args.yplus, args.nu))
    else:
        if args.s0 is not None:
            options['s0'] = args.s0

    # Inline update of options from args
    for key in [
        'marchDist', 'ps0', 'pGridRatio', 'cMax', 'epsE', 'epsI', 'theta', 'volCoef', 'volBlend', 'volSmoothIter']:
        val = getattr(args, key, None)
        if val is not None:
            options[key] = val

    # Prepare data dictionary
    verbose = args.verbose

    if verbose:
        print('Arguments received:')
        print(f"       Input file      : {args.input}")  
        print(f"       Output file     : {args.output}")
        print(f"       Grid parameter N: {options['N']}")
        print(f"       reynolds        : {args.re}")
        print(f"       s0              : {options.get('s0')}")
        print(f"       marchDist       : {options.get('marchDist')}")
        print(f"       ps0             : {options.get('ps0')}")
        print(f"       pGridRatio      : {options.get('pGridRatio')}")
        print(f"       cMax            : {options.get('cMax')}")
        print(f"       epsE            : {options.get('epsE')}")
        print(f"       epsI            : {options.get('epsI')}")
        print(f"       theta           : {options.get('theta')}")
        print(f"       volCoef         : {options.get('volCoef')}")
        print(f"       volBlend        : {options.get('volBlend')}")
        print(f"       volSmoothIter   : {options.get('volSmoothIter')}")
    
    output_file_fmt = args.output+'_fmt'
    output_file_exo = args.output

    # --- Convert airfoil if needed
    try:
        coords, dims = read_plot3d(args.input)
        if not np.array_equal(dims[0][1:], [2, 1]):
            raise ValueError(f"Unexpected dimensions for {args.input}: {dims}")
        print('[ OK ] Input file is in PLOT3D format.')
        input_file = args.input
        input_file_tmp = None
    except:
        print('[INFO] Input file is not in PLOT3D format, converting it')
        input_file_tmp = args.input+'_temp.fmt'
        convert_airfoil(args.input, input_file_tmp, out_format='plot3d', thick=True)
        input_file = input_file_tmp
    options['inputFile'] = input_file


    # --- Detect if pyhyp can be imported
    print('############################## PYHYP ####################################')
    print('Options:')
    for k,v in options.items():
        print('{:27s}: {}'.format(k,v))
    try:
        from pyhyp import pyHyp
        # --- Run pyhyp
        hyp = pyhyp_run(options, output_file_fmt=output_file_fmt)

    except ImportError:
        # See if we are on windows, if so, we'll call a function that writes a pyhyp script for wsl
        if sys.platform.startswith('win'):
            pyhyp_run_WSL(options, output_file_fmt=output_file_fmt, verbose=args.verbose)
        else:
            print("pyHyp is not installed. Please install it before running this script.")
            sys.exit(1) 
    print('########################## END PYHYP ####################################')

    # --- Check if plot3d file was generated
    if not os.path.exists(output_file_fmt):
        raise Exception('>>> Plot3d file not generated', output_file_fmt)
    else:
        if verbose:
            print('[INFO] Fmt file generated:', output_file_fmt)
    
    # --- Generate exodus file
    check_zpos = not args.no_zpos_check

    plt3d2exo(output_file_fmt, output_file_exo, flatten=True, check_zpos=check_zpos,
        inlet_name=args.inlet_name,
        outlet_name=args.outlet_name,
        block_base=args.block_base,
        final_print=False
        )

    if not os.path.exists(output_file_exo):
        raise Exception('>>> Output file not generated:', output_file_exo)
    print('[INFO] Exo file generated:', output_file_exo)

    # --- Cleanup
    if not args.no_cleanup:
        if os.path.exists(output_file_fmt):
            os.remove(output_file_fmt)
            if verbose:
                print('>>> Removed temporary file:', output_file_fmt)    
        if input_file_tmp is not None and os.path.exists(input_file_tmp):
            os.remove(input_file_tmp)
            if verbose:
                print('>>> Removed temporary input file:', input_file_tmp)
    print(f"[INFO] s0 used: {options.get('s0')}")










if __name__ == "__main__":
    pyhyp_cmdline_CLI()

    #options = _DEFAULT_OPTIONS.copy()
    #options['inputFile'] = '../_examples_big/naca0012_euler.fmt'
    #pyhyp_run_WSL(options, output_file='_output.fmt', verbose=True)




