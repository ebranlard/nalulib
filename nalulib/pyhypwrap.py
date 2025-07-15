import sys, json;
import argparse
import numpy as np
import os



_DEFAULT_OPTIONS = {
    # ---    Input Parameters
    "unattachedEdgesAreSymmetry": False,
    "outerFaceBC": "farfield",
    "autoConnect": True,
    "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
    "families": "wall",
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
    "volCoef": 0.25,
    "volBlend": 0.0001,
    "volSmoothIter": 100,
}

def pyhyp_run(options, output_file=None):
    from pyhyp import pyHyp;
    hyp = pyHyp(options=options);
    hyp.run();
    if output_file is not None:
        hyp.writePlot3D(output_file);    
        print(f"Output file: {output_file}")
    return hyp # todo return coords & connectivity





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

def pyhyp_run_WSL(options, output_file='output.fmt', input_file=None, verbose=False):
    import subprocess
    import json
    import platform

    # --- Input sanitization
    if options is None:
        options = _DEFAULT_OPTIONS.copy()
    if os.path.exists(output_file):
        os.remove(output_file)
    if input_file is not None:
        options['inputFile'] = input_file
    if 'inputFile' not in options.keys(): 
        raise Exception('inputFile need to be in options')
    if not os.path.exists(options['inputFile']):
        raise FileNotFoundError("Input file not found: {}".format( options['inputFile']))
    options['inputFile'] = winpath2wsl(os.path.abspath(options['inputFile']))
    if verbose:
        print('>>> inputFile', options['inputFile'])

    # --- Writing python script file wrapping pyhyp
    script_file = None
    script_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_TEMP.py')

    output_file_wsl = winpath2wsl(os.path.abspath(output_file))

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
    if not os.path.exists(output_file):
        raise Exception('>>> Output file not generate', output_file)
    else:
        print('>>> Output file generated:', output_file)


def pyhyp_cmdline_CLI():

    parser = argparse.ArgumentParser(description="Run pyHyp with input and output files.")
    parser.add_argument('-i', '--input', required=True, help='Input file path')
    parser.add_argument('-o', '--output', required=False, help='Output file path', default='_OUTPUT.fmt')
    parser.add_argument('-n', '--N', type=int, help='Grid parameter N')
    parser.add_argument('--s0', type=float, help='Grid parameter s0')
    parser.add_argument('--marchDist', type=float, help='Grid parameter marchDist')
    parser.add_argument('--ps0', type=float, help='Pseudo grid parameter ps0')
    parser.add_argument('--pGridRatio', type=float, help='Pseudo grid parameter pGridRatio')
    parser.add_argument('--cMax', type=float, help='Pseudo grid parameter cMax')
    parser.add_argument('--epsE', type=float, help='Smoothing parameter epsE')
    parser.add_argument('--epsI', type=float, help='Smoothing parameter epsI')
    parser.add_argument('--theta', type=float, help='Smoothing parameter theta')
    parser.add_argument('--volCoef', type=float, help='Smoothing parameter volCoef')
    parser.add_argument('--volBlend', type=float, help='Smoothing parameter volBlend')
    parser.add_argument('--volSmoothIter', type=int, help='Smoothing parameter volSmoothIter')
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    # Prepare options dictionary
    options = _DEFAULT_OPTIONS.copy()
    if args.N is not None:
        options['N'] = args.N
    if args.s0 is not None:
        options['s0'] = args.s0
    if args.marchDist is not None:
        options['marchDist'] = args.marchDist
    if args.ps0 is not None:
        options['ps0'] = args.ps0
    if args.pGridRatio is not None:
        options['pGridRatio'] = args.pGridRatio
    if args.cMax is not None:
        options['cMax'] = args.cMax
    if args.epsE is not None:
        options['epsE'] = args.epsE
    if args.epsI is not None:
        options['epsI'] = args.epsI
    if args.theta is not None:
        options['theta'] = args.theta
    if args.volCoef is not None:
        options['volCoef'] = args.volCoef
    if args.volBlend is not None:
        options['volBlend'] = args.volBlend
    if args.volSmoothIter is not None:
        options['volSmoothIter'] = args.volSmoothIter

    # Prepare data dictionary
    options['inputFile'] = args.input

    if args.verbose:
        print('Arguments received:')
        print(f"Input file      : {args.input}")  
        print(f"Output file     : {args.output}")
        print(f"Grid parameter N: {options['N']}")
        print(f"s0              : {options.get('s0')}")
        print(f"marchDist       : {options.get('marchDist')}")
        print(f"ps0             : {options.get('ps0')}")
        print(f"pGridRatio      : {options.get('pGridRatio')}")
        print(f"cMax            : {options.get('cMax')}")
        print(f"epsE            : {options.get('epsE')}")
        print(f"epsI            : {options.get('epsI')}")
        print(f"theta           : {options.get('theta')}")
        print(f"volCoef         : {options.get('volCoef')}")
        print(f"volBlend        : {options.get('volBlend')}")
        print(f"volSmoothIter   : {options.get('volSmoothIter')}")

    # --- Detect if pyhyp can be imported
    print('############################## PYHYP ####################################')
    try:
        from pyhyp import pyHyp
        # --- Run pyhyp
        hyp = pyhyp_run(options, output_file=args.output)

    except ImportError:
        # See if we are on windows, if so, we'll call a function that writes a pyhyp script for wsl
        if sys.platform.startswith('win'):
            pyhyp_run_WSL(options, output_file=args.output, verbose=args.verbose)
        else:
            print("pyHyp is not installed. Please install it before running this script.")
            sys.exit(1) 
    print('########################## END PYHYP ####################################')

if __name__ == "__main__":
    pyhyp_cmdline_CLI()

    #options={}
    ##options['Hello']=3
    #options = _DEFAULT_OPTIONS.copy()
    ##options['inputFile'] = 
    #input_file='../_examples_big/naca0012_euler.fmt'




