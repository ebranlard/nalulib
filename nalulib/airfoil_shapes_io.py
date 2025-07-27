import os
from tabnanny import verbose
import numpy as np
import pandas as pd
import argparse

import nalulib.pyplot as plt # wrap matplotlib
from nalulib.essentials import *
#from nalulib.weio.csv_file import CSVFile, find_non_numeric_header_lines, WrongFormatError
#from nalulib.weio.plot3d_file import Plot3DFile, read_plot3d, write_plot3d
from nalulib.weio.airfoil_file import read_airfoil, write_airfoil, EXT_TO_FORMAT, FORMAT_TO_EXT


def convert_airfoil(input_file, output_file=None, out_format=None, verbose=False, thick=False, standardize=False, overwrite_allowed=False, plot=False):
    """"""
    from nalulib.airfoillib import standardize_airfoil_coords
    from nalulib.airfoillib import plot_airfoil
    # Determine output file and format
    if output_file is None and out_format is None:
        if standardize:
            if overwrite_allowed:
                output_file=input_file
            else:
                base, ext = os.path.splitext(input_file)
                output_file = base +  '_std' + ext
            print('Output file:', output_file)
        else:
            raise Exception("Error: You must provide either --output or --format.")

    if output_file is None:
        # Replace extension with format
        base, _ = os.path.splitext(input_file)
        out_format = out_format.lower()
        if out_format not in FORMAT_TO_EXT.keys():
            raise ValueError(f"Unsupported output format: {out_format}. Supported formats are: {', '.join(FORMAT_TO_EXT.keys())}.")
        output_file = base + '.' + FORMAT_TO_EXT[out_format]

    if out_format is None:
        ext = os.path.splitext(output_file)[1].strip('.').lower()
        if ext not in EXT_TO_FORMAT.keys():
            raise ValueError(f"Unsupported output file extension: {ext}. Supported extensions are: {', '.join(EXT_TO_FORMAT.keys())}.")
        out_format = EXT_TO_FORMAT[ext]

    if verbose:
        print(f"Input file : {input_file}")
        print(f"Output file: {output_file}")
        print(f"Format     : {out_format}")

    # Read and write
    x, y, d = read_airfoil(input_file)

    if standardize:
        if verbose:
            print("Standardizing airfoil coordinates: looping, anticlockwise, starting from upper TE.")
        x, y = standardize_airfoil_coords(x, y, verbose=verbose)

    if plot:
        plot_airfoil(x, y)
        plt.show()
    
    if output_file == input_file and not overwrite_allowed:
        print("[WARN] Output file is the same as input file, not overwriting it.")
        base, ext = os.path.splitext(output_file)
        output_file =base +  '_output' + ext
        print('New output file:', output_file)

    write_airfoil(x, y, output_file, format=out_format, thick=thick)
    if verbose:
        print("Conversion complete.")


def airfoil_plot(input_file, standardize=False, verbose=False, simple=False):
    from nalulib.airfoillib import standardize_airfoil_coords
    from nalulib.airfoillib import plot_airfoil
    if verbose:
        print('Reading: ', input_file)
    x, y, d = read_airfoil(input_file, verbose=verbose)
    if standardize:
        print("Standardizing airfoil coordinates: looping, anticlockwise, starting from upper TE.")
        x, y = standardize_airfoil_coords(x, y, verbose=verbose)
    plot_airfoil(x, y, verbose=verbose, simple=simple)
    plt.show()

def airfoil_plot_CLI():
    parser = argparse.ArgumentParser(description="Plot airfoil files for any supported formats (csv, plot3d, pointwise, geo, etc).")
    parser.add_argument(      'input', type=str, help='Input airfoil file path')
    parser.add_argument(      '--std' , action='store_true', help='Standardize the format so the coordinates are: looped, anticlockwise, and starting from the upper TE.')
    parser.add_argument(      '--simple', action='store_true', help='Plot the airfoil without trying to split surface if standardized.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    args = parser.parse_args()
    airfoil_plot(args.input, standardize=args.std, verbose=args.verbose, simple=args.simple)


def convert_airfoil_CLI():

    parser = argparse.ArgumentParser(description="Convert airfoil files between formats (csv, plot3d, pointwise, geo, etc).")
    parser.add_argument('-i', '--input', required=True, help='Input airfoil file path')
    parser.add_argument('-o', '--output', required=False, help='Output file path. If not provided, the output file will be derived from the input file name and format.')
    parser.add_argument('-f', '--format', required=False, help='Output file format (csv [csv], plot3d [fmt, xyz], pointwise [pw], geo). If not provided, the output file format will be derived from the output file extension.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument(      '--std' , action='store_true', help='Standardize the format so the coorniates are: looped, anticlockwise, and starting from the upper TE.')
    parser.add_argument(      '--plot' , action='store_true', help='Plot the airfoil shape after reading (and standardizing if --std is used).')
    parser.add_argument(      '--thick' , action='store_true', help='Enable spanwise output for plot3d format.')

    args = parser.parse_args()

    convert_airfoil(args.input, output_file=args.output, out_format=args.format, verbose=args.verbose, thick=args.thick, standardize=args.std, plot=args.plot, overwrite_allowed=False)


if __name__ == "__main__":
    filename = r"C:/Work/Student_projects/CFD_airfoil/nalu-cases/airfoils/du00-w-212_re3M.csv"
    airfoil_plot(filename)
    #import plotext as _plt
    #_plt.plot([0,1], [0,1], color='red')
    #_plt.show()

#     x, y, d = read_airfoil(filename)
# 
#     fullbase, ext = os.path.splitext(filename)
#     base = os.path.basename(fullbase)
# 
#     filename_out = os.path.join(os.path.dirname(filename), '_'+base + '.fmt')
# 
#     write_airfoil_plot3d(x, y, filename_out, thick=True)
