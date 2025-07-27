import os
from tabnanny import verbose
import numpy as np
import pandas as pd
import argparse

import nalulib.pyplot as plt # wrap matplotlib
from nalulib.essentials import *
from nalulib.weio.csv_file import CSVFile, find_non_numeric_header_lines, WrongFormatError
from nalulib.weio.plot3d_file import Plot3DFile, read_plot3d, write_plot3d


FORMAT_TO_EXT = {
    'csv': 'csv',
    'plot3d': 'fmt',
    'fmt': 'fmt',
    'xyz': 'xyz',
    'xy': 'xy',
    'g': 'g',
    'pointwise': 'pwise',
    'geo': 'geo',
}
EXT_TO_FORMAT = {
    'csv': 'csv',
    'txt': 'csv',
    'dat': 'csv',
    'fmt': 'plot3d',
    'xyz': 'plot3d',
    'xy': 'plot3d',
    'g': 'plot3d',
    'pwise': 'pointwise',
    'pw':    'pointwise',
    'geo': 'geo',
}


# --------------------------------------------------------------------------------{
# --- Main wrappers
# --------------------------------------------------------------------------------}
def read_airfoil(filename, format=None, verbose=False, **kwargs):
    """ Read airfoil coordinates from a filename"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.") 
    if format is None:
        ext = os.path.splitext(filename)[1].lower().strip('.')
        format = EXT_TO_FORMAT[ext]
    format = format.lower()
    if verbose:
        print(f"Reading airfoil from {filename} with format {format}")

    if format in ['csv','tab']:
        x, y, d = read_airfoil_csv(filename)
    elif format in ['plot3d','fmt','g','xyz','xy','x']:
        x, y, d = read_airfoil_plot3d(filename)
    elif format in ['pointwise', 'pw','pwise']:
        x, y, d = read_airfoil_pointwise(filename, plot=False)
    else:
        raise  NotImplementedError(f"File type {ext} is not supported.")

    
    if not np.issubdtype(x.dtype, np.floating) or not np.issubdtype(y.dtype, np.floating):
        print('First values of x:',x[0:5], x.dtype)
        print('First values of y:',y[0:5], y.dtype)
        raise ValueError("Ffile must contain floating point numbers in both columns. Maybe the header was not detected correctly?")

    return x, y, d

def write_airfoil(x, y, filename, format=None, **kwargs):
    """ Write airfoil coordinates to a file"""
    if format is None:
        ext = os.path.splitext(filename)[1].lower().strip('.')
        format = EXT_TO_FORMAT[ext]
    format = format.lower()
    if format in ['csv','tab']:
        write_airfoil_csv(x, y, filename)
    elif format in ['plot3d','fmt','g','xyz','xy','x']:
        write_airfoil_plot3d(x, y, filename, **kwargs)
    elif format in ['pointwise', 'pw','pwise']:
        write_airfoil_pointwise(x, y, filename)
    elif format == 'geo':
        write_airfoil_geo(x, y, filename, **kwargs)
    else:
        raise NotImplementedError(f"Format {format} is not supported.")

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

# --------------------------------------------------------------------------------}
# --- CSV 
# --------------------------------------------------------------------------------{
def has_two_ints_on_second_line_and_third_empty(filename):
    """
    See for instance e850.dat

    Returns True if the second line of the file contains exactly two floats that can be coerced to integers,
    and the third line is empty (only whitespace or newline).
    """
    with open(filename, 'r', encoding='utf-8', errors='surrogateescape') as f:
        lines = []
        for _ in range(3):
            line = f.readline()
            if not line: break
            lines.append(line.rstrip('\n\r'))

    if len(lines) < 3: return False

    # Check second line
    parts = lines[1].strip().replace(',', ' ').split()
    if len(parts) != 2: return False
    try:
        floats = [float(p) for p in parts]
        ints   = [int(f) for f in floats]
        if not all(abs(f-i)<1e-8 for f,i in zip(floats,ints)):
            return False
    except Exception:
        return False

    # Check third line is empty
    if lines[2].strip() != '':
        return False

    return True

def read_airfoil_csv(filename):
    # --- Find non-numeric header lines
    header_indices, header_lines = find_non_numeric_header_lines(filename)

    # We expect mostly 0 or one line of header. If more than one header line is found, it's best if lines start with a '#'
    if len(header_indices)> 1:
        print("[INFO] Found more than one header line in file ", filename)
    #    # count lines that do not start with "#"
    #    not_comment_lines = [line for line in header_lines if not line.startswith('#')]
    #    nNotComment = len(not_comment_lines)
    #    print("[INFO] Found non-comment header lines:", not_comment_lines)
    #    #if nNotComment > 0:
    #    #    raise Exception("Error: More than one header line found in the file. Please ensure the file has a single header line, no header lines at all, or that all header lines start with `#`.")


    if has_two_ints_on_second_line_and_third_empty(filename):
        raise Exception('File format with separate Upper and Lower surfaces not yet supported, file {}.'.format(filename))

    #print('>>>> commentLines:', header_indices, 'header_lines:', header_lines)
    #csv = CSVFile(filename=filename, commentLines=header_indices, detectColumnNames=False, colNames=['x', 'y'], doRead=False)
    csv = CSVFile(filename=filename, commentLines=header_indices, doRead=False) #, detectColumnNames=False, colNames=['x', 'y'], doRead=False)
    try:
        csv._read()
    except WrongFormatError as e:
        print("[FAIL] {}".format(str(e).strip()))
        print("       > Trying to read the file with a slower method...")
        #print(csv)
        csv.read_slow_stop_at_first_empty_lines(numeric_only=True)

    df = csv.toDataFrame()
    #import pandas as pd
    #df = pd.read_csv(filename)
    #print(df)
    if df.shape[1] == 2:
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
    else:
        raise ValueError("CSV file must have exactly two columns for x and y coordinates.")
    # Check if numpy array are all floats, otherwise( e.g. if they are objects) raise an exception
    if not np.issubdtype(x.dtype, np.floating) or not np.issubdtype(y.dtype, np.floating):
        if x[-1] == 'ZZ':
            print('[WARN] File {} Last value of x is "ZZ", removing it and converting to float.'.format(filename))
            x=x[:-1].astype(float)
            y=y[:-1]
        else:
            print(csv)
            print('First values of x:',x[0:5], x.dtype)
            print('First values of y:',y[0:5], y.dtype)
            raise ValueError("CSV file must contain floating point numbers in both columns. Maybe the header was not detected correctly?")
    d={}
    return x, y, d

def write_airfoil_csv(x, y, filename):
    df = pd.DataFrame({'x': x, 'y': y})
    df.to_csv(filename, index=False)

# --------------------------------------------------------------------------------}
# --- Plot3D
# --------------------------------------------------------------------------------{
def read_airfoil_plot3d(filename):
    coords, dims = read_plot3d(filename, singleblock=True)
    x = coords[:, 0]
    y = coords[:, 1]
    # Make sure we keep only the first slice in z-direction
    nx = dims[0]
    x = x[:nx]
    y = y[:nx]
    d = {} 
    return x, y, d

# --------------------------------------------------------------------------------}
# --- Pointwise
# --------------------------------------------------------------------------------{
def read_airfoil_pointwise(filename, plot=False, verbose=False):
    # TODO this is horrible code, needs to be refactored
    lower = []
    upper = []
    TE = []
    d= {}

    with open(filename, 'r') as file:
        # Read the entire content of the file
        lines = file.readlines()

        current_section = 'lower'  # Starting with the lower section
        idx = 0  # Line index

        while idx < len(lines):
            line = lines[idx].strip()

            if line.isdigit():  # When the line is a number (point count)
                num_points = int(line)  # Get the number of points in the section
                idx += 1  # Move to the next line containing the coordinates

                # Read the next `num_points` lines and store x, y, z coordinates
                for _ in range(num_points):
                    if idx < len(lines):
                        x, y, z = map(float, lines[idx].strip().split())  # Parse x, y, z values
                        if current_section == 'lower':
                            lower.append((x, y, z))  # Append to the lower section
                        elif current_section == 'upper':
                            upper.append((x, y, z))  # Append to the upper section
                        elif current_section == 'TE':
                            TE.append((x, y, z))  # Append to the TE section
                        idx += 1  # Move to the next line containing coordinates

                # Switch sections after processing each part
                if current_section == 'lower':
                    current_section = 'upper'
                elif current_section == 'upper':
                    current_section = 'TE'
            
            else:
                idx += 1  # Skip lines that are not point counts or coordinates
    TE    = np.asarray(TE)[:,:2]# Keep only x and y coordinates
    lower = np.asarray(lower)[:,:2] 
    upper = np.asarray(upper)[:,:2]

    from nalulib.curves import contour_is_clockwise
    coords1 = np.vstack((lower[:-1], upper[:-1], TE))
    assert contour_is_clockwise(coords1), "Pointwise format is expected to be clockwise."
    assert np.allclose(coords1[0, :], coords1[-1, :], rtol=1e-10, atol=1e-12), "First and last points must be the same in Pointwise format."

    # NOTE: Pointwise is assumed to be clockwise
    TE = TE[::-1]  # Reverse the order of TE points to match the convention
    lower = lower[::-1]  # Reverse the order of lower surface points        
    upper = upper[::-1]  # Reverse the order of upper surface points

    # NOTE: coords are anticlockwise with first and last point being the same
    coords = np.vstack((upper[:-1], lower[:-1], TE))
    assert np.allclose(coords[0, :], coords[-1, :], rtol=1e-10, atol=1e-12), "First and last points must be the same in Pointwise format."

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(coords[:,0], coords[:,1], '.-', label='Airfoil Shape', color='black')
        plt.plot(lower[:,0], lower[:,1], label='Lower Surface', color='blue')
        plt.plot(upper[:,0], upper[:,1], label='Upper Surface', color='red')
        plt.plot(TE[:,0], TE[:,1], label='Trailing Edge', color='green') 
        plt.title('Airfoil Shape with Upper and Lower Surfaces')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()

    return coords[:,0], coords[:,1], d 

def write_airfoil_pointwise(x, y, output_file):
    # === Load airfoil data ===
    x_orig, y_orig = x, y

    # === Find leading edge index (minimum x)
    le_index = np.argmin(x_orig)

    # === convert to .txt file format for Pointwise
    
    # === 1. Split into upper and lower surfaces
    x_orig_upper, y_orig_upper = x_orig[:le_index+1], y_orig[:le_index+1]
    x_orig_lower, y_orig_lower = x_orig[le_index:], y_orig[le_index:]

    # === 2. Sort both surfaces to save as .dat file without interpolation
    x_orig_lower_sorted, y_orig_lower_sorted = x_orig_lower[::-1], y_orig_lower[::-1]
    x_orig_upper_sorted, y_orig_upper_sorted =  x_orig_upper[::-1], y_orig_upper[::-1]

    
    # === 3. Save original data in .dat format ===
    with open(output_file, 'w') as f:
        # Write lower surface
        f.write(f"{len(x_orig_lower_sorted)}\n")
        for x, y in zip(x_orig_lower_sorted, y_orig_lower_sorted):
            f.write(f"{x:.6f} {y:.6f} 0.000000\n")

        # Write upper surface
        f.write(f"{len(x_orig_upper_sorted)}\n")
        for x, y in zip(x_orig_upper_sorted, y_orig_upper_sorted):
            f.write(f"{x:.6f} {y:.6f} 0.000000\n")

        # Write TE surface
        f.write(f"{3}\n")
        f.write(f"{x_orig_upper_sorted[-1]:.6f} {y_orig_upper_sorted[-1]:.6f} 0.000000\n")
        f.write(f"{((x_orig_upper_sorted[-1]+x_orig_lower_sorted[0])/2):.6f} {((y_orig_lower_sorted[0]+y_orig_upper_sorted[-1])/2):.6f} 0.000000\n")
        f.write(f"{x_orig_lower_sorted[0]:.6f} {y_orig_lower_sorted[0]:.6f} 0.000000\n")


# --------------------------------------------------------------------------------}
# --- gmesh 
# --------------------------------------------------------------------------------{
def write_airfoil_geo(x, y, output_file, lc=1.0):
    with open(output_file, 'w') as f_out:
        f_out.write("// Gmsh .geo file generated from 2D airfoil .txt\n\n")
        zz=0
        for xx, yy in enumerate(zip(x, y)):
            f_out.write(f"Point({i}) = {{{xx}, {yy}, {zz}, {lc}}};\n")

        # Connect all points in a closed loop
        f_out.write("\n// Single Line connecting all points in a loop\n")
        line_str = ", ".join(str(pid) for pid in point_ids + [point_ids[0]])
        f_out.write(f"Line(1) = {{{line_str}}};\n")


def write_airfoil_plot3d(x, y, filename, thick=False):
    """ Write airfoil coordinates to a Plot3D file"""
    if thick:
        # We duplicate the x y coordiantes and have z=0 and z=1
        coords = np.column_stack((x, y, np.zeros_like(x)))
        coords = np.concatenate((coords, coords + np.array([0, 0, 1])))
        dims = (len(x), 2, 1)  # Two slices in the z-direction
    else:
        coords = np.column_stack((x, y, np.zeros_like(x)))  # Assuming z=0 for 2D airfoil
        dims = (len(x), 1, 1)  # Assuming a single slice in the z-direction
    write_plot3d(filename, coords, dims)

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
