import os
import numpy as np
import pandas as pd
from nalulib.essentials import *
from nalulib.weio.csv_file import CSVFile
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
def read_airfoil(filename, format=None, **kwargs):
    """ Read airfoil coordinates from a filename"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.") 
    if format is None:
        _, ext = os.path.splitext(filename)
        format = ext.strip('.')
    format = format.lower()

    if format in ['csv','tab']:
        x, y, d = read_airfoil_csv(filename)
    elif format in ['plot3d','fmt','g','xyz','xy','x']:
        x, y, d = read_airfoil_plot3d(filename)
    elif format in ['pointwise', 'pw','pwise']:
        x, y, d = read_airfoil_pointwise(filename, plot=True)
    else:
        raise  NotImplementedError(f"File type {ext} is not supported.")
    d['filename'] = filename
    d['format'] = format
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

def convert_airfoil(input_file, output_file=None, out_format=None, verbose=False, thick=False):
    """"""
    # Determine output file and format
    if output_file is None and out_format is None:
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
    from nalulib.airfoil_shapes_io import read_airfoil, write_airfoil
    x, y, d = read_airfoil(input_file)
    write_airfoil(x, y, output_file, format=out_format, thick=thick)
    if verbose:
        print("Conversion complete.")

# --------------------------------------------------------------------------------}
# --- CSV 
# --------------------------------------------------------------------------------{
def read_airfoil_csv(filename):
    csv = CSVFile(filename)
    df = csv.toDataFrame()
    #import pandas as pd
    #df = pd.read_csv(filename)
    #print(df)
    if df.shape[1] == 2:
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
    else:
        raise ValueError("CSV file must have exactly two columns for x and y coordinates.")
    d={}
    return x, y, d

def write_airfoil_csv(x, y, filename):
    df = pd.DataFrame({'x': x, 'y': y})
    df.to_csv(filename, index=False)

# --------------------------------------------------------------------------------}
# --- Plot3D
# --------------------------------------------------------------------------------{
def read_airfoil_plot3d(filename):

    coords, dims = read_plot3d(filename)
    x= coords[:, 0]
    y = coords[:, 1]
    d = {} 
    return x, y, d


# --------------------------------------------------------------------------------}
# --- Pointwise
# --------------------------------------------------------------------------------{
def read_airfoil_pointwise(filename, plot=False, verbose=False):
    lower = []
    upper = []
    TE = []
    section_points = []

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
    lower=lower[1:-1]  # Remove the last point to avoid duplication
    upper=upper[:-1]  # Remove the last point to avoid duplication
        
    lower_x,lower_y = [coord[0] for coord in lower],[coord[1] for coord in lower]
    upper_x,upper_y = [coord[0] for coord in upper],[coord[1] for coord in upper]
    TE_x, TE_y = [coord[0] for coord in TE],[coord[1] for coord in TE]

    ILower = np.arange(len(lower_x))
    IUpper = np.arange(len(upper_x))+len(lower_x)
    ITE = np.arange(len(TE_x))+len(lower_x)+len(upper_x)
    x = np.concatenate((lower_x, upper_x[:], TE_x[:]))
    y = np.concatenate((lower_y, upper_y[:], TE_y[:]))
    if verbose:
        print('TE_x', TE_x)
        print('TE_y', TE_y)


    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, '.-', label='Airfoil Shape', color='black')
        plt.plot(lower_x, lower_y, label='Lower Surface', color='blue')
        plt.plot(upper_x, upper_y, label='Upper Surface', color='red')
        plt.plot(TE_x, TE_y, label='Trailing Edge', color='green') 
        plt.title('Airfoil Shape with Upper and Lower Surfaces')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)

    d = {}
    #d['lower'] = lower
    #d['upper'] = upper
    #d['TE'] = TE
    d['ILower'] = ILower
    d['IUpper'] = IUpper
    d['ITE']    = ITE

    #return lower_x, lower_y, upper_x, upper_y, TE_x, TE_y
    return x, y, d 

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



def convert_airfoil_API():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Convert airfoil files between formats (csv, plot3d, pointwise, geo, etc).")
    parser.add_argument('-i', '--input', required=True, help='Input airfoil file path')
    parser.add_argument('-o', '--output', required=False, help='Output file path. If not provided, the output file will be derived from the input file name and format.')
    parser.add_argument('-f', '--format', required=False, help='Output file format (csv [csv], plot3d [fmt, xyz], pointwise [pw], geo). If not provided, the output file format will be derived from the output file extension.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument(      '--thick' , action='store_true', help='Enable spanwise output for plot3d format.')

    args = parser.parse_args()

    if args.output is None and args.format is None:
        print("Error: You must provide --output and/or --format.")
        sys.exit(1)
    
    convert_airfoil(args.input, output_file=args.output, out_format=args.format, verbose=args.verbose, thick=args.thick)


if __name__ == "__main__":
    filename = r"C:\Work\UAA\UAlib\ua-tuning\experiments\OSU\S809.csv"
    x, y, d = read_airfoil(filename)

    fullbase, ext = os.path.splitext(filename)
    base = os.path.basename(fullbase)

    filename_out = os.path.join(os.path.dirname(filename), '_'+base + '.fmt')

    write_airfoil_plot3d(x, y, filename_out, thick=True)
