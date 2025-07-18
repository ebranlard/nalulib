
import os
from nalulib.essentials import myprint


def nalu_batch(batch_file_template=None, nalu_input_file=None, cluster=None, verbose=False, jobname=None):
    """ Create a batch file for a nalu simulation based on a template and a cluster"""
    if batch_file_template is None:
        if cluster == 'unity':
            batch_file_template = os.path.dirname(__file__) + '/_template_submit-unity.sh'
        elif cluster == 'kestrel':
            batch_file_template = os.path.dirname(__file__) + '/_template_submit-kestrel.sh'
        else:
            raise Exception('Unknown cluster {}'.format(cluster))

    if verbose:
        myprint('Using batch_file', batch_file_template)

    with open(batch_file_template, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # look for the line nalu_input=XXX and replace it with nalu_input=nalu_input_file
    for i, line in enumerate(lines):
        if jobname is not None:
            # Check if line contains the string "--job_name" anywher
            if '--job-name' in line:
                # Replace the line with the new job name
                lines[i] = "#SBATCH --job-name={}\n".format(jobname)
        if line.startswith('nalu_input'):
            nalu_input_file = nalu_input_file.replace('./','').replace('.\\','')
            lines[i] = "nalu_input={}".format(nalu_input_file)+'\n'
            break
    # Write the new batch file in the current directory, based on the output_filename
    base_dir = os.path.dirname(nalu_input_file)
    base_name = os.path.basename(nalu_input_file)
    base, ext = os.path.splitext(base_name)
    # new_batch = os.path.splitext(nalu_input_file)[0] + '.sh'
    new_batch = os.path.join(base_dir, 'submit-'+base+'.sh') 
    with open(new_batch, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    if verbose:
        myprint('Written              ', new_batch)
    return new_batch
