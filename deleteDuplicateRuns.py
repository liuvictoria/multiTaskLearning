""" Docstring for the deleteDuplicateRuns.py module.

Sometimes a run freezes before completion.
Although models are overwritten in a future completed run,
the exploratory tensorboard looks bad with unfinished runs overlaid.

Use this script to delete those unfinished runs.
All runs except the largest-file-size run are sent to trash.

Run this script from the command line with the appropriate 
runnames and and directory as command line arguments.
The program prints which runs are sent to trash.

"""

import os
from send2trash import send2trash
import argparse

# command line argument parser
parser = argparse.ArgumentParser(
    description = 'define parameters and roots for STL training'
)

# which runs to investigate for prematurely stopped runs
parser.add_argument(
    '--runnames', nargs = '+',
    help="""which runs to investigate for prematurely stopped runs; 
        i.e. MTL_dwa_varnet0:10_div_coronal_pd_fs_div_coronal_pd""",
    required = True,
)

parser.add_argument(
    '--rundir',
    help="""directory of runs""",
    required = True,
)

opt = parser.parse_args()


for runname in opt.runnames:
    os.chdir(os.path.join(opt.rundir, runname))
    root = os.getcwd()
    listItems = os.listdir()
    for directory in listItems:
        # i.e. item = N=32_N=481_l1_train
        if os.path.isdir(directory):
            # set current directory
            os.chdir(directory)

            # find directories with more than one file
            output1 = os.popen("find . -type f -printf '%h\n' | sort | uniq -d").read()
            problem_dirs = output1.split('\n')[:-1]
            print(f'working on directory {directory}')

            # for each directory with more than one file
            for problem_dir in problem_dirs:
                print(f'problematic directory {problem_dir}')
                os.chdir(problem_dir)
                # i.e. problem_dir = './div_coronal_pd'
                # delete everything except largest file
                output2 = os.popen("find . -type f -printf '%s %p\n' | sort -nr").read()
                problem_files = output2.split('\n')[1:-1]
                problem_files = [problem_file.split(' ')[1] for problem_file in problem_files]
                print(f'problematic files {problem_files}')
                send2trash(problem_files)

                os.chdir('../')

        os.chdir(root)