import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", help="Dataset to run (default Toy)", default="Toy", type=str)
parser.add_argument("-o","--hold_out", help="Fraction of hold-out examples for reps (default 0.0)", default=0.0, type =float)
parser.add_argument("-r","--reps", help="number repetitions (default 1)", default=1, type =int)
parser.add_argument("-e","--experiment", help="active experiment [0-2] (default 1)", default=1, type =int)

arguments = parser.parse_args(sys.argv[1:])
print('Parameters:')
print arguments
