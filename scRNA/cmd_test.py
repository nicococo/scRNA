import argparse, sys

parser = argparse.ArgumentParser()
arguments = parser.parse_args(sys.argv[1:])
print('Parameters:')
print arguments
