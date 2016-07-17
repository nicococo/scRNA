import argparse, sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arguments = parser.parse_args(sys.argv[1:])
    print('Parameters:')
    print arguments
    print
