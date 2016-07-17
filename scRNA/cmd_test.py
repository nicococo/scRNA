import argparse

def main(args):
    parser = argparse.ArgumentParser()
    arguments = parser.parse_args(args)
    print('Parameters:')
    print arguments
