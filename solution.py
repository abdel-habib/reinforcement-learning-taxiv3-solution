import argparse

if __name__ == "__main__":
    # optional arguments from the command line 
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, default='qlearning', help='algorithm to solve the task. Has to be one of ["qlearning", "random"]')

    # parse the arguments
    args = parser.parse_args()

    # assert args values
    assert args.algorithm in ["qlearning", "random"], 'Wronge input values for the algorithm to solve the challenge! Pass one of ["qlearning", "random"] in the argument `--algorithm`.'
    
