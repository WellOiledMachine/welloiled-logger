import argparse
from pathlib import PurePosixPath

parser = argparse.ArgumentParser("Dumb parser")
parser.add_argument("path", type=PurePosixPath)
args = parser.parse_args()

args.path = args.path.expanduser().resolve()
print(str(args.path))
