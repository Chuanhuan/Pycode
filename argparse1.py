import argparse

# Create a parser object
parser = argparse.ArgumentParser(description="A simple command-line application")

# Create a group for input arguments
input_group = parser.add_argument_group("Input arguments")

# Add arguments to input group
input_group.add_argument("--input-file", type=str, help="The path to the input file")
input_group.add_argument(
    "--input-format",
    type=str,
    choices=["csv", "json", "xml"],
    help="The format of the input file",
)

# Create a group for output arguments
output_group = parser.add_argument_group("Output arguments")

# Add arguments to output group
output_group.add_argument("--output-file", type=str, help="The path to the output file")
output_group.add_argument(
    "--output-format",
    type=str,
    choices=["csv", "json", "xml"],
    help="The format of the output file",
)

# Parse the arguments
args = parser.parse_args()


# Print the input and output information
print(f"Input file: {args.input_file} ({args.input_format})")
print(f"Output file: {args.output_file} ({args.output_format})")

# python argparse1.py --input-file data.csv --input-format csv --output-file results.csv --output-format csv 
