import argparse

# Create a parser object
parser = argparse.ArgumentParser(description='A simple command-line calculator')

# Add arguments to the parser
parser.add_argument('operator', choices=['add', 'sub', 'mult', 'div'], help='The operator to use in the calculation')
parser.add_argument('operand1', type=int, help='The first operand')
parser.add_argument('operand2', type=int, help='The second operand')

# Parse the arguments
args = parser.parse_args()

# Perform the calculation based on the input
if args.operator == 'add':
    result = args.operand1 + args.operand2
elif args.operator == 'sub':
    result = args.operand1 - args.operand2
elif args.operator == 'mult':
    result = args.operand1 * args.operand2
else:
    result = args.operand1 / args.operand2

print(f'The result is: {result}')
# python calculator.py add 5 7
