import argparse

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--infor", required=True, help="information")
    ap.add_argument("-m", "--comment", required=True, help="comment ")
    ap.add_argument("-o", "--output", default=123, help=" output")
    args = vars(ap.parse_args())

    print(args)
    # print(
    #     "this is information %s, comment: %s, output: $d"
    #     % (ap.infor, ap.comment, ap.output)
    # )

#     python argparse/argparse3.py -i 'abc 123' -m show
# {'infor': 'abc 123', 'comment': 'show', 'output': 123}

