import argparse

parser = argparse.ArgumentParser(description="Process some integers")
parser.add_argument("--json", type=str, required=True, help="Path to the json file")
parser.add_argument("--output", "-o", type=str, required=True, help="Path to the txt file")

args = parser.parse_args()

import json


def decode_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    captions = data["annotations"]
    decode_list = []
    for caption in captions:
        decode_list.append(caption["caption"])
    return decode_list


decode_list = decode_json(args.json)
json.dump(decode_list, open(args.output, "w"))
