import json
import argparse

def add_iscrowd_to_annotations(input_json, output_json):
    with open(input_json, 'r') as f:
        data = json.load(f)

    for annotation in data['annotations']:
        if 'iscrowd' not in annotation:
            annotation['iscrowd'] = 0

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Updated annotations saved to {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Add 'iscrowd' field to COCO annotations")
    parser.add_argument("-i", "--input", required=True, help="Path to input COCO JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to output COCO JSON file")
    args = parser.parse_args()

    add_iscrowd_to_annotations(args.input, args.output)

if __name__ == "__main__":
    main()
