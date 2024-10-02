import json
import argparse


def calculate_area(bbox):
    """Calculate the area of a bounding box."""
    width = bbox[2]
    height = bbox[3]
    return width * height


def process_json(input_file, output_file):
    """Process the input JSON file and add area to each annotation."""
    with open(input_file, 'r') as f:
        data = json.load(f)

    for annotation in data['annotations']:
        bbox = annotation['bbox']
        area = calculate_area(bbox)
        annotation['area'] = area

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Calculate and add area to COCO-style JSON annotations.")
    parser.add_argument('input', help="Input JSON file path")
    parser.add_argument('output', help="Output JSON file path")

    args = parser.parse_args()

    process_json(args.input, args.output)
    print(f"Processing complete. Output written to {args.output}")


if __name__ == "__main__":
    main()