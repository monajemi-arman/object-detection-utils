#!/usr/bin/env python
import json
from argparse import ArgumentParser
import regex
from Cython.Compiler.Future import annotations

parser = ArgumentParser()
parser.add_argument('-p', '--pattern', required=True, help="Regex pattern to match")
parser.add_argument('-r', '--reverse', required=False, help="Reverse pattern use, exclude the matches")
parser.add_argument('-i', '--input', required=True, help="Path to input JSON")
parser.add_argument('-o', '--output', required=True, help='Path to output JSON')
parsed = parser.parse_args()

pattern = parsed.pattern.strip("'\"")
input_json = parsed.input
output_json = parsed.output
match_toggle = True
if parsed.reverse:
    match_toggle = False

with open(input_json) as f:
    data = json.load(f)

# Iterate over images
ids_to_remove = []
new_images = []
for image in data['images']:
    image_id = image['id']
    file_name = image['file_name']
    match = regex.match(pattern, file_name)
    if not ((match_toggle and match) or (not match_toggle and not match)):
        ids_to_remove.append(image_id)
    else:
        new_images.append(image)

data['images'] = new_images

# Remove corresponding annotations
new_annotations = []
for annotation in data['annotations']:
    if annotation['image_id'] not in ids_to_remove:
        new_annotations.append(annotation)

data['annotations'] = new_annotations

# Save to output JSON
with open(output_json, 'w') as f:
    json.dump(data, 'output.json')