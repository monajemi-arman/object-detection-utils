import os
import cv2
import argparse

# Perform sanity check on a YOLO label file
def check_yolo_label(label_file, image_file):
    image = cv2.imread(image_file)
    if image is None:
        return False

    img_height, img_width, _ = image.shape

    with open(label_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        elements = line.strip().split()
        if len(elements) != 5:
            continue

        class_id, center_x, center_y, width, height = map(float, elements)

        center_x *= img_width
        center_y *= img_height
        width *= img_width
        height *= img_height

        x_min = center_x - width / 2
        x_max = center_x + width / 2
        y_min = center_y - height / 2
        y_max = center_y + height / 2

        if x_min < 0 or x_max > img_width or y_min < 0 or y_max > img_height:
            print(f"Bounding box in {label_file}, line {i+1} exceeds image boundaries")
            return False

    return True

# Check all labels in a directory
def check_all_labels(label_dir, image_dir):
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    for label_file in label_files:
        image_file = os.path.join(image_dir, label_file.replace('.txt', '.jpg'))
        label_file_path = os.path.join(label_dir, label_file)

        if not os.path.exists(image_file):
            continue

        if not check_yolo_label(label_file_path, image_file):
            print(f"Sanity check failed for {label_file}")

def main():
    parser = argparse.ArgumentParser(description="Sanity check YOLO labels against image dimensions")
    parser.add_argument('--label_dir', type=str, required=True, help='Path to the directory containing YOLO label files')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing image files')
    args = parser.parse_args()

    check_all_labels(args.label_dir, args.image_dir)

if __name__ == "__main__":
    main()
