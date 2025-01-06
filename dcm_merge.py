import pydicom
import argparse
import os

def merge_dicom(base_path, pixels_path, output_name):
    base_ds = pydicom.dcmread(base_path, force=True)
    pixels_ds = pydicom.dcmread(pixels_path, force=True)

    # Copy pixel data and related attributes
    base_ds.PixelData = pixels_ds.PixelData
    base_ds.Rows = pixels_ds.Rows
    base_ds.Columns = pixels_ds.Columns
    base_ds.BitsAllocated = pixels_ds.BitsAllocated
    base_ds.BitsStored = pixels_ds.BitsStored
    base_ds.HighBit = pixels_ds.HighBit
    base_ds.PixelRepresentation = pixels_ds.PixelRepresentation
    base_ds.SamplesPerPixel = pixels_ds.SamplesPerPixel
    if hasattr(pixels_ds, "PhotometricInterpretation"):
        base_ds.PhotometricInterpretation = pixels_ds.PhotometricInterpretation

    base_dir = os.path.dirname(output_name)
    if base_dir:  # Ensure directory exists
        os.makedirs(base_dir, exist_ok=True)

    base_ds.save_as(output_name)

def process_directory(base_path, pixels_dir, output_dir, overwrite):
    for root, _, files in os.walk(pixels_dir):
        for file in files:
            if file.endswith(".dcm"):
                pixels_path = os.path.join(root, file)
                if overwrite:
                    output_path = pixels_path
                else:
                    relative_path = os.path.relpath(pixels_path, pixels_dir)
                    output_path = os.path.join(output_dir, relative_path)

                merge_dicom(base_path, pixels_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge DICOM metadata and pixel data.")
    parser.add_argument("-b", "--base", required=True, help="Path to the base DICOM file")
    parser.add_argument("-p", "--pixels", required=True, help="Path to the pixels DICOM file or directory")
    parser.add_argument("-o", "--output", help="Output directory for the merged DICOM files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the original files in the pixel directory")
    args = parser.parse_args()

    if os.path.isdir(args.pixels):
        if not args.overwrite and not args.output:
            parser.error("Output directory (-o) is required unless --overwrite is set.")

        output_dir = args.output if not args.overwrite else None
        process_directory(args.base, args.pixels, output_dir, args.overwrite)
    else:
        if not args.output:
            parser.error("Output file (-o) is required unless --overwrite is set.")

        output_name = args.pixels if args.overwrite else args.output
        merge_dicom(args.base, args.pixels, output_name)
