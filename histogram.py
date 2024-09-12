import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import os

class ImageHistogram:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img, self.is_dicom = self.load_image()

    def load_image(self):
        if self.image_path.lower().endswith('.dcm'):
            # DICOM image handling
            dicom_img = pydicom.dcmread(self.image_path)
            img = apply_voi_lut(dicom_img.pixel_array, dicom_img)
            if dicom_img.PhotometricInterpretation == "MONOCHROME1":
                img = np.max(img) - img  # Invert if MONOCHROME1
            return img.astype(np.float32), True  # Keep pixel values as float for DICOM
        else:
            # Standard image handling (JPEG, PNG, etc.)
            img = cv2.imread(self.image_path)
            return img, False

    def is_grayscale(self):
        if len(self.img.shape) == 2:  # Already grayscale
            return True
        return np.allclose(self.img[:, :, 0], self.img[:, :, 1]) and np.allclose(self.img[:, :, 1], self.img[:, :, 2])

    def compute_histogram(self):
        if self.is_grayscale():
            # For DICOM, maintain full pixel value range in histogram
            if self.is_dicom:
                hist, _ = np.histogram(self.img.ravel(), bins=256, range=[np.min(self.img), np.max(self.img)])
            else:
                hist, _ = np.histogram(self.img.ravel(), bins=256, range=[0, 255])
        else:
            colors = ('b', 'g', 'r')
            hist = {}
            for i, color in enumerate(colors):
                if self.is_dicom:
                    hist[color], _ = np.histogram(self.img[:, :, i].ravel(), bins=256, range=[np.min(self.img), np.max(self.img)])
                else:
                    hist[color], _ = np.histogram(self.img[:, :, i].ravel(), bins=256, range=[0, 255])
            return hist
        return hist

    def show_histogram(self, average_histograms=None):
        self.fig, self.ax = plt.subplots()

        if average_histograms:
            self.plot_average_histogram(average_histograms)
        else:
            hist = self.compute_histogram()
            if self.is_grayscale():
                print("Detected grayscale image")
                if self.is_dicom:
                    self.ax.hist(self.img.ravel(), bins=256, color='gray', range=[np.min(self.img), np.max(self.img)])
                    plt.title(f'Grayscale DICOM Histogram (Range: {np.min(self.img)} to {np.max(self.img)})')
                else:
                    self.ax.hist(self.img.ravel(), bins=256, color='gray', range=[0, 255])
                    plt.title('Grayscale Histogram (0-255)')
            else:
                print("Detected color image")
                colors = ('b', 'g', 'r')
                for color in colors:
                    hist = self.compute_histogram()[color]
                    self.ax.plot(hist, color=color)
                if self.is_dicom:
                    plt.title(f'Color DICOM Histogram (Range: {np.min(self.img)} to {np.max(self.img)})')
                else:
                    plt.title('Color Histogram (0-255)')

            self.ax.set_xlabel('Pixel Intensity')
            self.ax.set_ylabel('Frequency')

            # Connect the hover event
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)

        plt.show()

    def plot_average_histogram(self, histograms):
        self.ax.set_xlabel('Pixel Intensity')
        self.ax.set_ylabel('Frequency')
        if self.is_grayscale():
            avg_hist = np.mean(histograms, axis=0)
            self.ax.plot(avg_hist, color='gray')
            plt.title('Average Grayscale Histogram')
        else:
            colors = ('b', 'g', 'r')
            avg_hist = {}
            for i, color in enumerate(colors):
                avg_hist[color] = np.mean([hist[color] for hist in histograms], axis=0)
                self.ax.plot(avg_hist[color], color=color)
            plt.title('Average Color Histogram')

    def on_hover(self, event):
        if event.inaxes == self.ax:
            x = int(event.xdata)  # Pixel intensity value
            if self.is_dicom:
                pixel_range = (np.min(self.img), np.max(self.img))
                if pixel_range[0] <= x <= pixel_range[1]:
                    if self.is_grayscale():
                        y = int(self.hist_values[x]) if x < len(self.hist_values) else 0
                        self.ax.set_title(f'Pixel Intensity: {x}, Frequency: {y}')
                    else:
                        frequencies = [int(self.hist_values[c][x]) if x < len(self.hist_values[c]) else 0 for c in ('b', 'g', 'r')]
                        self.ax.set_title(f'Pixel Intensity: {x}, B: {frequencies[0]}, G: {frequencies[1]}, R: {frequencies[2]}')
            else:
                if 0 <= x < 256:
                    if self.is_grayscale():
                        y = int(self.hist_values[x]) if x < len(self.hist_values) else 0
                        self.ax.set_title(f'Pixel Intensity: {x}, Frequency: {y}')
                    else:
                        frequencies = [int(self.hist_values[c][x]) if x < len(self.hist_values[c]) else 0 for c in ('b', 'g', 'r')]
                        self.ax.set_title(f'Pixel Intensity: {x}, B: {frequencies[0]}, G: {frequencies[1]}, R: {frequencies[2]}')
            self.fig.canvas.draw_idle()

def process_directory(directory_path):
    histograms = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            try:
                hist = ImageHistogram(filepath)
                histograms.append(hist.compute_histogram())
            except Exception as e:
                print(f"Could not process {filename}: {e}")
    return histograms

def main():
    parser = argparse.ArgumentParser(description="Display histogram of an image's pixel colors with hover functionality (supports DICOM and standard images)")
    parser.add_argument('path', type=str, help='Path to the image file (JPEG, PNG, or DICOM) or directory containing images')
    args = parser.parse_args()

    if os.path.isdir(args.path):
        histograms = process_directory(args.path)
        if histograms:
            # Initialize with the first histogram to get the image properties (e.g., grayscale/color)
            first_image = ImageHistogram(os.path.join(args.path, os.listdir(args.path)[0]))
            first_image.show_histogram(average_histograms=histograms)
        else:
            print("No images found in the directory.")
    else:
        hist = ImageHistogram(args.path)
        hist.show_histogram()

if __name__ == "__main__":
    main()
