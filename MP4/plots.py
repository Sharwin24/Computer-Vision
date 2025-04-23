import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(csv_files):
    plt.figure(figsize=(12, 6))
    for i, file in enumerate(csv_files):
        data = np.loadtxt(file, delimiter=",")
        # CSV has 4 columns for [Pixel Value, B, G, R]
        name = file.split("_")[0]
        color_space = file.split("_")[1]
        plt.subplot(1, len(csv_files), i + 1)
        plt.title(f"2D {color_space.upper()} Histogram of {name.upper()}")
        plt.plot(data[:, 0], data[:, 1], color='blue', label='B')
        plt.plot(data[:, 0], data[:, 2], color='green', label='G')
        plt.plot(data[:, 0], data[:, 3], color='red', label='R')
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.grid()
        plt.xlim(0, 255)
        plt.ylim(0, np.max(data[:, 1]) * 1.1)
        plt.xticks(np.arange(0, 256, 32))
        plt.yticks(
            np.arange(0, np.max(data[:, 1]) * 1.1, np.max(data[:, 1]) / 10))
        plt.gca().set_aspect('auto', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"2D_{color_space.upper()}_Color_Histograms.png", dpi=300)
    print(f"Saved histogram to: 2D_{color_space.upper()}_Color_Histograms.png")
    # plt.show()


def bmp2png(bmp_file):
    """
    Convert a BMP file to PNG format using PIL.
    """
    from PIL import Image
    img = Image.open(bmp_file)
    png_file = bmp_file.replace(".bmp", ".png")
    img.save(png_file, "PNG")


if __name__ == "__main__":
    csv_files = ["gun1_BGR_histogram.csv",
                 "joy1_BGR_histogram.csv",
                 "pointer1_BGR_histogram.csv"
                 ]
    plot_histograms(csv_files)
    # images = []
    # [bmp2png(img) for img in images]
