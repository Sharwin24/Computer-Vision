import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(csv_files, figure_title):
    plt.figure(figsize=(12, 6))
    for i, h in enumerate(csv_files):
        data = np.loadtxt(h, delimiter=",")
        plt.subplot(1, 2, i + 1)
        if i == 0:
            plt.scatter(data[:, 0], data[:, 1], color='blue', s=10)
            # Connect the points with lines in x-order
            sorted_indices = np.argsort(data[:, 0])
            plt.plot(data[sorted_indices, 0],
                     data[sorted_indices, 1], color='blue', linewidth=2)
            plt.fill_between(data[sorted_indices, 0],
                             data[sorted_indices, 1], color='blue', alpha=0.2)
            area = np.trapezoid(data[:, 1], data[:, 0])
            plt.text(0.75, 0.5, f"Area = {area:.2f}", fontsize=12,
                     ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.plot(data[:, 0], data[:, 1], color='blue', linewidth=2)
            plt.fill_between(data[:, 0], data[:, 1], color='blue', alpha=0.2)
            area = np.trapezoid(data[:, 1], data[:, 0])
            plt.text(0.75, 0.5, f"Area = {area:.2f}", fontsize=12,
                     ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(histogram_titles[i])
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency" if i == 0 else "Cumulative Frequency")
        plt.grid()
        plt.xlim(0, 255)
        plt.ylim(0, np.max(data[:, 1]) * 1.1)
        plt.xticks(np.arange(0, 256, 32))
        plt.yticks(
            np.arange(0, np.max(data[:, 1]) * 1.1, np.max(data[:, 1]) / 10))
        plt.gca().set_aspect('auto', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"{figure_title}.png", dpi=300)
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
    csv_files = []
    plot_histograms(csv_files, "2D Color Histograms")
    images = []
    [bmp2png(img) for img in images]
