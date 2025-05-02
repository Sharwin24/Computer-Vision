/**
 * @file bmp.hpp
 * @author Sharwin Patil (sharwinpatil@u.northwestern.edu)
 * @brief Support for BMP image processing
 * @version 6.0
 * @date 2025-05-02
 * @details 1.0 - Includes support for Connected Component Labeling, size filtering, and colorizing components.
 * @details 2.0 - Includes support for morphological operations: erosion, dilation, opening, and closing.
 * @details 3.0 - Support reading 24-bit color images, histogram equilization, and lighting correction.
 * @details 4.0 - Support for user pixel selection, color space conversion, and color histogram
 * @details 5.0 - Implemented Gaussian Blur, Canny Edge Detection, and RobertCross/Sobel Kernel convolutions.
 * @details 6.0 - Implemented Hough Transform
 *
 * @note Features to add:
 * - Pixel struct/class
 * - Doxygen for everything
 * - Store Pixel locations as a set instead of a vector
 * - Delegate some functions to other classes in other files (Convolution, Pixel ColorSpace conversion)
 * - Clean up the structure of commonly used data structures (less saving data structures to files!)
 * - Add functions for common operations (reading file, 2D iteration, etc.)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef BMP_HPP
#define BMP_HPP
#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <algorithm> // For std::remove_if
#include <math.h>
#include <unordered_set>
#include <string>

 // Include OpenCV for Windows and user interaction
#include <opencv2/opencv.hpp>

#pragma pack(push, 1) // Ensure no padding is added to the structures

#define BMFILETYPE 0x4D42 // 'BM' in ASCII

 // --------------- BEGIN_CITATION [1] ---------------- //
 // https://solarianprogrammer.com/2018/11/19/cpp-reading-writing-bmp-images/
struct BMPFileHeader {
  uint16_t file_type{BMFILETYPE};      // File type always BM which is 0x4D42
  uint32_t file_size{0};               // Size of the file (in bytes)
  uint16_t reserved1{0};               // Reserved, always 0
  uint16_t reserved2{0};               // Reserved, always 0
  uint32_t offset_data{0};             // Start position of pixel data (bytes from the beginning of the file)
};

struct BMPInfoHeader {
  uint32_t size{0};                      // Size of this header (in bytes)
  int32_t width{0};                      // width of bitmap in pixels
  int32_t height{0};                     // height of bitmap in pixels
  //       (if positive, bottom-up, with origin in lower left corner)
  //       (if negative, top-down, with origin in upper left corner)
  uint16_t planes{1};                    // No. of planes for the target device, this is always 1
  uint16_t bit_count{0};                 // No. of bits per pixel
  uint32_t compression{0};               // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
  uint32_t size_image{0};                // 0 - for uncompressed images
  int32_t x_pixels_per_meter{0};
  int32_t y_pixels_per_meter{0};
  uint32_t colors_used{0};               // No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
  uint32_t colors_important{0};          // No. of colors used for displaying the bitmap. If 0 all colors are required
};

struct BMPColorHeader {
  uint32_t red_mask{0x00ff0000};         // Bit mask for the red channel
  uint32_t green_mask{0x0000ff00};       // Bit mask for the green channel
  uint32_t blue_mask{0x000000ff};        // Bit mask for the blue channel
  uint32_t alpha_mask{0xff000000};       // Bit mask for the alpha channel
  uint32_t color_space_type{0x73524742}; // Default "sRGB" (0x73524742)
  uint32_t unused[16]{0};                // Unused data for sRGB color space
};
#pragma pack(pop) // Restore the previous packing alignment
// --------------- END_CITATION [1] ---------------- //

/**
 * @brief Representation of a component in an image
 *        obtained through connected component labeling.
 *
 */
struct Component {
  uint32_t label; // Unique Label of the component
  uint32_t area; // Number of pixels in the component
  std::vector<std::pair<uint8_t, uint8_t>> pixels; // List of pixel locations [row, col]
};

/**
 * @brief Class describing a Kernel for
 * Morphological Operations and Convolutions.
 *
 */
class StructuringElement {
public:

  StructuringElement() = delete;

  /**
   * @brief Create a square SE Kernel with the given size. Kernel values all default to 1.
   *
   * @param size the number of rows and columns in the kernel
   * @param value the initial value for the kernel elements. Defaults to 1.
   */
  StructuringElement(int size, float value = 1.0f) : kernelSize(size) {
    this->kernel.resize(size, std::vector<float>(size, value));
  }

  StructuringElement(const StructuringElement& original) : kernelSize(original.kernelSize) {
    this->kernel.resize(original.kernelSize, std::vector<float>(original.kernelSize, 0.0f));
    for (int i = 0; i < original.kernelSize; ++i) {
      for (int j = 0; j < original.kernelSize; ++j) {
        this->kernel[i][j] = original.kernel[i][j];
      }
    }
  }

  int getKernelSize() const { return this->kernelSize; }

  void setElement(int r, int c, float value) {
    if (r >= 0 && r < this->kernelSize && c >= 0 && c < this->kernelSize) {
      this->kernel[r][c] = value;
    }
  }

  float operator()(int r, int c) const {
    if (r >= 0 && r < this->kernelSize && c >= 0 && c < this->kernelSize) {
      return this->kernel[r][c];
    } else {
      throw std::out_of_range("Kernel Index (" + std::to_string(r) + "," +
        std::to_string(c) + ") out of range for kernel size: " +
        std::to_string(this->kernelSize) + "x" + std::to_string(this->kernelSize)
      );
    }
  }

  void initGaussianKernel(const float sigma) {
    // Create a Gaussian kernel with the given sigma
    // G(x,y) = (1/(2*π*σ²)) * e^(-(x²+y²)/(2*σ²))
    int ksize = this->kernelSize;
    float sum = 0.0f;
    int halfSize = ksize / 2;
    for (int x = -halfSize; x <= halfSize; ++x) {
      for (int y = -halfSize; y <= halfSize; ++y) {
        float value = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
        this->kernel[x + halfSize][y + halfSize] = value;
        sum += value;
      }
    }
    // Normalize the kernel
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        this->kernel[i][j] /= sum;
      }
    }
  }

  void initRobertCrossKernel(bool Gx) {
    this->kernel.clear();
    this->kernelSize = 2;
    this->kernel.resize(2, std::vector<float>(2, 0.0f));
    if (Gx) {
      this->kernel[0][0] = 1.0f; this->kernel[0][1] = 0.0f;
      this->kernel[1][0] = 0.0f; this->kernel[1][1] = -1.0f;
    } else {
      this->kernel[0][0] = 0.0f; this->kernel[0][1] = 1.0f;
      this->kernel[1][0] = -1.0f; this->kernel[1][1] = 0.0f;
    }
  }

  void initSobelKernel(bool Gx) {
    this->kernel.clear();
    this->kernelSize = 3;
    this->kernel.resize(3, std::vector<float>(3, 0.0f));
    if (Gx) {
      // Left column
      this->kernel[0][0] = -1.0f;
      this->kernel[1][0] = -2.0f;
      this->kernel[2][0] = -1.0f;
      // Middle Column is all zeros
      // Right column
      this->kernel[0][2] = 1.0f;
      this->kernel[1][2] = 2.0f;
      this->kernel[2][2] = 1.0f;
    } else {
      // Top Row
      this->kernel[0][0] = -1.0f;
      this->kernel[0][1] = -2.0f;
      this->kernel[0][2] = -1.0f;
      // Middle Row is all zeros
      // Bottom column
      this->kernel[2][0] = 1.0f;
      this->kernel[2][1] = 2.0f;
      this->kernel[2][2] = 1.0f;
    }
  }

private:
  std::vector<std::vector<float>> kernel;
  int kernelSize;
};

struct ImageGradient {
  float magnitude{0}; // Normalized magnitude
  float direction{0}; // [rad]
};

enum class ColorSpace {
  BGR, // Blue, Green, Red (default for BMP)
  HSI, // Hue, Saturation, Intensity
  NBGR, // Normalized BGR
};

class BMPImage {
public:
  BMPImage() = delete;

  BMPImage(const char* filename) {
    this->read(filename);
    this->name = getImageName(filename);
  }

  BMPImage(const BMPImage& original)
    : fileHeader(original.fileHeader),
    infoHeader(original.infoHeader),
    colorHeader(original.colorHeader),
    pixelData2D(original.pixelData2D),
    name(original.name),
    components(original.components) {
  }

  BMPImage(const BMPImage& original, const std::vector<std::vector<uint8_t>>& pixelData)
    : BMPImage(original) {
    this->pixelData2D = pixelData;
  }

  ~BMPImage() = default;

  void save(const std::string filename) {
    if (this->colorSpace == ColorSpace::HSI) {
      this->changeColorSpace(ColorSpace::BGR);
    }
    this->write(filename.c_str());
  }

  void printInfo() const {
    std::cout << "BMP Image: " << this->name << std::endl;
    std::cout << "File Size: " << this->fileHeader.file_size << " bytes" << std::endl;
    std::cout << "Width: " << this->infoHeader.width << " pixels" << std::endl;
    std::cout << "Height: " << this->infoHeader.height << " pixels" << std::endl;
    std::cout << "NumPixels: " << this->infoHeader.width * this->infoHeader.height << std::endl;
    std::cout << "ImageSize: " << this->infoHeader.size_image << " bytes" << std::endl;
    std::cout << "Bit Count: " << this->infoHeader.bit_count << std::endl;
    std::cout << "Compression: " << this->infoHeader.compression << std::endl;
    std::cout << "Colors Used: " << this->infoHeader.colors_used << std::endl;
    std::cout << "Color Space: " << this->colorSpaceName(this->colorSpace) << std::endl;
  }

  void printPixelData() const {
    std::cout << "Pixel Data (2D):\n";
    for (const auto& row : this->pixelData2D) {
      for (const auto& pixel : row) {
        std::cout << static_cast<int>(pixel) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    // Print to a log file to visually verify raw pixel data
    std::ofstream logFile("raw_pixel_data.txt");
    if (logFile.is_open()) {
      logFile << "Pixel Data (2D):\n";
      for (const auto& row : this->pixelData2D) {
        for (const auto& pixel : row) {
          logFile << static_cast<int>(pixel) << " ";
        }
        logFile << std::endl;
      }
      logFile.close();
    } else {
      std::cerr << "Unable to open log file" << std::endl;
    }
  }

  std::string getName() const { return this->name; }

  void connectedComponentLabeling() {
    std::cout << "Connected Component Labeling" << std::endl;
    // Convert the pixel data to a 2D binary image
    std::vector<std::vector<int>> binaryImage = this->convertToBinaryImage();

    const int numRows = binaryImage.size();
    const int numCols = binaryImage[0].size();
    std::cout << "Binary Image Size: " << numRows << " x " << numCols << std::endl;
    // First pass: Assign labels and record equivalences
    uint32_t label = 1;
    std::vector<std::vector<int>> labeledImage(numRows, std::vector<int>(numCols, 0));
    std::vector<int> parent(numRows * numCols + 1, 0);
    for (unsigned int i = 0; i < parent.size(); ++i) {
      parent[i] = i; // Initialize parent to itself
    }
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        // Skip Background pixels
        if (binaryImage[r][c] == 0) { continue; }

        // Check neighbors
        int left = (c > 0) ? labeledImage[r][c - 1] : 0;
        int up = (r > 0) ? labeledImage[r - 1][c] : 0;

        if (up == left && up != 0 && left != 0) { // Same non-zero label
          // L(r,c) = up
          labeledImage[r][c] = up;
        } else if (up != left && !(up && left)) { // Either is zero
          // L(r,c) = max(up, left)
          labeledImage[r][c] = std::max(up, left);
        } else if (up != left && up != 0 && left != 0) { // Both are non-zero
          // L(r,c) = min(up, left)
          labeledImage[r][c] = std::min(up, left);
          // E_table(up, left)
          this->unionLabels(parent, up, left);
        } else { // Both are zero
          // L(r,c) = L + 1
          labeledImage[r][c] = label++;
        }
      }
    }

    std::cout << "Labeling completed. Number of labels: " << label - 1 << std::endl;

    // Second pass: Resolve equivalences
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        if (labeledImage[r][c] != 0) {
          labeledImage[r][c] = this->findRoot(parent, labeledImage[r][c]);
        }
      }
    }

    std::set<int> uniqueLabels;
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        if (labeledImage[r][c] != 0) {
          uniqueLabels.insert(labeledImage[r][c]);
        }
      }
    }
    std::cout << "Unique labels found: " << uniqueLabels.size() << std::endl;

    // Create components
    this->components.clear();
    std::unordered_map<int, int> labelToComponentIndex;
    int componentIndex = 0;
    for (int label : uniqueLabels) {
      labelToComponentIndex[label] = componentIndex++;
      this->components.push_back({static_cast<uint32_t>(label), 0, {}});
    }

    // Populate components
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        if (labeledImage[r][c] != 0) {
          int lab = labeledImage[r][c];
          int compIndex = labelToComponentIndex[lab];
          this->components[compIndex].pixels.emplace_back(r, c);
          this->components[compIndex].area++;
        }
      }
    }

    // Sort components by label and reassign to be contiguous
    std::sort(this->components.begin(), this->components.end(),
      [](const Component& a, const Component& b) {
      return a.label < b.label;
    });
    uint32_t newLabel = 1;
    for (auto& c : this->components) {
      c.label = newLabel++;
    }
    // Print component information
    std::cout << "Found " << this->components.size() << " components" << std::endl;
    for (const auto& c : this->components) {
      std::cout << "Component " << c.label << ": Area = " << c.area << std::endl;
    }
  }

  void applySizeFilter(const unsigned int sizeThreshold = 10) {
    // Add safety check for empty components
    if (this->components.empty()) {
      std::cout << "No components to filter!" << std::endl;
      return;
    }

    std::cout << "Applying size filter with threshold: " << sizeThreshold << std::endl;
    std::cout << "Number of components before size filter: " << this->components.size() << std::endl;

    // Track the components we need to filter out and remove
    std::vector<Component> filteredComponents;
    // Filter components by area, save components that are smaller than the threshold
    for (const auto& c : this->components) {
      if (c.area < sizeThreshold) {
        filteredComponents.push_back(c);
      }
    }

    std::cout << "Number of components after size filter: " << this->components.size() - filteredComponents.size() << std::endl;

    // Check if we have any components that passed the filter
    if (filteredComponents.empty()) { return; }

    // Set pixels part of filtered components to background (0)
    int pixelsSet = 0;
    for (const auto& c : filteredComponents) {
      // Add safety check for empty pixels
      if (c.pixels.empty()) {
        std::cout << "Warning: Component has no pixels" << std::endl;
        continue;
      }

      // Set pixels that are filtered out to background (0)
      for (const auto& pixel : c.pixels) {
        this->pixelData2D[pixel.first][pixel.second] = 0; // Set to background
        pixelsSet++;
      }
    }

    std::cout << "Set " << pixelsSet << " pixels to background in filtered image" << std::endl;

    // Remove filtered components from the original components
    for (const auto& c : filteredComponents) {
      auto it = std::remove_if(this->components.begin(), this->components.end(),
        [&c](const Component& comp) { return comp.label == c.label; });
      this->components.erase(it, this->components.end());
    }

    // Sort the components by label
    std::sort(this->components.begin(), this->components.end(),
      [](const Component& a, const Component& b) {
      return a.label < b.label;
    });

    // Update the labels of the remaining components to be contiguous
    uint32_t newLabel = 1;
    for (auto& c : this->components) {
      c.label = newLabel++;
    }

    std::cout << "Filtered out " << filteredComponents.size() << " components" << std::endl;
    std::cout << "Filtered pixel data size: " << this->pixelData2D.size() << std::endl;
  }

  void colorizeComponents() {
    // Sort the components by label
    std::sort(this->components.begin(), this->components.end(),
      [](const Component& a, const Component& b) {
      return a.label < b.label;
    });

    // Collect unique labels from sorted components
    std::set<uint32_t> uniqueLabels;
    for (const auto& c : this->components) {
      uniqueLabels.insert(c.label);
    }

    // Create a color map for each unique label
    // where each label is assigned a unique color
    std::unordered_map<uint32_t, std::vector<uint8_t>> colorMap;
    uint32_t colorIndex = 1;
    for (const auto& label : uniqueLabels) {
      // Generate a unique color for each label
      uint8_t r = (colorIndex * 123) % 256;
      uint8_t g = (colorIndex * 456) % 256;
      uint8_t b = (colorIndex * 789) % 256;
      colorMap[label] = {r, g, b};
      colorIndex++;
    }

    // Calculate the width of each row in bytes (including padding to 4-byte boundary)
    int rowWidth = this->infoHeader.width * 3; // 3 bytes per pixel (RGB)

    // Create a new pixel data array for the colored image
    // Each pixel now needs 3 bytes (R,G,B) instead of 1
    std::vector<std::vector<uint8_t>> coloredPixelData(this->pixelData2D.size(),
      std::vector<uint8_t>(rowWidth, 0)); // RGB image

    // Color each component with its assigned color
    for (const auto& c : this->components) {
      // Get the color for the current label
      auto color = colorMap[c.label];
      for (const auto& pixel : c.pixels) {
        int row = pixel.first;
        int col = pixel.second;
        // Set the RGB values in the colored pixel data (BGR order for BMP)
        coloredPixelData[row][col * 3] = color[2];     // Blue
        coloredPixelData[row][col * 3 + 1] = color[1]; // Green
        coloredPixelData[row][col * 3 + 2] = color[0]; // Red
      }
    }

    // Update the BMP info header for 24-bit color depth
    this->infoHeader.bit_count = 24;
    int rowSize = ((this->infoHeader.bit_count * this->infoHeader.width + 31) / 32) * 4;
    this->infoHeader.size_image = rowSize * std::abs(this->infoHeader.height);
    this->infoHeader.colors_used = 0; // Not used for 24-bit images
    this->infoHeader.compression = 0; // No compression
    this->infoHeader.size = sizeof(BMPInfoHeader);

    // Update the pixel data to the new colored pixel data
    this->pixelData2D = coloredPixelData;

    // Update the file header offset to point to the new pixel data
    this->fileHeader.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    // Update the file size
    this->fileHeader.file_size = this->fileHeader.offset_data + this->infoHeader.size_image;

    // Save the colored image
    std::string outputFilename = this->name + "_components.bmp";
    this->write(outputFilename.c_str());

    std::cout << "Saved colored components to " << outputFilename << std::endl;
  }

  std::vector<std::vector<uint8_t>> convolution(const std::vector<std::vector<uint8_t>>& image, const StructuringElement& kernel) {
    // Convolve the image with the kernel
    const int R = image.size();
    const int C = image[0].size();
    const int K = kernel.getKernelSize();
    const int KHalf = K / 2;
    std::vector<std::vector<uint8_t>> convolvedImage(R, std::vector<uint8_t>(C, 0));
    for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        // Apply kernel to pixel location
        for (int kr = -KHalf; kr <= KHalf; kr++) {
          for (int kc = -KHalf; kc <= KHalf; kc++) {
            // Skip if kernel value is 0
            if (kernel(kr + KHalf, kc + KHalf) == 0) { continue; }
            int imageR = r + kr;
            int imageC = c + kc;
            if (imageR >= 0 && imageR < R && imageC >= 0 && imageC < C) {
              sum += static_cast<float>(image[imageR][imageC]) * kernel(kr + KHalf, kc + KHalf);
            }
          }
        }
        // After applying the kernel, set the pixel value
        convolvedImage[r][c] = static_cast<uint8_t>(std::clamp(sum, 0.0f, 255.0f));
      }
    }
    return convolvedImage;
  }

  std::vector<std::vector<uint8_t>> erosion(const StructuringElement& kernel = StructuringElement(3, 3)) {
    // E = A \ominus B = \{z \mid B_z \subseteq A \}
    // Convolve the image with the kernel
    const int R = this->pixelData2D.size();
    const int C = this->pixelData2D[0].size();

    std::vector<std::vector<uint8_t>> erodedImage(R, std::vector<uint8_t>(C, 0));
    std::cout << "Erosion on " << this->infoHeader.bit_count << "-bit image" << std::endl;
    for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
        uint8_t minValue = this->infoHeader.bit_count == 1 ? 1 : 255;
        // Apply kernel to pixel location
        for (int kr = -kernel.getKernelSize() / 2; kr <= kernel.getKernelSize() / 2; kr++) {
          for (int kc = -kernel.getKernelSize() / 2; kc <= kernel.getKernelSize() / 2; kc++) {
            // Skip if kernel value is 0
            if (kernel(kr + kernel.getKernelSize() / 2, kc + kernel.getKernelSize() / 2) == 0) { continue; }
            int imageR = r + kr;
            int imageC = c + kc;
            if (imageR >= 0 && imageR < R && imageC >= 0 && imageC < C) {
              minValue = std::min(minValue, this->pixelData2D[imageR][imageC]);
            } else {
              minValue = 0;
            }
          }
        }

        // After applying the kernel, set the pixel value
        erodedImage[r][c] = minValue;
      }
    }

    return erodedImage;
  }

  std::vector<std::vector<uint8_t>> dilation(const StructuringElement& kernel = StructuringElement(3, 3)) {
    // A \oplus B = \left\{z \mid \left(\hat{B}\right)_z \cap A \neq \phi \right\} 
    // = \cup_{a_i \in A} B_{a_i}
    // Convolve the image with the kernel
    const int R = this->pixelData2D.size();
    const int C = this->pixelData2D[0].size();

    std::vector<std::vector<uint8_t>> dilatedImage(R, std::vector<uint8_t>(C, 0));
    std::cout << "Dilation on " << this->infoHeader.bit_count << "-bit image" << std::endl;
    for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
        uint8_t maxValue = 0;
        // Apply kernel to pixel location
        for (int kr = -kernel.getKernelSize() / 2; kr <= kernel.getKernelSize() / 2; kr++) {
          for (int kc = -kernel.getKernelSize() / 2; kc <= kernel.getKernelSize() / 2; kc++) {
            // Skip if kernel value is 0
            if (kernel(kr + kernel.getKernelSize() / 2, kc + kernel.getKernelSize() / 2) == 0) { continue; }
            int imageR = r + kr;
            int imageC = c + kc;
            if (imageR >= 0 && imageR < R && imageC >= 0 && imageC < C) {
              maxValue = std::max(maxValue, this->pixelData2D[imageR][imageC]);
            }
          }
        }

        // After applying the kernel, set the pixel value
        dilatedImage[r][c] = maxValue;
      }
    }

    return dilatedImage;
  }

  std::vector<std::vector<uint8_t>> opening(const StructuringElement& kernel = StructuringElement(3, 3)) {
    // A \circ B = (A \ominus B) \oplus B
    std::vector<std::vector<uint8_t>> erodedImage = this->erosion(kernel);
    BMPImage erodedBmp(*this, erodedImage);
    std::vector<std::vector<uint8_t>> openedImage = erodedBmp.dilation(kernel);
    return openedImage;
  }

  std::vector<std::vector<uint8_t>> closing(const StructuringElement& kernel = StructuringElement(3, 3)) {
    // A \bullet B = (A \oplus B) \ominus B
    std::vector<std::vector<uint8_t>> dilatedImage = this->dilation(kernel);
    BMPImage dilatedBmp(*this, dilatedImage);
    std::vector<std::vector<uint8_t>> closedImage = dilatedBmp.erosion(kernel);
    return closedImage;
  }

  std::vector<std::vector<uint8_t>> boundary(const StructuringElement& kernel = StructuringElement(3, 3)) {
    // \beta(A) = A - (A \ominus B)
    std::vector<std::vector<uint8_t>> erodedImage = this->erosion(kernel);
    std::vector<std::vector<uint8_t>> boundaryImage(
      this->pixelData2D.size(), std::vector<uint8_t>(this->pixelData2D[0].size(), 0)
    );
    // Perform pixel-wise subtraction
    const int R = this->pixelData2D.size();
    const int C = this->pixelData2D[0].size();
    for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
        boundaryImage[r][c] = std::max(this->pixelData2D[r][c] - erodedImage[r][c], 0);
      }
    }
    return boundaryImage;
  }

  void histogramEquilization(bool saveToCSV = false) {
    // Convert 24-bit image to grayscale
    std::vector<std::vector<uint8_t>> grayscaleImage = this->convertToGrayscaleImage();
    // Apply histogram equilization to the grayscale image
    const int numRows = grayscaleImage.size();
    const int numCols = grayscaleImage[0].size();
    std::unordered_map<uint8_t, int> histogram;
    histogram.reserve(256); // Reserve space for 256 possible pixel values
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        histogram[grayscaleImage[r][c]]++;
      }
    }
    // --------------- BEGIN_CITATION [3] ---------------- //
    // https://hypjudy.github.io/2017/03/19/dip-histogram-equalization/
    // Transform histogram to cumulative histogram
    // T(r_k) = \sum_{j=0}^k P(r_j) \cdot L_2
    // P(r_j): The mass distribution of r_j
    // L2: The number of levels in the image (256 for 8-bit grayscale)
    const int L2 = 256;
    std::unordered_map<int, int> cumulativeHistogram;
    cumulativeHistogram.reserve(256); // Reserve space for 256 possible pixel values
    const int numPixels = numRows * numCols;
    int cumulativeSum = 0;
    for (int k = 0; k < 256; ++k) {
      // Transform from histogram to cumulative histogram
      // s_k = T(r_k) = round(cdf(r_k) * (L2 - 1))
      cumulativeSum += histogram[k];
      float cdf = static_cast<float>(cumulativeSum) / numPixels;
      cumulativeHistogram[k] = std::round(cdf * (L2 - 1)); // Scale to [0, L2-1]
    }
    // --------------- END_CITATION [3] ---------------- //
    // Apply the transformation to the grayscale image
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        // Apply the transformation to the pixel value
        grayscaleImage[r][c] = cumulativeHistogram[grayscaleImage[r][c]];
      }
    }
    // Copy the image into pixel data
    this->pixelData2D.resize(numRows, std::vector<uint8_t>(numCols * 3, 0)); // RGB image
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        // Set the RGB values in the pixel data (BGR order for BMP)
        this->pixelData2D[r][c * 3] = grayscaleImage[r][c];     // Blue
        this->pixelData2D[r][c * 3 + 1] = grayscaleImage[r][c]; // Green
        this->pixelData2D[r][c * 3 + 2] = grayscaleImage[r][c]; // Red
      }
    }
    // Update the BMP info header for 24-bit color depth
    this->infoHeader.bit_count = 24;
    int rowSize = ((this->infoHeader.bit_count * this->infoHeader.width + 31) / 32) * 4;
    this->infoHeader.size_image = rowSize * std::abs(this->infoHeader.height);
    this->infoHeader.colors_used = 0; // Not used for 24-bit images
    this->infoHeader.compression = 0; // No compression
    this->infoHeader.size = sizeof(BMPInfoHeader);
    // Update the file header offset to point to the new pixel data
    this->fileHeader.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    // Update the file size
    this->fileHeader.file_size = this->fileHeader.offset_data + this->infoHeader.size_image;
    // Save the histogram to a CSV file if requested
    if (saveToCSV) {
      const std::string HName = this->name + "_histogram.csv";
      const std::string CHName = this->name + "_cumulative_histogram.csv";
      std::ofstream csvFile(HName);
      std::ofstream csvFileC(CHName);
      if (csvFile.is_open() && csvFileC.is_open()) {
        for (const auto& entry : histogram) {
          csvFile << entry.first << "," << entry.second << "\n";
        }
        for (const auto& entry : cumulativeHistogram) {
          csvFileC << entry.first << "," << entry.second << "\n";
        }
        csvFile.close();
        csvFileC.close();
        std::cout << "Histogram saved to " << HName << std::endl;
        std::cout << "Cumulative histogram saved to " << CHName << std::endl;
      } else {
        std::cerr << "Unable to open histogram CSV file" << std::endl;
      }
    }
  }

  void lightingCorrection(bool linear = true) {
    // Convert 24-bit image to grayscale
    std::vector<std::vector<uint8_t>> grayscaleImage = this->convertToGrayscaleImage();
    // Find the min and max pixel values in the grayscale image
    uint8_t minPixelValue = 255;
    uint8_t maxPixelValue = 0;
    for (const auto& row : grayscaleImage) {
      for (const auto& pixel : row) {
        minPixelValue = std::min(minPixelValue, pixel);
        maxPixelValue = std::max(maxPixelValue, pixel);
      }
    }
    // Least-squares linear correction
    const float MAX_PIXEL_VALUE = 255.0f;
    float A[2];
    if (linear) { // Linear correction
      A[0] = static_cast<float>(maxPixelValue - minPixelValue) / MAX_PIXEL_VALUE; // Scale factor
      A[1] = static_cast<float>(minPixelValue); // Bias term
    } else { // Quadratic correction
      A[0] = static_cast<float>(maxPixelValue - minPixelValue) / (MAX_PIXEL_VALUE * MAX_PIXEL_VALUE); // Scale factor
      A[1] = static_cast<float>(minPixelValue); // Bias term
    }
    // Apply the correction to the grayscale image
    const int numRows = grayscaleImage.size();
    const int numCols = grayscaleImage[0].size();
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        // Apply the correction to the pixel value
        if (linear) {
          grayscaleImage[r][c] = std::min(static_cast<int>(A[0] * grayscaleImage[r][c] + A[1]), 255);
        } else {
          grayscaleImage[r][c] = std::min(static_cast<int>(A[0] * grayscaleImage[r][c] * grayscaleImage[r][c] + A[1]), 255);
        }
      }
    }
    // Copy the image into pixel data
    this->pixelData2D.resize(numRows, std::vector<uint8_t>(numCols * 3, 0)); // RGB image
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        // Set the RGB values in the pixel data (BGR order for BMP)
        this->pixelData2D[r][c * 3] = grayscaleImage[r][c];     // Blue
        this->pixelData2D[r][c * 3 + 1] = grayscaleImage[r][c]; // Green
        this->pixelData2D[r][c * 3 + 2] = grayscaleImage[r][c]; // Red
      }
    }
    // Update the BMP info header for 24-bit color depth
    this->infoHeader.bit_count = 24;
    int rowSize = ((this->infoHeader.bit_count * this->infoHeader.width + 31) / 32) * 4;
    this->infoHeader.size_image = rowSize * std::abs(this->infoHeader.height);
    this->infoHeader.colors_used = 0; // Not used for 24-bit images
    this->infoHeader.compression = 0; // No compression
    this->infoHeader.size = sizeof(BMPInfoHeader);
    // Update the file header offset to point to the new pixel data
    this->fileHeader.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    // Update the file size
    this->fileHeader.file_size = this->fileHeader.offset_data + this->infoHeader.size_image;
  }

  std::vector<std::pair<int, int>> selectRegion() {
    // Allow the user to select pixels on the image as a region of interest (ROI)
    std::cout << "Select a region of interest (ROI) by clicking on the image." << std::endl;
    std::cout << "Press 'q' to finish selecting." << std::endl;
    // Avoid duplicate pixel selection by using a set
    std::vector<std::pair<int, int>> selectedPixels;
    cv::Mat originalImage = this->convertPixelToMat();
    cv::Mat displayImage;
    originalImage.copyTo(displayImage); // Copy the original image to display
    // Struct to maintain the state as we move the mouse around
    struct MouseState {
      bool isDragging = false;
      std::vector<std::pair<int, int>>* selectedPixels;
      cv::Mat* display;
    } mouseState = {false, &selectedPixels, &displayImage};
    const std::string windowName = "Select ROI for " + this->name;
    // Open a window to display the image
    cv::namedWindow(windowName, cv::WINDOW_KEEPRATIO);
    cv::setMouseCallback(windowName, [](int event, int x, int y, [[maybe_unused]] int flags, void* param) {
      auto* state = static_cast<MouseState*>(param);
      if (event == cv::EVENT_LBUTTONDOWN) {
        state->isDragging = true; // Start dragging
      } else if (event == cv::EVENT_LBUTTONUP) {
        state->isDragging = false; // Stop dragging
      } else if (event == cv::EVENT_MOUSEMOVE && state->isDragging) {
        // Avoid duplicate pixel selection
        if (std::find(state->selectedPixels->cbegin(), state->selectedPixels->cend(),
          std::make_pair(y, x)) != state->selectedPixels->cend()) {
          return; // Pixel already selected
        }
        state->selectedPixels->emplace_back(y, x); // Store the selected pixel
        // Draw a circle on the selected pixel
        cv::circle(*state->display, cv::Point(x, y), 0, cv::Scalar(0, 255, 0), -1);
        // std::cout << "Selected pixel: (" << y << ", " << x << ")" << std::endl;
      }
    }, &mouseState);
    // Wait for user input
    // cv::resizeWindow(windowName, 800, 600); // Make it big so the pixels are visible
    while (true) {
      cv::imshow(windowName, displayImage);
      char key = cv::waitKey(10);
      if (key == 'q') {
        break; // Exit the loop when 'q' is pressed
      }
    }
    cv::destroyWindow(windowName); // Close the window
    std::cout << "Selected region size: " << selectedPixels.size() << " Pixels" << std::endl;
    return selectedPixels;
  }

  void createHistogramFromFile(const std::string filename, const ColorSpace colorSpace = ColorSpace::BGR) {
    // The file passed in is a txt file containing pixel coordinates
    // The pixel values correspond to the pixel values in the original image
    // that were selected by the user
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
    }
    std::cout << "Creating histogram from file: " << filename << std::endl;
    std::vector<std::pair<int, int>> selectedPixels;
    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      int x, y;
      if (iss >> y >> x) {
        selectedPixels.emplace_back(y, x); // Store the selected pixel
      }
    }
    file.close();
    std::cout << "Found " << selectedPixels.size() << " selected pixels" << std::endl;
    // Construct a 2D color histogram based on the
    // color pixels you have collected (R-G, NR-NG, H-S)
    this->changeColorSpace(colorSpace);
    // Images are 24-bit color so we have access to 3 channels
    std::vector<std::vector<int>> histogram(256, std::vector<int>(3, 0)); // 3 channels (R, G, B)
    for (const auto& pixel : selectedPixels) {
      switch (this->colorSpace) {
      case ColorSpace::BGR: case ColorSpace::NBGR: {
        int b = this->pixelData2D[pixel.first][pixel.second * 3 + 0]; // Blue
        int g = this->pixelData2D[pixel.first][pixel.second * 3 + 1]; // Green
        int r = this->pixelData2D[pixel.first][pixel.second * 3 + 2]; // Red
        histogram[r][0]++;
        histogram[g][1]++;
        histogram[b][2]++;
        break;
      }
      case ColorSpace::HSI: {
        int h = this->pixelData2D[pixel.first][pixel.second * 3 + 0]; // Hue
        int s = this->pixelData2D[pixel.first][pixel.second * 3 + 1]; // Saturation
        int i = this->pixelData2D[pixel.first][pixel.second * 3 + 2]; // Intensity
        histogram[h][0]++;
        histogram[s][1]++;
        histogram[i][2]++;
        break;
      }
      }
    }
    // Save the histogram to a CSV file
    const std::string HName = this->name + "_" + this->colorSpaceName(this->colorSpace) + "_histogram.csv";
    std::ofstream csvFile(HName);
    if (csvFile.is_open()) {
      for (int i = 0; i < 256; ++i) {
        csvFile << i << "," << histogram[i][0] << "," << histogram[i][1] << "," << histogram[i][2] << "\n";
      }
      csvFile.close();
      std::cout << "Histogram saved to " << HName << std::endl;
    } else {
      std::cerr << "Unable to open histogram CSV file" << std::endl;
    }
    std::cout << "2D Color Histogram saved to " << HName << std::endl;
  }

  void thresholdFromHistogram(const std::string histogramFile, const int threshold) {
    // Given a histogram (combined histogram)
    // Use a threshold parameter to block out pixels that don't pass
    // Save the resulting image to a new file
    std::ifstream file(histogramFile);
    if (!file.is_open()) {
      std::cerr << "Error opening file: " << histogramFile << std::endl;
      return;
    }
    std::cout << "Thresholding from histogram file: " << histogramFile << std::endl;
    // File contains: Pixel Value, Value1, Value2, Value3 (either BGR or HSI)
    // Pixel Value: 0-255
    const std::string colorSpace = histogramFile.substr(histogramFile.find("_") + 1, histogramFile.find("_histogram.csv") - histogramFile.find("_") - 1);
    std::vector<std::vector<int>> histogram(256, std::vector<int>(3, 0));
    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::vector<std::string> tokens;
      std::string token;
      while (std::getline(iss, token, ',')) {
        tokens.push_back(token);
      }
      if (tokens.size() != 4) {
        std::cerr << "Invalid line format: " << line << std::endl;
        continue;
      }
      // Parse the line
      int index = std::stoi(tokens[0]);
      int value1 = std::stoi(tokens[1]);
      int value2 = std::stoi(tokens[2]);
      int value3 = std::stoi(tokens[3]);
      // Create the histogram
      histogram[index][0] += value1;
      histogram[index][1] += value2;
      histogram[index][2] += value3;
    }
    file.close();
    std::cout << "Successfully read Histogram from " << histogramFile << std::endl;
    std::vector<std::pair<int, int>> thresholdedPixels;
    // Convert the pixel data to the color space of the histogram
    this->changeColorSpace(colorSpace == "BGR" ? ColorSpace::BGR : ColorSpace::HSI);
    const int numRows = std::abs(this->infoHeader.height);
    const int numCols = this->infoHeader.width;
    // Threshold the histogram
    for (int r = 0; r < numRows; r++) {
      for (int c = 0; c < numCols; c++) {
        int value1 = this->pixelData2D[r][c * 3 + 0]; // Blue or Hue
        int value2 = this->pixelData2D[r][c * 3 + 1]; // Green or Saturation
        int value3 = this->pixelData2D[r][c * 3 + 2]; // Red or Intensity
        if (histogram[value1][0] < threshold || histogram[value2][1] < threshold || histogram[value3][2] < threshold) {
          thresholdedPixels.emplace_back(r, c);
        }
      }
    }
    std::cout << "Found " << thresholdedPixels.size() << " pixels below threshold of " << threshold << std::endl;
    // Convert Image back to BGR if needed
    if (this->colorSpace == ColorSpace::HSI) {
      this->changeColorSpace(ColorSpace::BGR);
    }
    // Black out pixels that were thresholded
    for (const auto& pixel : thresholdedPixels) {
      this->pixelData2D[pixel.first][pixel.second * 3 + 0] = 0; // Blue
      this->pixelData2D[pixel.first][pixel.second * 3 + 1] = 0; // Green
      this->pixelData2D[pixel.first][pixel.second * 3 + 2] = 0; // Red
    }
    // Save the resulting image
    std::string outputFilename = this->name + "_thresholded.bmp";
    this->write(outputFilename.c_str());
  }

  void changeColorSpace(const ColorSpace colorSpace) {
    if (colorSpace == this->colorSpace) { return; } // No change
    // Convert the pixel data to the new color space
    std::cout << "Converting " << this->name << " from " << this->colorSpaceName(this->colorSpace) <<
      " to " << this->colorSpaceName(colorSpace) << std::endl;
    const int numRows = std::abs(this->infoHeader.height);
    const int numCols = this->infoHeader.width;
    switch (colorSpace) {
    case ColorSpace::BGR: {
      // Convert HSI to BGR
      const float RAD_60 = M_PI / 3.0f;
      const float RAD_120 = 2.0f * M_PI / 3.0f;
      const float RAD_180 = M_PI;
      const float RAD_240 = 4.0f * M_PI / 3.0f;
      const float RAD_300 = 5.0f * M_PI / 3.0f;
      const float RAD_360 = 2.0f * M_PI;
      for (int r = 0; r < numRows; r++) {
        for (int c = 0; c < numCols; c++) {
          int H = this->pixelData2D[r][c * 3 + 0]; // Hue
          int S = this->pixelData2D[r][c * 3 + 1]; // Saturation
          int I = this->pixelData2D[r][c * 3 + 2]; // Intensity
          // Convert Each value to their corresponding ranges
          // Hue: [0, 255] -> [0, 360]
          // Saturation: [0, 255] -> [0, 1]
          // Intensity: [0, 255] -> [0, 255]
          float Hf = std::fmod(static_cast<float>(H), 360) * (360.0f / 255.0f); // Hue [0, 360]
          float Sf = static_cast<float>(S) / 255.0f; // Saturation [0, 1]
          float If = static_cast<float>(I); // Intensity [0, 255]
          Hf *= (M_PI / 180.0f); // Convert to radians
          // Convert HSI to RGB
          float R = If + 2 * If * Sf;
          float G = If - If * Sf;
          float B = If - If * Sf;
          if (Hf < RAD_120) {
            R = If + If * Sf * std::cos(Hf) / std::cos(RAD_60 - Hf);
            G = If + If * Sf * (1 - std::cos(Hf) / std::cos(RAD_60 - Hf));
            B = If - If * Sf;
          } else if (RAD_120 < Hf && Hf < RAD_240) {
            R = If - If * Sf;
            G = If + If * Sf * std::cos(Hf - RAD_120) / std::cos(RAD_180 - Hf);
            B = If + If * Sf * (1 - std::cos(Hf - RAD_120) / std::cos(RAD_180 - Hf));
          } else if (RAD_240 < Hf && Hf < RAD_360) {
            R = If + If * Sf * (1 - std::cos(Hf - RAD_240) / std::cos(RAD_300 - Hf));
            G = If - If * Sf;
            B = If + If * Sf * std::cos(Hf - RAD_240) / std::cos(RAD_300 - Hf);
          }
          // Clamp RGB values to [0, 255]
          R = std::min(std::max(R, 0.0f), 255.0f);
          G = std::min(std::max(G, 0.0f), 255.0f);
          B = std::min(std::max(B, 0.0f), 255.0f);
          // Assign the RGB values to the pixel data
          this->pixelData2D[r][c * 3 + 0] = static_cast<uint8_t>(B); // Blue
          this->pixelData2D[r][c * 3 + 1] = static_cast<uint8_t>(G); // Green
          this->pixelData2D[r][c * 3 + 2] = static_cast<uint8_t>(R); // Red
        }
      }
      break;
    }
    case ColorSpace::NBGR: {
      for (int r = 0; r < numRows; r++) {
        for (int c = 0; c < numCols; c++) {
          // Convert BGR to NBGR
          int B = this->pixelData2D[r][c * 3 + 0]; // Blue
          int G = this->pixelData2D[r][c * 3 + 1]; // Green
          int R = this->pixelData2D[r][c * 3 + 2]; // Red
          // Normalize RGB values
          float sum = B + G + R + 1e-6f; // Avoid division by zero
          this->pixelData2D[r][c * 3 + 0] = static_cast<uint8_t>((B / sum) * 255); // N-Red
          this->pixelData2D[r][c * 3 + 1] = static_cast<uint8_t>((G / sum) * 255); // N-Green
          this->pixelData2D[r][c * 3 + 2] = static_cast<uint8_t>((R / sum) * 255); // N-Blue
        }
      }
      break;
    }
    case ColorSpace::HSI: {
      // BGR -> HSI
      for (int r = 0; r < numRows; r++) {
        for (int c = 0; c < numCols; c++) {
          // Convert BGR to HSI
          const int B = this->pixelData2D[r][c * 3 + 0]; // Blue
          const int G = this->pixelData2D[r][c * 3 + 1]; // Green
          const int R = this->pixelData2D[r][c * 3 + 2]; // Red
          // --------------- BEGIN_CITATION [4] ---------------- //
          // https://answers.opencv.org/question/62446/conversion-from-rgb-to-hsi/
          float numerator = R - 0.5f * ((R - G) + (R - B));
          float denominator = std::sqrt(((R - G) * (R - G)) + ((R - B) * (G - B)));
          const float EPSILON = 1e-6f; // Small value to avoid division by zero
          float theta = std::acos(numerator / (denominator + EPSILON)); // [rad]
          const int sum = B + G + R;
          float I = sum / 3.0f; // Intensity [0, 255]
          float S = (I == 0) ? 0.0f : 1.0f - 3.0f * (std::min({R, G, B}) / (sum + EPSILON)); // Saturation [0, 1]
          float H = (G >= B) ? theta : ((2.0f * M_PI) - theta); // Hue [0, pi] or [pi, 2pi]
          // Convert Hue to Degrees
          H *= (180.0f / M_PI); // Hue [0, 360]
          // Normalize Hue to be between [0, 255]
          H *= (255.0f / 360.0f); // Hue [0, 255]
          // Normalize Saturation to be between [0, 255]
          S *= 255.0f; // Saturation [0, 255]
          // --------------- END_CITATION [4] ---------------- //
          // Assign the HSI values to the pixel data
          this->pixelData2D[r][c * 3 + 0] = static_cast<uint8_t>(H);
          this->pixelData2D[r][c * 3 + 1] = static_cast<uint8_t>(S);
          this->pixelData2D[r][c * 3 + 2] = static_cast<uint8_t>(I);
        }
      }
      break;
    }
    default: {
      throw std::runtime_error("Unsupported color space");
    }
    }// END switch
    std::cout << "Converted image to " << this->colorSpaceName(colorSpace) << " color space" << std::endl;
    this->colorSpace = colorSpace; // Update the color space
  }

  std::vector<std::vector<uint8_t>> gaussianSmoothing(const float sigma = 1.0f) {
    // Use a Gaussian filter to smooth the grayscale image.
    // First convert the image to grayscale
    std::vector<std::vector<uint8_t>> grayscaleImage = this->convertToGrayscaleImage();
    // Apply Gaussian smoothing using a 5x5 kernel
    StructuringElement kernel(5);
    kernel.initGaussianKernel(sigma);
    std::vector<std::vector<uint8_t>> smoothedImage = this->convolution(grayscaleImage, kernel);
    std::cout << "Applied Gaussian Smoothing (5x5 kernel) using sigma = " << sigma << std::endl;
    // Return the smoothed image
    return smoothedImage;
  }

  void cannyEdgeDetector(float sigma, float percentNonEdge, std::string suppressionMethod, std::string gradientMethod) {
    auto smoothed = this->gaussianSmoothing(sigma);
    auto grad = this->imageGradient(smoothed, gradientMethod);
    auto suppressedGradient = this->nonMaximaSuppression(grad, suppressionMethod);
    std::vector<std::vector<bool>> edgeMap = this->hysteresisThresholding(suppressedGradient, percentNonEdge);
    // Using the edge map, black out pixels that are not edges
    // and create a new image with the edges highlighted
    const int numRows = smoothed.size();
    const int numCols = smoothed[0].size();
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        if (edgeMap[r][c]) {
          this->pixelData2D[r][c * 3 + 0] = 255;     // Blue
          this->pixelData2D[r][c * 3 + 1] = 255;     // Green
          this->pixelData2D[r][c * 3 + 2] = 255;     // Red
        } else {
          this->pixelData2D[r][c * 3 + 0] = 0; // Black out pixel
          this->pixelData2D[r][c * 3 + 1] = 0;
          this->pixelData2D[r][c * 3 + 2] = 0;
        }
      }
    }
    // Update the BMP info header for 24-bit color depth
    this->infoHeader.bit_count = 24;
    int rowSize = ((this->infoHeader.bit_count * this->infoHeader.width + 31) / 32) * 4;
    this->infoHeader.size_image = rowSize * std::abs(this->infoHeader.height);
    this->infoHeader.colors_used = 0; // Not used for 24-bit images
    this->infoHeader.compression = 0; // No compression
    this->infoHeader.size = sizeof(BMPInfoHeader);
    // Update the file header offset to point to the new pixel data
    this->fileHeader.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    // Update the file size
    this->fileHeader.file_size = this->fileHeader.offset_data + this->infoHeader.size_image;
  }

  void houghTransform(const int threshold, const int lineLength) {
    // Assume Canny edge detection has already been applied
    // The image is 24-bit and edges are highlighted
    // Edges are white (255, 255, 255) and non-edges are black (0, 0, 0)
    const int numRows = this->pixelData2D.size();
    const int numCols = this->pixelData2D[0].size();
    // Create a Hough accumulator
    const int numAngles = 180; // Number of angles to consider [0, 180]
    const int numRho = std::hypot(numRows, numCols); // Maximum rho value
    std::vector <std::vector<int>> accumulator(2 * numRho, std::vector<int>(numAngles, 0));
    for (int r = 0; r < numRows; r++) {
      for (int c = 0; c < numCols; c++) {
        uint8_t blue = this->pixelData2D[r][c * 3 + 0]; // Blue
        uint8_t green = this->pixelData2D[r][c * 3 + 1]; // Green
        uint8_t red = this->pixelData2D[r][c * 3 + 2]; // Red
        if (blue != 0 && green != 0 && red != 0) { // Edge Pixel
          // Accummulate the rho values for each angle
          for (int theta = 0; theta < numAngles; theta++) {
            float rad = theta * M_PI / numAngles; // Convert to radians
            int rho = static_cast<int>(c * std::cos(rad) + r * std::sin(rad)); // Rho value
            rho += numRho; // Shift rho to positive range
            // Accumulate rho if within bounds
            if (rho >= 0 && rho < 2 * numRho) { accumulator[rho][theta]++; }
          }
        }
      }
    }

    // Find the peaks in the accumulator
    std::vector<std::pair<int, int>> lines; // (rho, theta)
    for (int rho = 0; rho < numRho; rho++) {
      for (int theta = 0; theta < numAngles; theta++) {
        if (accumulator[rho][theta] > threshold) {
          lines.emplace_back(rho, theta); // Store the peak
        }
      }
    }

    // For each (rho, theta), draw a line on the original image
    for (const auto& line : lines) {
      int rho = line.first;
      int theta = line.second;
      float rad = theta * M_PI / 180.0f; // Convert to radians

      // Calculate the endpoints of the line
      float cosRho = std::cos(rad);
      float sinRho = std::sin(rad);
      int x0 = static_cast<int>(rho * cosRho); // X coordinate
      int y0 = static_cast<int>(rho * sinRho); // Y coordinate
      int x1 = static_cast<int>(x0 + lineLength * -sinRho); // X coordinate of endpoint 1
      int y1 = static_cast<int>(y0 + lineLength * cosRho); // Y coordinate of endpoint 1

      // Draw line on original image
      // interpolate between the two points and fill in nearest pixels
      for (int i = -lineLength / 2; i <= lineLength / 2; ++i) {
        int lineX = static_cast<int>(x0 + i * -sinRho);
        int lineY = static_cast<int>(y0 + i * cosRho);
        // Check bounds before drawing
        if (lineX >= 0 && lineX < numCols && lineY >= 0 && lineY < numRows) {
          this->pixelData2D[lineY][lineX * 3 + 0] = 255; // Blue
          this->pixelData2D[lineY][lineX * 3 + 1] = 255;   // Green
          this->pixelData2D[lineY][lineX * 3 + 2] = 255;   // Red
        }
      }
    }
    std::cout << "Hough Transform found " << lines.size() << " lines" << std::endl;

    // // Update the BMP info header for 24-bit color depth
    // this->infoHeader.bit_count = 24;
    // int rowSize = ((this->infoHeader.bit_count * this->infoHeader.width + 31) / 32) * 4;
    // this->infoHeader.size_image = rowSize * std::abs(this->infoHeader.height);
    // this->infoHeader.colors_used = 0; // Not used for 24-bit images
    // this->infoHeader.compression = 0; // No compression
    // this->infoHeader.size = sizeof(BMPInfoHeader);
    // // Update the file header offset to point to the new pixel data
    // this->fileHeader.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    // // Update the file size
    // this->fileHeader.file_size = this->fileHeader.offset_data + this->infoHeader.size_image;
  }

private:
  BMPFileHeader fileHeader;
  BMPInfoHeader infoHeader;
  BMPColorHeader colorHeader;
  std::vector<std::vector<uint8_t>> pixelData2D; // 2D representation of pixel data
  std::string name;
  std::vector<Component> components;
  ColorSpace colorSpace = ColorSpace::BGR; // Default color space for BMP

  std::vector<std::vector<ImageGradient>> imageGradient(const std::vector<std::vector<uint8_t>>& image, std::string kernel = "Sobel") {
    // Compute image gradient on the given image
    StructuringElement GX(2);
    StructuringElement GY(2);
    if (kernel == "Sobel") {
      GX.initSobelKernel(true);
      GY.initSobelKernel(false);
    } else if (kernel == "RobertCross") {
      GX.initRobertCrossKernel(true);
      GY.initRobertCrossKernel(false);
    } else {
      throw std::runtime_error("Unsupported kernel type: " + kernel);
    }
    std::cout << "Using " << kernel << " kernel for gradient computation" << std::endl;
    std::vector<std::vector<ImageGradient>> imageGradient;
    const int numRows = image.size();
    const int numCols = image[0].size();
    imageGradient.resize(numRows, std::vector<ImageGradient>(numCols));
    float gradMax = 0.0f;
    for (int r = 0; r < numRows - 1; r++) {
      for (int c = 0; c < numCols - 1; c++) {
        // Gx = I(i,j) - I(i+1,j+1)
        // Gy = I(i + 1, j) - I(i, j + 1)
        if (kernel == "Sobel") {
          if (r == 0 || c == 0) { continue; } // Skip r and c == 0
          double gx = (image[r - 1][c + 1] - image[r - 1][c - 1]) +
            2 * (image[r][c + 1] - image[r][c - 1]) +
            (image[r + 1][c + 1] - image[r + 1][c - 1]);
          double gy = (image[r + 1][c - 1] - image[r - 1][c - 1]) +
            2 * (image[r + 1][c] - image[r - 1][c]) +
            (image[r + 1][c + 1] - image[r - 1][c + 1]);
          imageGradient[r - 1][c - 1].magnitude = std::sqrt(gx * gx + gy * gy);
          imageGradient[r - 1][c - 1].direction = std::atan2(gy, gx); // [rad]
          gradMax = std::max(gradMax, imageGradient[r - 1][c - 1].magnitude);
        } else if (kernel == "RobertCross") {
          double gx = image[r][c] - image[r + 1][c + 1];
          double gy = image[r + 1][c] - image[r][c + 1];
          imageGradient[r][c].magnitude = std::sqrt(gx * gx + gy * gy);
          imageGradient[r][c].direction = std::atan2(gy, gx); // [rad]
          gradMax = std::max(gradMax, imageGradient[r][c].magnitude);
        }
      }
    }

    // Normalize Gradient
    for (int r = 0; r < numRows; r++) {
      for (int c = 0; c < numCols; c++) {
        imageGradient[r][c].magnitude /= gradMax; // [0, 1]
        imageGradient[r][c].magnitude *= 255.0f; // [0, 255]
      }
    }

    return imageGradient;
  }

  std::vector<std::vector<bool>> hysteresisThresholding(
    const std::vector<std::vector<ImageGradient>>& imageGradient,
    float percentNonEdge) {
    // Find the threshold values for edge linking
    // Calculate high threshold and low will be extrapolated as 0.5 * highThreshold
    // percentNonEdge is the specified percentage of Non-edge area in magnitudes
    const int numRows = imageGradient.size();
    const int numCols = imageGradient[0].size();
    std::vector<float> magnitudes;
    magnitudes.reserve(numRows * numCols);
    for (const auto& row : imageGradient) {
      for (const auto& pixel : row) {
        magnitudes.push_back(pixel.magnitude);
      }
    }
    // Sort the magnitudes in descending order
    std::sort(magnitudes.begin(), magnitudes.end(), std::greater<float>());
    const int numPixels = magnitudes.size();
    // Find threshold based on the percentage of non-edge pixels
    const int highThresholdIndex = static_cast<int>(numPixels * (1.0f - percentNonEdge));
    const float highThreshold = magnitudes[std::min(highThresholdIndex, numPixels - 1)];
    const float lowThreshold = highThreshold * 0.5f;
    std::cout << "Hysteresis Thresholding using " << (percentNonEdge * 100) << "% non-edge pixels" << std::endl;
    std::cout << "Low: " << lowThreshold << " High: " << highThreshold << std::endl;
    // Build strong and weak edge maps
    std::vector<std::vector<bool>> strong(numRows, std::vector<bool>(numCols, false));
    std::vector<std::vector<bool>> weak(numRows, std::vector<bool>(numCols, false));
    for (int r = 0; r < numRows; r++) {
      for (int c = 0; c < numCols; c++) {
        if (imageGradient[r][c].magnitude >= highThreshold) {
          strong[r][c] = true; // Strong edge
        } else if (imageGradient[r][c].magnitude >= lowThreshold) {
          weak[r][c] = true; // Weak edge
        }
      }
    }
    // Trace edges to create final edge map
    return this->edgeLinking(strong, weak);
  }

  std::vector<std::vector<bool>> edgeLinking(
    const std::vector<std::vector<bool>>& strong,
    const std::vector<std::vector<bool>>& weak) {
    std::vector<std::vector<bool>> linkedEdges = strong;
    // Link strong and weak edges
    const int numRows = strong.size();
    const int numCols = strong[0].size();
    for (int r = 1; r < numRows - 1; r++) {
      for (int c = 1; c < numCols - 1; c++) {
        if (strong[r][c]) {
          // Check 8-connected neighbors
          for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
              if (i < 0 || j < 0 || i > numRows - 1 || j > numCols - 1) {
                continue; // Skip out of bounds
              } else if (weak[r + i][c + j] && !linkedEdges[r + i][c + j]) {
                linkedEdges[r + i][c + j] = true; // Link weak edge to strong edge
              }
            }
          }
        }
      }
    }
    return linkedEdges;
  }

  std::vector<std::vector<ImageGradient>> nonMaximaSuppression(
    const std::vector<std::vector<ImageGradient>>& imageGradient,
    std::string method = "interpolation") {
    const int numRows = imageGradient.size();
    const int numCols = imageGradient[0].size();
    std::vector<std::vector<ImageGradient>> suppressed(numRows, std::vector<ImageGradient>(numCols));
    for (int r = 1; r < numRows - 1; r++) {
      for (int c = 1; c < numCols - 1; c++) {
        // Get the gradient direction and magnitude
        float magnitude = imageGradient[r][c].magnitude;
        if (magnitude < 1e-6f) { continue; } // Skip if magnitude is too small
        float direction = imageGradient[r][c].direction;
        // Normalize direction between [0, pi]
        while (direction < 0) { direction += M_PI; }
        while (direction > M_PI) { direction -= M_PI; }
        if (method == "interpolation") {
          // Calculate unit vector in the direction of the gradient
          float x = std::cos(direction);
          float y = std::sin(direction);
          float x1 = static_cast<float>(c) + x;
          float y1 = static_cast<float>(r) + y;
          float x2 = static_cast<float>(c) - x;
          float y2 = static_cast<float>(r) - y;
          float mag1 = this->bilinearInterpolation(imageGradient, y1, x1);
          float mag2 = this->bilinearInterpolation(imageGradient, y2, x2);
          if (magnitude >= mag1 && magnitude >= mag2) {
            suppressed[r][c].magnitude = magnitude; // Keep the pixel
            suppressed[r][c].direction = direction; // Keep the direction
          }
        } else if (method == "quantized") {
          // Quantized directions: 0 = 0, 1 = 45, 2 = 90, 3 = 135 degrees
          int quantizedDirection = static_cast<int>(std::round(direction / (M_PI / 4))) % 4;
          // Get neighbors based on quantized direction
          int nx1, ny1, nx2, ny2;
          switch (quantizedDirection) {
          case 0: { // 0 degrees
            nx1 = c + 1; ny1 = r; // Right
            nx2 = c - 1; ny2 = r; // Left
            break;
          }
          case 1: { // 45 degrees
            nx1 = c + 1; ny1 = r - 1; // Top-right
            nx2 = c - 1; ny2 = r + 1; // Bottom-left
            break;
          }
          case 2: { // 90 degrees
            nx1 = c; ny1 = r - 1; // Top
            nx2 = c; ny2 = r + 1; // Bottom
            break;
          }
          case 3: { // 135 degrees
            nx1 = c - 1; ny1 = r - 1; // Top-left
            nx2 = c + 1; ny2 = r + 1; // Bottom-right
            break;
          }
          }
          // Compare magnitudes
          float mag1 = imageGradient[ny1][nx1].magnitude;
          float mag2 = imageGradient[ny2][nx2].magnitude;
          if (magnitude >= mag1 && magnitude >= mag2) {
            suppressed[r][c].magnitude = magnitude; // Keep the pixel
            suppressed[r][c].direction = direction; // Keep the direction
          }
        } else {
          throw std::runtime_error("Unsupported non-maxima suppression method: " + method);
        }
      }
    }
    return suppressed;
  }

  float bilinearInterpolation(const std::vector<std::vector<ImageGradient>>& imageGradient,
    const float y, const float x) {
    // Verify bounds
    const int numRows = imageGradient.size();
    const int numCols = imageGradient[0].size();
    // Out of bounds returns 0
    if (y < 0 || y >= numRows || x < 0 || x >= numCols) { return 0.0f; }
    // Get 4 neighbors
    int xFloor = static_cast<int>(std::floor(x));
    int yFloor = static_cast<int>(std::floor(y));
    int xCeil = static_cast<int>(std::ceil(x));
    int yCeil = static_cast<int>(std::ceil(y));
    // Interpolation weights
    float wx = x - xFloor;
    float wy = y - yFloor;
    // Collect magnitudes
    float mag11 = imageGradient[yFloor][xFloor].magnitude; // Top-left
    float mag12 = imageGradient[yFloor][xCeil].magnitude; // Top-right
    float mag21 = imageGradient[yCeil][xFloor].magnitude; // Bottom-left
    float mag22 = imageGradient[yCeil][xCeil].magnitude; // Bottom-right
    // --------------- BEGIN_CITATION [5] ---------------- //
    // Bilinear interpolation
    // https://arm-software.github.io/CMSIS-DSP/v1.10.1/group__BilinearInterpolate.html
    // f(x, y) = f(XF, YF) * (1 - (x - XF)) * (1 - (y - YF))
    //   + f(XF + 1, YF) * (x - XF) * (1 - (y - YF))
    //   + f(XF, YF + 1) * (1 - (x - XF)) * (y - YF)
    //   + f(XF + 1, YF + 1) * (x - XF) * (y - YF)
    float interpolatedValue = mag11 * (1 - wx) * (1 - wy) +
      mag12 * wx * (1 - wy) +
      mag21 * (1 - wx) * wy +
      mag22 * wx * wy;
    // --------------- END_CITATION [5] ---------------- //
    // Normalize the interpolated value to [0, 255]
    return std::clamp(interpolatedValue, 0.0f, 255.0f);
  }

  cv::Mat convertPixelToMat() {
    // Convert 2D Pixel Data to cv::Mat type for displaying
    const int numRows = std::abs(this->infoHeader.height);
    const int numCols = this->infoHeader.width;
    cv::Mat image(numRows, numCols, CV_8UC3);
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        // Convert pixel data to BGR format
        image.at<cv::Vec3b>(r, c)[2] = this->pixelData2D[r][c * 3 + 2]; // Blue
        image.at<cv::Vec3b>(r, c)[1] = this->pixelData2D[r][c * 3 + 1]; // Green
        image.at<cv::Vec3b>(r, c)[0] = this->pixelData2D[r][c * 3 + 0]; // Red
      }
    }
    return image;
  }

  int findRoot(std::vector<int>& parent, int label) {
    // Path compression: make every examined node point directly to the root
    if (parent[label] != label) {
      parent[label] = findRoot(parent, parent[label]);
    }
    return parent[label];
  }

  void unionLabels(std::vector<int>& parent, int label1, int label2) {
    int root1 = findRoot(parent, label1);
    int root2 = findRoot(parent, label2);

    if (root1 != root2) {
      // Make the smaller label the parent of the larger one
      // This helps maintain a flatter tree structure
      if (root1 < root2) {
        parent[root2] = root1;
      } else {
        parent[root1] = root2;
      }
    }
  }

  std::vector<std::vector<int>> convertToBinaryImage() {
    std::vector<std::vector<int>> binaryImage;
    const int numRows = std::abs(this->infoHeader.height);
    const int numCols = this->infoHeader.width;
    binaryImage.resize(numRows, std::vector<int>(numCols, 0));
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        binaryImage[r][c] = this->pixelData2D[r][c] == 0 ? 0 : 1;
      }
    }
    return binaryImage;
  }

  std::vector<std::vector<uint8_t>> convertToGrayscaleImage() {
    std::vector<std::vector<uint8_t>> grayscaleImage;
    const int numRows = std::abs(this->infoHeader.height);
    const int numCols = this->infoHeader.width;
    grayscaleImage.resize(numRows, std::vector<uint8_t>(numCols, 0));
    for (int r = 0; r < numRows; ++r) {
      for (int c = 0; c < numCols; ++c) {
        // Convert RGB to grayscale using the luminosity method
        int rValue = this->pixelData2D[r][c * 3 + 2]; // Red
        int gValue = this->pixelData2D[r][c * 3 + 1]; // Green
        int bValue = this->pixelData2D[r][c * 3 + 0]; // Blue
        grayscaleImage[r][c] = BMPImage::rgbToGrayScale(rValue, gValue, bValue);
      }
    }
    return grayscaleImage;
  }

  inline static uint8_t rgbToGrayScale(const int r, const int g, const int b) {
    // --------------- BEGIN_CITATION [2] ---------------- //
    // https://www.grayscaleimage.com/three-algorithms-for-converting-color-to-grayscale/
    // Luminosity method
    float grayscale = 0.299f * r + 0.587f * g + 0.114f * b;
    return static_cast<uint8_t>(std::clamp(grayscale, 0.0f, 255.0f)); // [0, 255]
    // --------------- END_CITATION [2] ---------------- //
  }

  std::string colorSpaceName(const ColorSpace colorSpace) const {
    switch (colorSpace) {
    case ColorSpace::BGR: { return "BGR"; }
    case ColorSpace::HSI: { return "HSI"; }
    case ColorSpace::NBGR: { return "NBGR"; }
    default: { return "Unknown"; }
    }
  }

  std::string getImageName(const char* filename) {
    // Extract the image name from the filename
    // "test.bmp" -> "test"
    std::string fname = filename;
    size_t pos = fname.find_last_of('/');
    if (pos != std::string::npos) {
      fname = fname.substr(pos + 1);
    }
    pos = fname.find_last_of('.');
    if (pos != std::string::npos && fname.substr(pos) == ".bmp") {
      fname = fname.substr(0, pos);
    }
    return fname;
  }

  void read(const char* filename) {
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Could not open file" + std::string(filename));
    }

    // Read File Header
    file.read(reinterpret_cast<char*>(&this->fileHeader), sizeof(this->fileHeader));
    if (this->fileHeader.file_type != BMFILETYPE) {
      throw std::runtime_error(std::string(filename) + " is not a BMP file");
    }

    // Read Info Header
    file.read(reinterpret_cast<char*>(&this->infoHeader), sizeof(this->infoHeader));

    // Validate that image is uncompressed (Only 0 is supported)
    if (this->infoHeader.compression != 0) {
      throw std::runtime_error(std::string(filename) + " is not an uncompressed BMP file");
    }

    // If 32 bits per pixel, read the color header as well
    // 24-bit images do not have a color header
    // 1-bit images have a color table instead of a color header
    if (this->infoHeader.bit_count == 32) {
      file.read(reinterpret_cast<char*>(&this->colorHeader), sizeof(this->colorHeader));
    } else if (this->infoHeader.bit_count <= 8) {
      int colorTableEntries = 1 << this->infoHeader.bit_count;
      if (this->infoHeader.colors_used > 0) {
        colorTableEntries = this->infoHeader.colors_used;
      }
      // Skip color table
      file.seekg(colorTableEntries * sizeof(uint32_t), std::ios::cur);
    }

    // Move file pointer to beginning of pixel data
    file.seekg(this->fileHeader.offset_data, std::ios::beg);

    // The row size must be a multiple of 4 bytes
    const int rowSize = ((this->infoHeader.bit_count * this->infoHeader.width + 31) / 32) * 4;
    // For safety, use absolute height since BMP height can be negative
    const int imageSize = rowSize * std::abs(this->infoHeader.height);

    // Determine image size
    if (this->infoHeader.size_image == 0) {
      this->infoHeader.size_image = imageSize;
    }

    // Read the pixel data
    std::vector<uint8_t> pixelData(this->infoHeader.size_image);
    file.read(reinterpret_cast<char*>(pixelData.data()), this->infoHeader.size_image);
    if (!file) {
      throw std::runtime_error("Error reading pixel data from " + std::string(filename));
    }
    std::cout << "Successfully read " << imageSize << " bytes from " << filename << std::endl;

    // Convert pixel data to 2D array matching image dimensions
    const int bytesPerRow = this->infoHeader.bit_count == 24 ? 3 * this->infoHeader.width : this->infoHeader.width;
    this->pixelData2D.clear();
    this->pixelData2D.resize(std::abs(this->infoHeader.height), std::vector<uint8_t>(bytesPerRow, 0));
    for (int r = 0; r < std::abs(this->infoHeader.height); ++r) {
      const int rowIndex = (std::abs(this->infoHeader.height) - 1 - r) * rowSize; // BMP stores pixel data in reverse order (bottom-up)
      for (int c = 0; c < this->infoHeader.width; ++c) {
        // Origin is at the bottom left corner
        // BMP stores pixel data in reverse order (bottom-up)
        // For 1-bit images, we need to extract the bits from the byte
        if (this->infoHeader.bit_count == 1) {
          // c / 8 gives the byte index, Then we shift the byte to get to the bit
          // c % 8 gives the bit index in the byte, and then the bit is extracted by ANDing with 0x01
          uint8_t bit = ((pixelData[r * rowSize + c / 8] >> (7 - (c % 8))) & 0x01);
          this->pixelData2D[std::abs(this->infoHeader.height) - 1 - r][c] = bit;
        } else if (this->infoHeader.bit_count == 24) {
          int pixelIndex = rowIndex + c * 3;
          this->pixelData2D[r][c * 3 + 0] = pixelData[pixelIndex + 0]; // Blue
          this->pixelData2D[r][c * 3 + 1] = pixelData[pixelIndex + 1]; // Green
          this->pixelData2D[r][c * 3 + 2] = pixelData[pixelIndex + 2]; // Red
        } else {
          this->pixelData2D[std::abs(this->infoHeader.height) - 1 - r][c] = pixelData[r * rowSize + c];
        }
      }
    }

    // Close the file
    file.close();
  }

  void write(const char* filename) {
    // Create output stream in binary mode
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
      throw std::runtime_error("Could not open file " + std::string(filename));
    }

    // Determine the offset for pixel data
    this->fileHeader.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    if (this->infoHeader.bit_count == 32) {
      // Only include color header if pixel depth is 32 bits
      this->fileHeader.offset_data += sizeof(BMPColorHeader);
    } else if (this->infoHeader.bit_count <= 8) {
      int colorTableEntries = 1 << this->infoHeader.bit_count;
      if (this->infoHeader.colors_used > 0) {
        colorTableEntries = this->infoHeader.colors_used;
      }
      this->fileHeader.offset_data += colorTableEntries * sizeof(uint32_t);
    }

    // Calculate row size with padding to 4-byte boundary
    int rowSize = ((this->infoHeader.bit_count * this->infoHeader.width + 31) / 32) * 4;

    // Calculate image size
    int imageSize = rowSize * std::abs(this->infoHeader.height);
    this->infoHeader.size_image = imageSize;

    // Update file size
    this->fileHeader.file_size = this->fileHeader.offset_data + imageSize;

    // Write the headers
    output.write(reinterpret_cast<const char*>(&this->fileHeader), sizeof(this->fileHeader));
    output.write(reinterpret_cast<const char*>(&this->infoHeader), sizeof(this->infoHeader));

    if (this->infoHeader.bit_count == 32) {
      output.write(reinterpret_cast<const char*>(&this->colorHeader), sizeof(this->colorHeader));
    }

    // Write color table for bit depths <= 8
    if (this->infoHeader.bit_count <= 8) {
      int colorTableEntries = 1 << this->infoHeader.bit_count;
      if (this->infoHeader.colors_used > 0) {
        colorTableEntries = this->infoHeader.colors_used;
      }
      // Write color table
      // For 1-bit image, typically we need two entries: 0 (black) and 1 (white)
      for (int i = 0; i < colorTableEntries; i++) {
        uint8_t color[4];
        if (i == 0) { // Black
          color[0] = 0;    // Blue
          color[1] = 0;    // Green
          color[2] = 0;    // Red
          color[3] = 0;    // Reserved
        } else { // White
          color[0] = 255;  // Blue
          color[1] = 255;  // Green
          color[2] = 255;  // Red
          color[3] = 0;    // Reserved
        }
        output.write(reinterpret_cast<const char*>(color), 4);
      }
    }

    // Generate and write pixel data - this section needs to be outside the if/else blocks
    std::vector<uint8_t> pixelData(imageSize, 0);

    if (this->infoHeader.bit_count == 1) {
      // for 1-bit images, we need to pack bits
      for (int r = 0; r < std::abs(this->infoHeader.height); ++r) {
        for (int c = 0; c < this->infoHeader.width; c += 8) {
          uint8_t byte = 0;
          // Pack up to 8 pixels into a byte
          for (int b = 0; b < 8 && (c + b) < this->infoHeader.width; ++b) {
            if (this->pixelData2D[r][c + b] != 0) {
              byte |= (1 << (7 - b)); // Set the bit if pixel is not 0 (MSB first)
            }
          }
          // Calculate position in pixelData
          // Origin is at the bottom left corner
          // BMP stores pixel data in reverse order (bottom-up)
          int byteIndex = (std::abs(this->infoHeader.height) - 1 - r) * rowSize + (c / 8);
          pixelData[byteIndex] = byte;
        }
      }
    } else if (this->infoHeader.bit_count == 24) {
      // Handle 24-bit RGB images
      for (int r = 0; r < std::abs(this->infoHeader.height); ++r) {
        for (int c = 0; c < this->infoHeader.width; ++c) {
          // Calculate position in pixelData
          // BMP stores pixel data in reverse order (bottom-up) and as BGR
          unsigned int pixelIndex = (std::abs(this->infoHeader.height) - 1 - r) * rowSize + c * 3;

          if (pixelIndex + 2 < pixelData.size()) {
            pixelData[pixelIndex] = this->pixelData2D[r][c * 3];         // Blue
            pixelData[pixelIndex + 1] = this->pixelData2D[r][c * 3 + 1]; // Green
            pixelData[pixelIndex + 2] = this->pixelData2D[r][c * 3 + 2]; // Red
          }
        }
      }
    } else {
      // Basic implementation for other bit depths
      std::cerr << "Warning: Images with other bit-depths may not be properly supported." << std::endl;
      for (int r = 0; r < std::abs(this->infoHeader.height); ++r) {
        for (int c = 0; c < this->infoHeader.width; ++c) {
          unsigned int bytesPerPixel = this->infoHeader.bit_count / 8;
          unsigned int pixelIndex = (std::abs(this->infoHeader.height) - 1 - r) * rowSize + c * bytesPerPixel;

          if (pixelIndex < pixelData.size()) {
            // Copy each byte of the pixel
            for (unsigned int b = 0; b < bytesPerPixel && b < this->pixelData2D[r].size() / this->infoHeader.width; ++b) {
              if (c * bytesPerPixel + b < this->pixelData2D[r].size()) {
                pixelData[pixelIndex + b] = this->pixelData2D[r][c * bytesPerPixel + b];
              }
            }
          }
        }
      }
    }

    // Write the pixel data
    output.write(reinterpret_cast<const char*>(pixelData.data()), pixelData.size());
    if (!output) {
      throw std::runtime_error("Error writing pixel data to " + std::string(filename));
    }

    std::cout << "Successfully wrote " << pixelData.size() << " bytes to " << filename << std::endl;

    // Close the file
    output.close();
  }
};


#endif // !BMP_HPP