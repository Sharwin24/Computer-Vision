/**
 * @file bmp.hpp
 * @author Sharwin Patil (sharwinpatil@u.northwestern.edu)
 * @brief Support for BMP image processing
 * @version 2.0
 * @date 2025-04-14
 * @details Includes support for Connected Component Labeling, size filtering, and colorizing components.
 * @details Includes support for morphological operations: erosion, dilation, opening, and closing.
 *
 * @copyright Copyright (c) 2025
 *
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
 * @brief Struct describing a Kernel for Morphological Operations
 *
 */
struct StructuringElement {
  std::vector<std::vector<uint8_t>> element;
  const int rows;
  const int cols;

  StructuringElement() = delete;

  /**
   * @brief Create a SE Kernel with the given size. Kernel values all default to 1.
   *
   * @param r the number of rows in the kernel
   * @param c the number of columns in the kernel
   */
  StructuringElement(int r, int c) : rows(r), cols(c) {
    this->element.resize(r, std::vector<uint8_t>(c, 1));
  }

  void setElement(int r, int c, uint8_t value) {
    if (r >= 0 && r < this->rows && c >= 0 && c < this->cols) {
      this->element[r][c] = value;
    }
  }

  void print() const {
    for (const auto& row : this->element) {
      for (const auto& val : row) {
        std::cout << static_cast<int>(val) << " ";
      }
      std::cout << std::endl;
    }
  }

  uint8_t operator()(int r, int c) const {
    if (r >= 0 && r < this->rows && c >= 0 && c < this->cols) {
      return this->element[r][c];
    }
    return 0;
  }
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

  void save(const char* filename) {
    this->write(filename);
  }
  void printInfo() const {
    std::cout << "BMP Image: " << this->name << std::endl;
    std::cout << "File Size: " << this->fileHeader.file_size << " bytes" << std::endl;
    std::cout << "Width: " << this->infoHeader.width << " pixels" << std::endl;
    std::cout << "Height: " << this->infoHeader.height << " pixels" << std::endl;
    std::cout << "NumPixels: " << this->pixelData2D.size() * this->pixelData2D[0].size() << std::endl;
    std::cout << "ImageSize: " << this->infoHeader.size_image << " bytes" << std::endl;
    std::cout << "Bit Count: " << this->infoHeader.bit_count << std::endl;
    std::cout << "Compression: " << this->infoHeader.compression << std::endl;
    std::cout << "Colors Used: " << this->infoHeader.colors_used << std::endl;
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
    std::ofstream logFile("test_cpp_raw.txt");
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
    for (int i = 0; i < parent.size(); ++i) {
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

  void applySizeFilter(const int sizeThreshold = 10) {
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

  std::vector<std::vector<uint8_t>> erosion(const StructuringElement& kernel = StructuringElement(3, 3)) {
    // E = A \oplus B = \{z \mid B_z \subseteq A \}
    // Convolve the image with the kernel
    const int R = this->pixelData2D.size();
    const int C = this->pixelData2D[0].size();

    std::vector<std::vector<uint8_t>> erodedImage(R, std::vector<uint8_t>(C, 0));
    std::cout << "Erosion on " << this->infoHeader.bit_count << "-bit image" << std::endl;
    for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
        uint8_t minValue = this->infoHeader.bit_count == 1 ? 1 : 255;
        // Apply kernel to pixel location
        for (int kr = -kernel.rows / 2; kr <= kernel.rows / 2; kr++) {
          for (int kc = -kernel.cols / 2; kc <= kernel.cols / 2; kc++) {
            // Skip if kernel value is 0
            if (kernel(kr + kernel.rows / 2, kc + kernel.cols / 2) == 0) { continue; }
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
        for (int kr = -kernel.rows / 2; kr <= kernel.rows / 2; kr++) {
          for (int kc = -kernel.cols / 2; kc <= kernel.cols / 2; kc++) {
            // Skip if kernel value is 0
            if (kernel(kr + kernel.rows / 2, kc + kernel.cols / 2) == 0) { continue; }
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

  std::string getName() const { return this->name; }

private:
  BMPFileHeader fileHeader;
  BMPInfoHeader infoHeader;
  BMPColorHeader colorHeader;
  std::vector<std::vector<uint8_t>> pixelData2D; // 2D representation of pixel data
  std::string name;
  std::vector<Component> components;

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
    // // Print binary image for debugging
    // std::cout << "Binary Image:" << std::endl;
    // for (const auto& row : binaryImage) {
    //   for (const auto& pixel : row) {
    //     std::cout << pixel << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << std::endl;
    return binaryImage;
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

    int rowSize = ((this->infoHeader.bit_count * this->infoHeader.width + 31) / 32) * 4;
    // For safety, use absolute heigh since BMP height can be negative
    int imageSize = rowSize * std::abs(this->infoHeader.height);

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
    std::cout << "Successfully read " << imageSize << " bytes of pixel data" << std::endl;
    std::cout << "Pixel data size: " << pixelData.size() << " bytes" << std::endl;

    // Convert pixel data to 2D array matching image dimensions
    this->pixelData2D.resize(std::abs(this->infoHeader.height), std::vector<uint8_t>(this->infoHeader.width, 0));
    for (int r = 0; r < std::abs(this->infoHeader.height); ++r) {
      for (int c = 0; c < this->infoHeader.width; ++c) {
        // Origin is at the bottom left corner
        // BMP stores pixel data in reverse order (bottom-up)
        // For 1-bit images, we need to extract the bits from the byte
        if (this->infoHeader.bit_count == 1) {
          this->pixelData2D[std::abs(this->infoHeader.height) - 1 - r][c] = ((pixelData[r * rowSize + c / 8] >> (7 - (c % 8))) & 0x01);
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
          int pixelIndex = (std::abs(this->infoHeader.height) - 1 - r) * rowSize + c * 3;

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
          int bytesPerPixel = this->infoHeader.bit_count / 8;
          int pixelIndex = (std::abs(this->infoHeader.height) - 1 - r) * rowSize + c * bytesPerPixel;

          if (pixelIndex < pixelData.size()) {
            // Copy each byte of the pixel
            for (int b = 0; b < bytesPerPixel && b < this->pixelData2D[r].size() / this->infoHeader.width; ++b) {
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