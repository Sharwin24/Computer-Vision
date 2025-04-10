/**
 * @file bmp.hpp
 * @author Sharwin Patil (sharwinpatil@u.northwestern.edu)
 * @brief Support for BMP image processing
 * @version 0.1
 * @date 2025-04-10
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

struct Component {
  uint32_t label; // Unique Label of the component
  uint32_t area; // Number of pixels in the component
  std::vector<std::pair<uint8_t, uint8_t>> pixels; // List of pixel locations [row, col]
};

class BMPImage {
public:
  BMPImage() = delete;

  BMPImage(const char* filename) {
    this->read(filename);
    this->name = getImageName(filename);
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
    std::vector<std::vector<int>> labeledImage(numRows, std::vector<int>(numCols, 1));
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
    if (filteredComponents.empty()) {
      std::cerr << "Warning: No components passed the size filter. Components stay the same." << std::endl;
      return;
    }

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

  void showComponentImages() {
    // For the components that we found, create one image
    // where each component is in a different color or grayscale intensity
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
    }

    // Move file pointer to beginning of pixel data
    file.seekg(this->fileHeader.offset_data, std::ios::beg);

    // Determine image size
    if (this->infoHeader.size_image == 0) {
      // For safety, use absolute heigh since BMP height can be negative
      this->infoHeader.size_image = this->infoHeader.width * std::abs(this->infoHeader.height) * (this->infoHeader.bit_count / 8);
    }

    // Read the pixel data
    this->pixelData.resize(this->infoHeader.size_image);
    file.read(reinterpret_cast<char*>(this->pixelData.data()), this->infoHeader.size_image);
    if (!file) {
      throw std::runtime_error("Error reading pixel data from " + std::string(filename));
    }

    // Check if the pixel data was read correctly
    if (this->pixelData.size() != this->infoHeader.size_image) {
      throw std::runtime_error("Error: Pixel data size mismatch in " + std::string(filename));
    }

    // Populate 2D pixel data representation that matches the image dimensions
    this->pixelData2D.resize(std::abs(this->infoHeader.height), std::vector<int>(this->infoHeader.width, 0));
    for (int i = 0; i < std::abs(this->infoHeader.height); ++i) {
      for (int j = 0; j < this->infoHeader.width; ++j) {
        int index = i * this->infoHeader.width + j;
        this->pixelData2D[i][j] = this->pixelData[index];
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
    }

    // Update file size
    this->fileHeader.file_size = this->fileHeader.offset_data + static_cast<uint32_t>(pixelData.size());

    // Write the headers
    output.write(reinterpret_cast<const char*>(&this->fileHeader), sizeof(this->fileHeader));
    output.write(reinterpret_cast<const char*>(&this->infoHeader), sizeof(this->infoHeader));
    if (this->infoHeader.bit_count == 32) {
      output.write(reinterpret_cast<const char*>(&this->colorHeader), sizeof(this->colorHeader));
    }

    // Write the pixel data
    output.write(reinterpret_cast<const char*>(this->pixelData.data()), this->pixelData.size());
    if (!output) {
      throw std::runtime_error("Error writing pixel data to " + std::string(filename));
    }

    // Close the file
    output.close();
  }
};


#endif // !BMP_HPP