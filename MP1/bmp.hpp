#ifndef BMP_HPP
#define BMP_HPP
#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>

#pragma pack(push, 1) // Ensure no padding is added to the structures

#define BMFILETYPE 0x4D42 // 'BM' in ASCII

// --------------- BEGIN_CITATION [1] ---------------- //
struct BMPFileHeader {
  uint16_t file_type{BMFILETYPE};          // File type always BM which is 0x4D42
  uint32_t file_size{0};               // Size of the file (in bytes)
  uint16_t reserved1{0};               // Reserved, always 0
  uint16_t reserved2{0};               // Reserved, always 0
  uint32_t offset_data{0};             // Start position of pixel data (bytes from the beginning of the file)
};

struct BMPInfoHeader {
  uint32_t size{0};                      // Size of this header (in bytes)
  int32_t width{0};                      // width of bitmap in pixels
  int32_t height{0};                     // width of bitmap in pixels
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
  int label;
  int area; // Number of pixels in the component
  std::vector<int> pixels; // List of pixel indices in the component
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
    std::cout << "Bit Count: " << this->infoHeader.bit_count << std::endl;
    std::cout << "Compression: " << this->infoHeader.compression << std::endl;
    std::cout << "Colors Used: " << this->infoHeader.colors_used << std::endl;
  }

  void printPixelData() const {
    std::cout << "Pixel Data:\n";
    for (size_t i = 0; i < this->pixelData.size(); ++i) {
      std::cout << static_cast<int>(this->pixelData[i]) << " ";
      if ((i + 1) % this->infoHeader.width == 0) {
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }

  void connectedComponentLabeling() {
    const int W = this->infoHeader.width;
    const int H = std::abs(this->infoHeader.height);

    // Initialize parent vector where each element is its own parent
    std::vector<int> parent(W * H + 1);
    for (int i = 0; i <= W * H; i++) {
      parent[i] = i;
    }

    std::vector<int> labels(W * H, 0);

    // Helper function to find the root label
    auto findLabel = [&parent](int label) {
      while (parent[label] != label) {
        // Path compression - make every node point to its grandparent
        parent[label] = parent[parent[label]];
        label = parent[label];
      }
      return label;
    };

    // Helper function to union two labels
    auto unionLabels = [&parent, &findLabel](int a, int b) {
      int rootA = findLabel(a);
      int rootB = findLabel(b);
      if (rootA != rootB) {
        parent[rootB] = rootA;
      }
    };

    int labelCount = 1;

    // First Pass: Assign labels and record equivalences
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        int index = y * W + x;

        // Only process foreground pixels (non-zero)
        if (this->pixelData[index] == 0) {
          continue;
        }

        // Check neighbors (safely)
        int left = (x > 0) ? labels.at(y * W + (x - 1)) : 0;
        int top = (y > 0) ? labels.at((y - 1) * W + x) : 0;

        if (left == 0 && top == 0) {
          // New component
          labels[index] = labelCount++;
        } else if (left != 0 && top == 0) {
          // Only left neighbor
          labels[index] = left;
        } else if (left == 0 && top != 0) {
          // Only top neighbor
          labels[index] = top;
        } else {
          // Both neighbors
          labels[index] = std::min(left, top);
          unionLabels(left, top);
        }
      }
    }

    // Second Pass: Resolve equivalences
    for (int i = 0; i < W * H; ++i) {
      if (labels[i] != 0) {
        labels[i] = findLabel(labels[i]);
      }
    }

    std::set<int> uniqueLabels(labels.begin(), labels.end());
    uniqueLabels.erase(0); // Remove background label

    // Create components
    int numComponents = uniqueLabels.size();
    std::cout << "Number of connected components: " << numComponents << std::endl;

    this->components.clear();
    std::unordered_map<int, int> labelToComponentIndex;
    int componentIndex = 0;

    for (int label : uniqueLabels) {
      labelToComponentIndex[label] = componentIndex++;
      this->components.push_back({label, 0, {}});
    }

    // Populate components
    for (int i = 0; i < W * H; ++i) {
      if (labels[i] != 0) {
        int lab = labels[i];
        int compIndex = labelToComponentIndex[lab];
        this->components[compIndex].pixels.push_back(i);
        this->components[compIndex].area++;
      }
    }

    std::cout << "Found " << this->components.size() << " connected components" << std::endl;
  }

  void applySizeFilter(const int sizeThreshold = 10) {
    // Add safety check for empty components
    if (this->components.empty()) {
      std::cout << "No components to filter!" << std::endl;
      return;
    }

    std::vector<Component> filteredComponents;
    filteredComponents.reserve(this->components.size());

    std::cout << "Applying size filter with threshold: " << sizeThreshold << std::endl;
    std::cout << "Number of components before size filter: " << this->components.size() << std::endl;

    // Filter components by area
    for (const auto& c : this->components) {
      if (c.area >= sizeThreshold) {
        filteredComponents.push_back(c);
      }
    }

    std::cout << "Number of components after size filter: " << filteredComponents.size() << std::endl;

    // Create filtered image (initialize all to background/0)
    std::vector<uint8_t> filteredPixelData(this->pixelData.size(), 0);

    // Check if we have any components that passed the filter
    if (filteredComponents.empty()) {
      std::cout << "Warning: No components passed the size filter." << std::endl;
      this->components = filteredComponents;  // Update to empty component list
      return;  // Keep all pixels as background (0)
    }

    // Set foreground pixels (255) for components that passed the filter
    int pixelsSet = 0;
    for (const auto& c : filteredComponents) {
      // Add safety check for empty pixels
      if (c.pixels.empty()) {
        std::cout << "Warning: Component has no pixels" << std::endl;
        continue;
      }

      // Set pixels
      for (const auto& pixelIndex : c.pixels) {
        // Bounds check
        if (pixelIndex < 0 || pixelIndex >= filteredPixelData.size()) {
          std::cerr << "Error: Pixel index " << pixelIndex << " out of bounds (0-"
            << filteredPixelData.size() - 1 << ")" << std::endl;
          continue;
        }

        filteredPixelData[pixelIndex] = 255;  // Set to foreground
        pixelsSet++;
      }
    }

    std::cout << "Set " << pixelsSet << " pixels to foreground in filtered image" << std::endl;

    // Update pixel data and components
    this->pixelData = filteredPixelData;
    this->components = filteredComponents;

    std::cout << "Filtered pixel data size: " << this->pixelData.size() << std::endl;
  }

  std::string getName() const {
    return this->name;
  }

private:
  BMPFileHeader fileHeader;
  BMPInfoHeader infoHeader;
  BMPColorHeader colorHeader;
  std::vector<uint8_t> pixelData; // Pixel data
  std::string name;
  std::vector<Component> components;

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