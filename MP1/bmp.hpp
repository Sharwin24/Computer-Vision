#ifndef BMP_HPP
#define BMP_HPP
#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>

#pragma pack(push, 1) // Ensure no padding is added to the structures
#pragma pack(pop) // Restore the previous packing alignment

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

// --------------- END_CITATION [1] ---------------- //

class BMPImage {
public:
  BMPImage() = delete;

  BMPImage(const char* filename) {
    this->read(filename);
  }

  ~BMPImage() {
    // Destructor
  }

  void save(const char* filename) {
    this->write(filename);
  }

  void printInfo() const {
    std::cout << "File Size: " << fileHeader.file_size << " bytes\n";
    std::cout << "Width: " << infoHeader.width << " pixels\n";
    std::cout << "Height: " << infoHeader.height << " pixels\n";
    std::cout << "Bit Count: " << infoHeader.bit_count << "\n";
    std::cout << "Compression: " << infoHeader.compression << "\n";
  }

  void printPixelData() const {
    std::cout << "Pixel Data:\n";
    for (size_t i = 0; i < pixelData.size(); ++i) {
      std::cout << static_cast<int>(pixelData[i]) << " ";
      if ((i + 1) % infoHeader.width == 0) {
        std::cout << "\n";
      }
    }
  }

private:
  BMPFileHeader fileHeader;
  BMPInfoHeader infoHeader;
  BMPColorHeader colorHeader;
  std::vector<uint8_t> pixelData; // Pixel data

  void read(const char* filename) {
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Could not open file" + std::string(filename));
    }
    // Read File Header
    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    if (fileHeader.file_type != BMFILETYPE) {
      throw std::runtime_error(std::string(filename) + " is not a BMP file");
    }

    // Read Info Header
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    // Validate that image is uncompressed (Only 0 is supported)
    if (infoHeader.compression != 0) {
      throw std::runtime_error(std::string(filename) + " is not an uncompressed BMP file");
    }

    // If 32 bits per pixel, read the color header as well
    if (infoHeader.bit_count == 32) {
      file.read(reinterpret_cast<char*>(&colorHeader), sizeof(colorHeader));
    }

    // Move file pointer to beginning of pixel data
    file.seekg(fileHeader.offset_data, std::ios::beg);

    // Determine image size
    if (infoHeader.size_image == 0) {
      // For safety, use absolute heigh since BMP height can be negative
      infoHeader.size_image = infoHeader.width * std::abs(infoHeader.height) * (infoHeader.bit_count / 8);
    }

    // Read the pixel data
    pixelData.resize(infoHeader.size_image);
    file.read(reinterpret_cast<char*>(pixelData.data()), infoHeader.size_image);
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
    if (infoHeader.bit_count == 32) {
      // Only include color header if pixel depth is 32 bits
      this->fileHeader.offset_data += sizeof(BMPColorHeader);
    }

    // Update file size
    this->fileHeader.file_size = this->fileHeader.offset_data + static_cast<uint32_t>(pixelData.size());

    // Write the headers
    output.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    output.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
    if (infoHeader.bit_count == 32) {
      output.write(reinterpret_cast<const char*>(&colorHeader), sizeof(colorHeader));
    }

    // Write the pixel data
    output.write(reinterpret_cast<const char*>(pixelData.data()), pixelData.size());
    if (!output) {
      throw std::runtime_error("Error writing pixel data to " + std::string(filename));
    }

    // Close the file
    output.close();
  }
};


#endif // !BMP_HPP