#include "bmp.hpp"


BMPImage::BMPImage(const char* filename) {
  this->read(filename);
  this->name = getImageName(filename);
}

// Public Functions

void BMPImage::save(const char* filename) {
  this->write(filename);
}

void BMPImage::printInfo() const {
  std::cout << "BMP Image: " << this->name << "\n";
  std::cout << "File Size: " << this->fileHeader.file_size << " bytes\n";
  std::cout << "Width: " << this->infoHeader.width << " pixels\n";
  std::cout << "Height: " << this->infoHeader.height << " pixels\n";
  std::cout << "Bit Count: " << this->infoHeader.bit_count << "\n";
  std::cout << "Compression: " << this->infoHeader.compression << "\n\n";
}

void BMPImage::printPixelData() const {
  std::cout << "Pixel Data:\n";
  for (size_t i = 0; i < this->pixelData.size(); ++i) {
    std::cout << static_cast<int>(this->pixelData[i]) << " ";
    if ((i + 1) % this->infoHeader.width == 0) {
      std::cout << "\n";
    }
  }
}

// Private Functions
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

void BMPImage::read(const char* filename) {
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

void BMPImage::write(const char* filename) {
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