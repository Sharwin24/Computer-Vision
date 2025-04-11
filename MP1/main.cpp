#include <iostream>
#include <fstream>
#include <vector>

#include "bmp.hpp"

int main(int argc, const char* argv[]) {
  std::cout << "BMP Image Processing" << std::endl;
  std::vector<std::string> images;
  images.push_back("test.bmp");
  images.push_back("face.bmp");
  images.push_back("face_old.bmp");
  images.push_back("gun.bmp");

  std::cout << "Processing images" << std::endl;
  for (const auto& image : images) {
    std::cout << " - " << image << std::endl;
  }
  std::cout << std::endl;

  for (const auto& image : images) {
    try {
      std::cout << "==========================" << std::endl;
      std::cout << "Processing " << image << std::endl;
      BMPImage bmp(image.c_str());
      bmp.printInfo();
      // bmp.printPixelData();
      bmp.connectedComponentLabeling();
      // Apply size filter
      bmp.applySizeFilter(50);
      // Save the filtered image
      std::string outputFilename = bmp.getName() + "_filtered.bmp";
      bmp.save(outputFilename.c_str());
      // Save the Component Image
      bmp.colorizeComponents();
      std::cout << "==========================" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error processing " << image << ": " << e.what() << std::endl;
    }
  }

  return 0;
}