#include <iostream>
#include <fstream>
#include <vector>

#include "bmp.hpp"

int main(int argc, const char* argv[]) {
  std::cout << "BMP Image Processing" << std::endl;
  std::vector<std::string> images;
  images.push_back("gun.bmp");
  images.push_back("palm.bmp");

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
      std::vector<std::vector<uint8_t>> erodedImage = bmp.erosion();
      std::vector<std::vector<uint8_t>> dilatedImage = bmp.dilation();
      BMPImage erodedBmp(bmp, erodedImage);
      BMPImage dilatedBmp(bmp, dilatedImage);
      std::string imageName = bmp.getName();
      erodedBmp.save((std::string(imageName) + "_eroded.bmp").c_str());
      dilatedBmp.save((std::string(imageName) + "_dilated.bmp").c_str());
      std::cout << "==========================" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error processing " << image << ": " << e.what() << std::endl;
    }
  }

  return 0;
}