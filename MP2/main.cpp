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
      BMPImage erodedBmp(bmp, bmp.erosion());
      BMPImage dilatedBmp(bmp, bmp.dilation());
      BMPImage openedBmp(bmp, bmp.opening());
      BMPImage closedBmp(bmp, bmp.closing());
      BMPImage boundaryBmp(bmp, bmp.boundary());
      std::string imageName = bmp.getName();
      erodedBmp.save((std::string(imageName) + "_eroded.bmp").c_str());
      dilatedBmp.save((std::string(imageName) + "_dilated.bmp").c_str());
      openedBmp.save((std::string(imageName) + "_opened.bmp").c_str());
      closedBmp.save((std::string(imageName) + "_closed.bmp").c_str());
      boundaryBmp.save((std::string(imageName) + "_boundary.bmp").c_str());
      std::cout << "==========================" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error processing " << image << ": " << e.what() << std::endl;
    }
  }

  return 0;
}