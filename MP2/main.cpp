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
      StructuringElement A(3, 3); // Default kernel (3x3) all ones
      StructuringElement B(5, 5); // Default kernel (5x5) all ones
      StructuringElement C(3, 3); // Kernel (3x3)
      C.setElement(1, 1, 0); // Center
      C.setElement(1, 2, 0); // Right
      C.setElement(2, 1, 0); // Down
      C.setElement(2, 2, 0); // Down-Right
      std::vector<StructuringElement> kernels = {A, B, C};
      for (int i = 0; i < kernels.size(); i++) {
        StructuringElement k = kernels[i];
        BMPImage erodedBmp(bmp, bmp.erosion(k));
        BMPImage dilatedBmp(bmp, bmp.dilation(k));
        BMPImage openedBmp(bmp, bmp.opening(k));
        BMPImage closedBmp(bmp, bmp.closing(k));
        BMPImage boundaryBmp(bmp, bmp.boundary(k));
        std::string kernelName = i == 0 ? "A" : (i == 1 ? "B" : "C");
        std::string imageName = bmp.getName() + "_" + kernelName;
        erodedBmp.save((std::string(imageName) + "_eroded.bmp").c_str());
        dilatedBmp.save((std::string(imageName) + "_dilated.bmp").c_str());
        openedBmp.save((std::string(imageName) + "_opened.bmp").c_str());
        closedBmp.save((std::string(imageName) + "_closed.bmp").c_str());
        boundaryBmp.save((std::string(imageName) + "_boundary.bmp").c_str());
      }

      std::cout << "==========================" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error processing " << image << ": " << e.what() << std::endl;
    }
  }

  return 0;
}