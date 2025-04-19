#include <iostream>
#include <fstream>
#include <vector>

#include "bmp.hpp"

int main(int argc, const char* argv[]) {
  std::cout << "BMP Image Processing" << std::endl;
  std::vector<std::string> images;
  images.push_back("moon.bmp");

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
      bmp.histogramEquilization(true);
      bmp.save("moon_equalized.bmp");
      bmp.lightingCorrection();
      bmp.save("moon_lighting_linear.bmp");
      BMPImage bmp2(image.c_str());
      bmp2.lightingCorrection(false);
      bmp2.save("moon_lighting_quadratic.bmp");
      std::cout << "==========================" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error processing " << image << ": " << e.what() << std::endl;
    }
  }

  return 0;
}