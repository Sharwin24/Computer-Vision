#include <iostream>
#include <fstream>
#include <vector>

#include "bmp.hpp"

void savePixels(const std::vector<std::pair<int, int>> pixels, const std::string& filename) {
  // Save the list of pixels to a file
  std::ofstream outFile(filename);
  if (outFile.is_open()) {
    for (const auto& pixel : pixels) {
      outFile << pixel.first << " " << pixel.second << "\n";
    }
    outFile.close();
  } else {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
  }
}

int main() {
  std::cout << "BMP Image Processing" << std::endl;
  std::vector<std::string> images;
  images.push_back("gun1.bmp");
  images.push_back("joy1.bmp");
  images.push_back("pointer1.bmp");
  std::cout << "Using OpenCV Version: " << CV_VERSION << std::endl;
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
      // std::vector<std::pair<int, int>> region = bmp.selectRegion();
      const std::string regionPixelsFile = bmp.getName() + "_skin_pixels.txt";
      // savePixels(region, regionPixelsFile); // ONLY RUN THIS ONCE TO SAVE PIXELS
      bmp.createHistogramFromFile(regionPixelsFile, ColorSpace::BGR);
      bmp.createHistogramFromFile(regionPixelsFile, ColorSpace::HSI);
      std::cout << "==========================" << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error processing " << image << ": " << e.what() << std::endl;
    }
  }

  return 0;
}