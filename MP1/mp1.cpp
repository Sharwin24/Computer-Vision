#include <iostream>
#include <fstream>
#include <vector>

#include "bmp.hpp"

int main(int argc, const char *argv[]) {
  std::vector<std::string> images;
  if (argc > 2) {
    images.reserve(argc - 1);
    for (int i = 1; i < argc; ++i) {
      images.emplace_back(argv[i]);
    }
  }

  if (images.empty()) {
    std::cerr << "Usage: " << argv[0] << " <image1.bmp> <image2.bmp> ...\n";
    std::cerr << "No images provided. Using hardcoded test images.\n";
    // Use hardcoded test images if no arguments are provided
    images = {
      "test.bmp",
      "face.bmp",
      "gun.bmp",
      "face_old.bmp"
    };
  }

  std::vector<BMPImage> bmpImages;
  for (const auto& image : images) {
    try {
      BMPImage bmp(image.c_str());
      bmp.printInfo();
      // bmp.printPixelData();
      bmpImages.push_back(bmp);
      // Perform connected component labeling
      bmp.connectedComponentLabeling();
      // Apply size filter
      bmp.applySizeFilter(10);
      // Save the filtered image
      std::string outputFilename = bmp.getName() + "_filtered.bmp";
      bmp.save(outputFilename.c_str());
    } catch (const std::exception& e) {
      std::cerr << "Error processing " << image << ": " << e.what() << "\n";
    }
  }

  return 0;
}