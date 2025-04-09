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
    return 1;
  }

  return 0;
}