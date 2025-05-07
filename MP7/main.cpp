#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

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

void combineHistograms(const std::vector<std::string> files) {
  std::vector<std::vector<int>> histogramBGR(256, std::vector<int>(3, 0));
  std::vector<std::vector<int>> histogramHSI(256, std::vector<int>(3, 0));
  for (const auto& file : files) {
    // "gun1_BGR_histogram.csv" -> extract "gun1", "BGR"
    const std::string name = file.substr(0, file.find("_"));
    const std::string colorSpace = file.substr(file.find("_") + 1, file.find("_histogram.csv") - file.find("_") - 1);
    std::cout << "Adding Histogram from " << file << std::endl;
    // Read the histogram file
    std::ifstream inFile(file);
    if (!inFile.is_open()) {
      std::cerr << "Error opening file: " << file << std::endl;
      continue;
    }
    std::string line;
    while (std::getline(inFile, line)) {
      std::istringstream iss(line);
      std::vector<std::string> tokens;
      std::string token;
      while (std::getline(iss, token, ',')) {
        tokens.push_back(token);
      }
      if (tokens.size() != 4) {
        std::cerr << "Invalid line format: " << line << std::endl;
        continue;
      }
      // Parse the line
      int index = std::stoi(tokens[0]);
      int value1 = std::stoi(tokens[1]);
      int value2 = std::stoi(tokens[2]);
      int value3 = std::stoi(tokens[3]);
      // Update the histogram
      if (colorSpace == "BGR") {
        histogramBGR[index][0] += value1;
        histogramBGR[index][1] += value2;
        histogramBGR[index][2] += value3;
      } else if (colorSpace == "HSI") {
        histogramHSI[index][0] += value1;
        histogramHSI[index][1] += value2;
        histogramHSI[index][2] += value3;
      }
    }
    inFile.close();
  }
  // After combining histograms, save each to a CSV file
  std::ofstream outFileBGR("combined_BGR_histogram.csv");
  std::ofstream outFileHSI("combined_HSI_histogram.csv");
  if (outFileBGR.is_open() && outFileHSI.is_open()) {
    for (int i = 0; i < 256; ++i) {
      outFileBGR << i << "," << histogramBGR[i][0] << "," << histogramBGR[i][1] << "," << histogramBGR[i][2] << "\n";
      outFileHSI << i << "," << histogramHSI[i][0] << "," << histogramHSI[i][1] << "," << histogramHSI[i][2] << "\n";
    }
    outFileBGR.close();
    outFileHSI.close();
    std::cout << "Combined histograms saved to combined_BGR_histogram.csv and combined_HSI_histogram.csv" << std::endl;
  } else {
    std::cerr << "Error opening file for writing combined histograms" << std::endl;
  }
}

void bmp2png(const std::string file) {
  // Convert BMP files to PNG using OpenCV
  std::string pngFile = file.substr(0, file.find_last_of('.')) + ".png";
  cv::Mat image = cv::imread(file);
  if (image.empty()) {
    std::cerr << "Error reading image: " << file << std::endl;
  }
  if (!cv::imwrite(pngFile, image)) {
    std::cerr << "Error writing PNG file: " << pngFile << std::endl;
  } else {
    std::cout << "Converted " << file << " to " << pngFile << std::endl;
  }
}

std::vector<std::string> listFilesInDir(const std::string& dir, const std::string& ext = ".bmp") {
  std::vector<std::string> images;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ext) {
      images.push_back(entry.path().string());
    }
  }
  if (images.empty()) {
    std::cerr << "No " << ext << " files found in directory: " << dir << std::endl;
  }
  std::cout << "Found " << images.size() << " " << ext << " files in directory: " << dir << std::endl;
  return images;
}

int main() {
  std::cout << "BMP Image Processing" << std::endl;
  std::cout << "Using OpenCV Version: " << CV_VERSION << std::endl;
  std::cout << "Processing images" << std::endl;
  std::vector<std::string> images = listFilesInDir("image_girl/", ".jpg");
  std::vector<BMPImage> bmpImages;
  bmpImages.reserve(images.size());
  BMPImage targetImage("target.jpg");
  // targetImage.grayscale();
  targetImage.save("target.bmp");
  const std::string matchingMethod = "SSD"; // ["SSD", "CC", "NCC"]
  for (const auto& image : images) {
    BMPImage bmp(image.c_str());
    // bmp.imageMatching(targetImage, "SSD");
    bmp.save("output/" + bmp.getName() + ".bmp");
    // bmpImages.push_back(bmp);
  }

  return 0;
}