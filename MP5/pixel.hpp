#ifndef PIXEL_HPP
#define PIXEL_HPP
#include <cstdint>
#include <variant> // C++17 variant
#include <cmath>
#include <algorithm>
#include <iostream>


enum class ColorSpace {
  BGR, // Blue, Green, Red (default for BMP)
  HSI // Hue, Saturation, Intensity
};

// Structs for different color spaces
struct BGR {
  uint8_t blue; // [0, 255]
  uint8_t green; // [0, 255]
  uint8_t red; // [0, 255]
};
struct HSI {
  uint16_t hue; // [0, 360]
  uint8_t saturation; // [0, 255]
  uint8_t intensity; // [0, 255]
};

constexpr float RAD_60 = M_PI / 3.0f;
constexpr float RAD_120 = 2.0f * M_PI / 3.0f;
constexpr float RAD_180 = M_PI;
constexpr float RAD_240 = 4.0f * M_PI / 3.0f;
constexpr float RAD_300 = 5.0f * M_PI / 3.0f;
constexpr float RAD_360 = 2.0f * M_PI;
constexpr float EPSILON = 1e-6f;// Small value to avoid division by zero

class Pixel {
public:
  Pixel() = delete;
  ~Pixel() = default;
  Pixel(const ColorSpace cs = ColorSpace::BGR) : colorSpace(cs) {
    if (cs == ColorSpace::BGR) {
      this->color = BGR{0, 0, 0}; // Default to black
    } else if (cs == ColorSpace::HSI) {
      this->color = HSI{0, 0, 0}; // Default to black
    }
  }

  Pixel(const uint8_t r, const uint8_t g, const uint8_t b)
    : color(BGR{b, g, r}), colorSpace(ColorSpace::BGR) {
  }

  Pixel(const Pixel& original)
    : color(original.color), colorSpace(original.colorSpace) {
  }
  Pixel(const BGR& bgr)
    : color(bgr), colorSpace(ColorSpace::BGR) {
  }
  Pixel(const HSI& hsi)
    : color(hsi), colorSpace(ColorSpace::HSI) {
  }

  void setColorSpace(const ColorSpace cs) {
    if (cs == this->colorSpace) { return; } // No change
    if (this->colorSpace == ColorSpace::BGR && cs == ColorSpace::HSI) {
      // Convert BGR to HSI
      BGR bgr = std::get<BGR>(this->color);
      // --------------- BEGIN_CITATION [4] ---------------- //
      // https://answers.opencv.org/question/62446/conversion-from-rgb-to-hsi/
      float numerator = bgr.red - 0.5f * ((bgr.red - bgr.green) + (bgr.red - bgr.blue));
      float denominator = std::sqrt(((bgr.red - bgr.green) * (bgr.red - bgr.green)) +
        ((bgr.red - bgr.blue) * (bgr.green - bgr.blue)));
      float theta = std::acos(numerator / (denominator + 1e-6f)); // [rad]
      const float sum = bgr.blue + bgr.green + bgr.red;
      float I = sum / 3.0f; // Intensity [0, 255]
      float S = (I == 0) ? 0.0f : 1.0f - 3.0f * (std::min({bgr.red, bgr.green, bgr.blue}) / (sum + EPSILON)); // Saturation [0, 1]
      float H = (bgr.green >= bgr.blue) ? theta : ((2.0f * M_PI) - theta); // Hue [0, pi] or [pi, 2pi]
      // Convert Hue to Degrees
      H *= (180.0f / M_PI); // Hue [0, 360]
      // Normalize Saturation to be between [0, 255]
      S *= 255.0f; // Saturation [0, 255]
      // --------------- END_CITATION [4] ---------------- //
      // Assign the HSI values to the pixel data
      this->color = HSI{static_cast<uint16_t>(H), static_cast<uint8_t>(S), static_cast<uint8_t>(I)};
    } else if (this->colorSpace == ColorSpace::HSI && cs == ColorSpace::BGR) {
      // Convert HSI to BGR
      HSI hsi = std::get<HSI>(this->color);
      float H = std::fmod(hsi.hue, 360.0f); // Hue [0, 360]
      float S = static_cast<float>(hsi.saturation) / 255.0f; // Saturation [0, 1]
      float I = static_cast<float>(hsi.intensity); // Intensity [0, 255]
      H *= (M_PI / 180.0f); // Convert to radians
      float B = I - I * S;
      float G = I - I * S;
      float R = I + 2 * I * S;
      if (H < RAD_120) {
        B = I - I * S;
        G = I + I * S * (1 - std::cos(H) / std::cos(RAD_60 - H));
        R = I + I * S * std::cos(H) / std::cos(RAD_60 - H);
      } else if (RAD_120 < H && H < RAD_240) {
        B = I + I * S * (1 - std::cos(H - RAD_120) / std::cos(RAD_180 - H));
        G = I + I * S * std::cos(H - RAD_120) / std::cos(RAD_180 - H);
        R = I - I * S;
      } else if (RAD_240 < H && H < RAD_360) {
        B = I + I * S * std::cos(H - RAD_240) / std::cos(RAD_300 - H);
        G = I - I * S;
        R = I + I * S * (1 - std::cos(H - RAD_240) / std::cos(RAD_300 - H));
      }
      // Clamp RGB values to [0, 255]
      B = std::clamp(B, 0.0f, 255.0f);
      G = std::clamp(G, 0.0f, 255.0f);
      R = std::clamp(R, 0.0f, 255.0f);
      // Assign the RGB values to the pixel data
      this->color = BGR{static_cast<uint8_t>(B), static_cast<uint8_t>(G), static_cast<uint8_t>(R)};
    }
  }

  ColorSpace getColorSpace() const {
    return this->colorSpace;
  }

  BGR getBGR() const {
    if (this->colorSpace == ColorSpace::BGR) {
      return std::get<BGR>(this->color);
    } else {
      throw std::runtime_error("Pixel is not in BGR color space");
    }
  }

  HSI getHSI() const {
    if (this->colorSpace == ColorSpace::HSI) {
      return std::get<HSI>(this->color);
    } else {
      throw std::runtime_error("Pixel is not in HSI color space");
    }
  }

public:
  std::string name() const {
    if (this->colorSpace == ColorSpace::BGR) {
      return "BGR";
    } else if (this->colorSpace == ColorSpace::HSI) {
      return "HSI";
    } else {
      throw std::runtime_error("Unsupported color space");
    }
  }

  bool operator==(const Pixel& other) const {
    if (this->colorSpace != other.colorSpace) { return false; }
    // Compare color values based on the color space
    if (this->colorSpace == ColorSpace::BGR) {
      BGR bgr = std::get<BGR>(this->color);
      BGR bgrOther = std::get<BGR>(other.color);
      return bgr.blue == bgrOther.blue && bgr.green == bgrOther.green && bgr.red == bgrOther.red;
    } else if (this->colorSpace == ColorSpace::HSI) {
      HSI hsi = std::get<HSI>(this->color);
      HSI hsiOther = std::get<HSI>(other.color);
      return hsi.hue == hsiOther.hue && hsi.saturation == hsiOther.saturation && hsi.intensity == hsiOther.intensity;
    } else {
      throw std::runtime_error("Unsupported color space");
    }
  }
  bool operator!=(const Pixel& other) const {
    return !(*this == other);
  }

  Pixel& operator=(const Pixel& other) {
    if (this != &other) {
      this->color = other.color;
      this->colorSpace = other.colorSpace;
    }
    return *this;
  }

private:
  std::variant<BGR, HSI> color;
  ColorSpace colorSpace;
};

#endif // !PIXEL_HPP