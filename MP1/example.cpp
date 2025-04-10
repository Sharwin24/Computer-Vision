#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Function to perform the first pass of CCL
void firstPass(const vector<vector<int>>& binaryImage, vector<vector<int>>& labeledImage, vector<int>& parent) {
  int nextLabel = 1;
  int rows = binaryImage.size();
  int cols = binaryImage[0].size();

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (binaryImage[i][j] == 1) {
        int upNeighbor = (i > 0) ? labeledImage[i - 1][j] : 0;
        int leftNeighbor = (j > 0) ? labeledImage[i][j - 1] : 0;

        if (upNeighbor == 0 && leftNeighbor == 0) {
          labeledImage[i][j] = nextLabel;
          parent[nextLabel] = nextLabel;
          nextLabel++;
        } else if (upNeighbor != 0 && leftNeighbor == 0) {
          labeledImage[i][j] = upNeighbor;
        } else if (upNeighbor == 0 && leftNeighbor != 0) {
          labeledImage[i][j] = leftNeighbor;
        } else {
          labeledImage[i][j] = min(upNeighbor, leftNeighbor);
          int root1 = upNeighbor;
          while (parent[root1] != root1) {
            root1 = parent[root1];
          }
          int root2 = leftNeighbor;
          while (parent[root2] != root2) {
            root2 = parent[root2];
          }
          parent[max(root1, root2)] = min(root1, root2);
        }
      }
    }
  }
}

// Function to find the root of a label
int findRoot(vector<int>& parent, int label) {
  while (parent[label] != label) {
    label = parent[label];
  }
  return label;
}

// Function to perform the second pass of CCL
void secondPass(vector<vector<int>>& labeledImage, vector<int>& parent) {
  int rows = labeledImage.size();
  int cols = labeledImage[0].size();

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (labeledImage[i][j] != 0) {
        labeledImage[i][j] = findRoot(parent, labeledImage[i][j]);
      }
    }
  }
}

int main() {
  // Example binary image
  vector<vector<int>> binaryImage = {
      {0, 1, 1, 0, 0},
      {1, 1, 0, 1, 0},
      {0, 1, 0, 1, 0},
      {0, 0, 1, 0, 1},
      {0, 0, 1, 1, 1}
  };

  int rows = binaryImage.size();
  int cols = binaryImage[0].size();

  // Initialize labeled image and parent vector
  vector<vector<int>> labeledImage(rows, vector<int>(cols, 0));
  vector<int> parent(rows * cols);

  // Perform CCL
  firstPass(binaryImage, labeledImage, parent);
  secondPass(labeledImage, parent);

  // Print the labeled image
  for (const auto& row : labeledImage) {
    for (int label : row) {
      cout << label << " ";
    }
    cout << endl;
  }

  return 0;
}