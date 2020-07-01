#include <iostream>
#include <tree_finder.h>

using namespace std;


int main() {
    TreeFinder treeFinder = TreeFinder::loadTreeFinder();

    if (!treeFinder.isTrained()) {
        cout << "Training TreeFinder..." << endl;
        treeFinder.train(5, cv::ml::SVM::RBF);
        treeFinder.saveTreeFinder();
    } else {
        cout << "Success loading TreeFinder" << endl;
    }

    cout << "Tree: " << treeFinder.containsTree(imread("../data/images/2-1.jpg")) << endl;

    return 0;
}