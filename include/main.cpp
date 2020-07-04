#include <iostream>
#include <tree_finder.h>

using namespace std;


int main() {
    TreeFinder treeFinder = TreeFinder::loadTreeFinder();

    if (!treeFinder.isTrained()) {
        cout << "Training TreeFinder..." << endl;
        treeFinder.train();
        treeFinder.saveTreeFinder();
    } else {
        cout << "Success loading TreeFinder" << endl;
    }

    //cout << "Accuracy: " << treeFinder.measureAccuracy(TRAINING_PATH) << endl;

    // cout << "Tree: " << treeFinder.containsTree(imread("../data/images/10-0.jpg")) << endl;

    vector<string> images_path;
    glob("../data/images/test/*.*", images_path);
    vector<Mat> result;

    for (int i = 0; i < images_path.size(); i++) {
        string path = images_path[i];

        Mat img = imread(path);
        vector<float> confidences;
        vector<Rect2i> locations = treeFinder.locateTrees(img);

        cout << "Locations found: " << result.size() << endl;

        for (int j = 0; j < locations.size(); j++) {
            rectangle(img, locations[j], Scalar(250,255,0), 3);
        }

        result.push_back(img);
    }

    for (auto img : result) {
        namedWindow("Res", WINDOW_NORMAL);
        imshow("Res", img);
        waitKey(0);
    }


    destroyAllWindows();

    return 0;
}