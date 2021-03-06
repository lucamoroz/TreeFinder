#include <iostream>
#include <tree_finder.h>

using namespace std;

int main(int argc, char* argv[]) {
    TreeFinder treeFinder = TreeFinder::loadTreeFinder();

    if (!treeFinder.isTrained()) {
        cout << "Training TreeFinder..." << endl;
        treeFinder.train();
        treeFinder.saveTreeFinder();
    } else {
        cout << "Success loading TreeFinder" << endl;
    }

    if (argc < 2 || argc > 3) {
        cout << "USAGE: ./TreeFinder IMG_PATH [OPTIONAL MIN_CONF]" << endl;
        return 1;
    }

    string img_path = argv[1];

    float min_conf = DEFAULT_MIN_CONF;
    if (argc == 3)
        min_conf = stof(argv[2]);

    Mat img = imread(img_path);

    vector<Rect2i> locations = treeFinder.locateTrees(img, min_conf);

    // Draw found locations
    for (int j = 0; j < locations.size(); j++) {
        rectangle(img, locations[j], Scalar(250,255,0), 3);
    }

    namedWindow("Result", WINDOW_NORMAL);
    imshow("Result", img);
    waitKey(0);

    destroyAllWindows();

    return 0;
}