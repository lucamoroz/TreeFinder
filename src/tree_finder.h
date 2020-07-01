#ifndef TREEFINDER_TREE_FINDER_H
#define TREEFINDER_TREE_FINDER_H

#include <bag_of_leaves.h>
#include <svm_binary_classifier.h>

using namespace std;
using namespace cv;

const int DEFAULT_DICT_SIZE = 1000;
const SVM::KernelTypes DEFAULT_SVM_KERNEL = ml::SVM::RBF;
const string TRAINING_PATH = "../data/images";

class TreeFinder {
    BagOfLeaves bagOfLeaves;
    SvmBinaryClassifier svmBinaryClassifier;

public:

    void train(int dict_size, SVM::KernelTypes kernel_type) {
        BagOfLeaves newBagOfLeaves = BagOfLeaves();
        SvmBinaryClassifier newSvmBinaryClassifier = SvmBinaryClassifier();

        vector<String> images_path;
        vector<Mat> train_images;
        Mat train_descriptors;
        Mat labels;

        glob(TRAINING_PATH + "/*.*",images_path);

        for (const auto& p : images_path)
            train_images.push_back(imread(p));

        newBagOfLeaves.train(train_images, dict_size);

        // parse labels - pattern: IMG_ID-LABEL.*
        for (const auto& path : images_path) {
            Mat desc = newBagOfLeaves.computeBowDescriptor(imread(path));
            int label = atoi(path.substr(path.find('-') + 1, 1).c_str());

            train_descriptors.push_back(desc);
            labels.push_back(label);
        }

        newSvmBinaryClassifier.train(train_descriptors, labels, kernel_type);

        // No errors at this point: set bow and classifier
        this->bagOfLeaves = newBagOfLeaves;
        this->svmBinaryClassifier = newSvmBinaryClassifier;
    }

    bool containsTree(Mat img) {

        Mat desc = bagOfLeaves.computeBowDescriptor(img);
        bool containsTree = svmBinaryClassifier.getClass(desc) > 0;

        return containsTree;
    }

    vector<Rect2i> locateTrees(Mat img) {

    }

    static TreeFinder loadTreeFinder() {
        TreeFinder treeFinder;
        treeFinder.bagOfLeaves = BagOfLeaves::loadBagOfLeaves();
        treeFinder.svmBinaryClassifier = SvmBinaryClassifier::loadSvmBinaryClassifier();

        return treeFinder;
    }

    void saveTreeFinder() {
        this->bagOfLeaves.saveBagOfLeaves();
        this->svmBinaryClassifier.saveSvmBinaryClassifier();
    }

    bool isTrained() {
        return bagOfLeaves.isTrained() && svmBinaryClassifier.isTrained();
    }

};


#endif //TREEFINDER_TREE_FINDER_H
