#ifndef TREEFINDER_TREE_FINDER_H
#define TREEFINDER_TREE_FINDER_H

#include <bag_of_leaves.h>
#include <svm_binary_classifier.h>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;

const int DEFAULT_DICT_SIZE = 120;
const SVM::KernelTypes DEFAULT_SVM_KERNEL = ml::SVM::RBF;
const float DEFAULT_MIN_CONF = 0.43;

const string TRAINING_PATH = "../data/images/training";


class TreeFinder {
    BagOfLeaves bag_of_leaves;
    SvmBinaryClassifier svm_binary_classifier;

public:

    void train(int dict_size = DEFAULT_DICT_SIZE, SVM::KernelTypes kernel_type = DEFAULT_SVM_KERNEL) {
        BagOfLeaves new_bag_of_leaves = BagOfLeaves();
        SvmBinaryClassifier new_svm_binary_classifier = SvmBinaryClassifier();

        vector<String> images_path;
        vector<Mat> train_images;
        Mat train_descriptors;
        Mat labels;

        glob(TRAINING_PATH + "/*.*",images_path);

        cout << "Training BOVW..." << endl;
        new_bag_of_leaves.train(images_path, dict_size);

        // parse labels - pattern: IMG_ID-LABEL.*
        int n_class_one = 0;
        for (const auto& path : images_path) {
            Mat desc = new_bag_of_leaves.computeBowDescriptorFromImage(imread(path));
            int label = atoi(path.substr(path.find('-') + 1, 1).c_str());

            if (desc.empty())
                continue;

            if (label == 1)
                n_class_one++;

            train_descriptors.push_back(desc);
            labels.push_back(label);
        }

        cout << "Training SVM - class one samples: " << n_class_one << "\t class zero samples: " << to_string(images_path.size() - n_class_one) << endl;
        new_svm_binary_classifier.train(train_descriptors, labels, kernel_type);

        // No errors at this point: set bow and classifier
        this->bag_of_leaves = new_bag_of_leaves;
        this->svm_binary_classifier = new_svm_binary_classifier;
    }

    bool containsTree(Mat img) {

        Mat desc = bag_of_leaves.computeBowDescriptorFromImage(img);
        float conf = 0;
        bool containsTree = svm_binary_classifier.getClass(desc, conf) > 0;

        return containsTree;
    }

    vector<Rect2i> locateTrees(Mat& img, float min_conf = DEFAULT_MIN_CONF) {

        vector<Rect2i> tree_locations;
        vector<KeyPoint> all_keypoints;
        Mat all_descriptors;

        bag_of_leaves.feature_detector->detect(img, all_keypoints);
        bag_of_leaves.descriptor_extractor->compute(img, all_keypoints, all_descriptors);


        for (int i = 0; i < 4; i++) {
            // define window size

            int win_cols = (1- (float)i/4) * img.cols;
            // int win_rows = (1- (float)i/4) * img.rows;
            int win_rows = (win_cols * 4)/3;
            if (win_rows > img.rows)
                win_rows = img.rows;

            int col_step = win_cols / 15;
            int row_step = win_rows / 15;

            if (col_step == 0)
                col_step = 1;
            if (row_step == 0)
                row_step = 1;

            /*
            // Show rectangles size
            Mat tmp = img.clone();
            Rect2i ROI(0, 0, win_cols, win_rows);
            rectangle(tmp, ROI, Scalar(255,0,0));
            imshow("a", tmp);
            waitKey(0);
            destroyAllWindows();
            */

            vector<Rect2i> boxes;
            vector<float> confidences;

            for (int row = 0; row + win_rows <= img.rows; row += row_step) {
                for (int col = 0; col + win_cols <= img.cols; col += col_step) {
                    Rect2i ROI(col, row, win_cols, win_rows);

                    /*
                    // Show moving window
                    Mat tmp = img.clone();
                    rectangle(tmp, ROI, Scalar(255,0,0));
                    imshow("a", tmp);
                    waitKey(0);
                    destroyAllWindows();
                    */

                    Mat ROI_descriptors = getDescriptorsInsideROI(ROI, all_keypoints, all_descriptors);

                    if (ROI_descriptors.empty())
                        continue;

                    Mat bow_desc = bag_of_leaves.computeBowDescriptor(ROI_descriptors);
                    float confidence;
                    int predicted = svm_binary_classifier.getClass(bow_desc, confidence);

                    if (predicted == 1) {
                        boxes.push_back(ROI);
                        confidences.push_back(confidence);
                    }
                }
            }

            // Apply non-maxima suppression
            vector<int> maxima_indexes;
            dnn::NMSBoxes(boxes, confidences, min_conf, 0, maxima_indexes);
            for (int mi : maxima_indexes) {
                tree_locations.push_back(boxes[mi]);
            }
        }

        /*
        namedWindow("aaa");
        imshow("aaa", img);
        waitKey(0);
        */

        return tree_locations;
    }

    static Mat getDescriptorsInsideROI(Rect2i ROI, const vector<KeyPoint>& all_keypoints, const Mat& all_descriptors) {
        Mat result;

        for (int i = 0; i < all_keypoints.size(); i++) {
            if (ROI.contains(all_keypoints[i].pt))
                result.push_back(all_descriptors.row(i));
        }

        return result;
    }

    static TreeFinder loadTreeFinder() {
        TreeFinder tree_finder;
        tree_finder.bag_of_leaves = BagOfLeaves::loadBagOfLeaves();
        tree_finder.svm_binary_classifier = SvmBinaryClassifier::loadSvmBinaryClassifier();

        return tree_finder;
    }

    void saveTreeFinder() {
        this->bag_of_leaves.saveBagOfLeaves();
        this->svm_binary_classifier.saveSvmBinaryClassifier();
    }

    float measureAccuracy(string test_path) {
        vector<String> images_path;
        vector<Mat> images;
        Mat descriptors;
        Mat true_labels;

        glob(TRAINING_PATH + "/*.*",images_path);

        for (const auto& p : images_path)
            images.push_back(imread(p));

        // parse labels - pattern: IMG_ID-LABEL.*
        for (const auto& path : images_path) {
            Mat desc = bag_of_leaves.computeBowDescriptorFromImage(imread(path));
            int label = atoi(path.substr(path.find('-') + 1, 1).c_str());

            descriptors.push_back(desc);
            true_labels.push_back(label);
        }

        int n_wrong = 0;

        for (int i = 0; i < descriptors.rows; i++) {
            float conf = 0;
            int predicted = svm_binary_classifier.getClass(descriptors.row(i), conf);

            cout << "Predicted: " << predicted << "\t True: " << true_labels.at<int>(i, 0) << "\t Conf: " << conf << "\t File name: " << images_path[i] << endl;

            if (predicted != true_labels.at<int>(i, 0))
                n_wrong++;
        }

        return (float)(images.size() - n_wrong) / images.size();
    }

    bool isTrained() {
        return bag_of_leaves.isTrained() && svm_binary_classifier.isTrained();
    }

private:

    static vector<Rect2i> removeFullyOverlapping(vector<Rect2i> rects) {
        vector<Rect2i> filtered;
        for (int i = 0; i < rects.size(); i++) {
            bool to_remove = false;
            int x = rects[i].x;
            int y = rects[i].y;
            int width = rects[i].width;
            int height = rects[i].height;

            Point p1(x, y);
            Point p2(x + width, y);
            Point p3(x, y + height);
            Point p4(x + width, y + height);

            for (int j = 0; j < rects.size(); j++) {
                if (j == i)
                    continue;
                if (rects[j].contains(p1)
                    && rects[j].contains(p2)
                    && rects[j].contains(p3)
                    && rects[j].contains(p4)) {
                    // rect i contained in j
                    to_remove = true;
                    break;
                }
            }

            if (!to_remove)
                filtered.push_back(rects[i]);
        }

        return filtered;
    }

};


#endif //TREEFINDER_TREE_FINDER_H
