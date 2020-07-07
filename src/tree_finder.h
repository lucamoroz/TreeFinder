#ifndef TREEFINDER_TREE_FINDER_H
#define TREEFINDER_TREE_FINDER_H

#include <bag_of_leaves.h>
#include <svm_binary_classifier.h>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;

const int DEFAULT_DICT_SIZE = 110;
const SVM::KernelTypes DEFAULT_SVM_KERNEL = ml::SVM::RBF;
const float DEFAULT_MIN_CONF = 0.48;

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

    vector<Rect2i> locateTrees2(Mat& img, float min_conf = DEFAULT_MIN_CONF) {
        vector<Rect2i> tree_locations;

        // Sliding window using 4 window sizes (proportional to the input image)
        for (const auto& win_size : getWindowsSizes(img)) {

            int col_step = win_size.width / 10;
            int row_step = win_size.height / 10;

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

            for (int row = 0; row + win_size.height <= img.rows; row += row_step) {
                for (int col = 0; col + win_size.width <= img.cols; col += col_step) {
                    Rect2i ROI(Point2i(col, row), win_size);

                    Mat window = img(ROI);
                    Mat bow_desc = bag_of_leaves.computeBowDescriptorFromImage(window);

                    /*
                    // Show moving window
                    cout << "Features in ROI: " << bag_of_leaves.extractFeatureDescriptors(window).rows << endl;
                    Mat tmp = img.clone();
                    rectangle(tmp, ROI, Scalar(255,0,0), 3);
                    namedWindow("test", WINDOW_NORMAL);
                    imshow("test", tmp);
                    waitKey(0);
                    destroyAllWindows();
                    */

                    if (bow_desc.empty())
                        continue;

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

        return removeFullyOverlapping(tree_locations);

    }

    vector<Rect2i> locateTrees(Mat& img, float min_conf = DEFAULT_MIN_CONF) {
        vector<Rect2i> tree_locations;

        vector<KeyPoint> all_keypoints;
        Mat all_descriptors;

        bag_of_leaves.feature_detector->detect(img, all_keypoints);
        bag_of_leaves.descriptor_extractor->compute(img, all_keypoints, all_descriptors);

        for (auto &win_size : getWindowsSizes(img)) {

            int col_step = win_size.width / 15;
            int row_step = win_size.height / 15;

            if (col_step == 0)
                col_step = 1;
            if (row_step == 0)
                row_step = 1;

            /*
            // Show rectangles size
            Mat tmp = img.clone();
            Rect2i ROI(Point2i(0,0), win_size);
            rectangle(tmp, ROI, Scalar(255,0,0), 3);
            namedWindow("a", WINDOW_NORMAL);
            imshow("a", tmp);
            waitKey(0);
            destroyAllWindows();
            */

            vector<Rect2i> boxes;
            vector<float> confidences;

            for (int row = 0; row + win_size.height <= img.rows; row += row_step) {
                for (int col = 0; col + win_size.width <= img.cols; col += col_step) {
                    Rect2i ROI(Point2i(col, row), win_size);

                    Mat ROI_descriptors = getDescriptorsInsideROI(ROI, all_keypoints, all_descriptors);

                    /*
                    // Show moving window
                    cout << "Features in ROI: " << ROI_descriptors.rows << endl;
                    Mat tmp = img.clone();
                    rectangle(tmp, ROI, Scalar(255,0,0), 3);
                    namedWindow("test", WINDOW_NORMAL);
                    imshow("test", tmp);
                    waitKey(0);
                    destroyAllWindows();
                    */

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

    vector<Size2i> getWindowsSizes(const Mat& img) {
        vector<Size2i> sizes;

        int win1_cols = (int) (0.25 * (float) img.cols);
        int win1_rows = (win1_cols * 4)/3;
        if (win1_rows > img.rows)
            win1_rows = img.rows;

        int win2_cols = (int) (0.4 * (float) img.cols);
        int win2_rows = (win2_cols * 4)/3;
        if (win2_rows > img.rows)
            win2_rows = img.rows;

        int win3_cols = (int) (0.6 * (float) img.cols);
        int win3_rows = (win3_cols * 4)/3;
        if (win3_rows > img.rows)
            win3_rows = img.rows;

        int win4_cols = (int) (0.8 * (float) img.cols);
        int win4_rows = (win4_cols * 4)/3;
        if (win4_rows > img.rows)
            win4_rows = img.rows;

        sizes.push_back(Size(win4_cols, win4_rows));
        sizes.push_back(Size(win3_cols, win3_rows));
        sizes.push_back(Size(win2_cols, win2_rows));
        sizes.push_back(Size(win1_cols, win1_rows));

        return sizes;
    }

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
