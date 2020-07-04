#ifndef TREEFINDER_BAGOFLEAVES_H
#define TREEFINDER_BAGOFLEAVES_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const string DICT_PATH = "../data/dictionary.yml";

class BagOfLeaves {

public:

    Mat dictionary = Mat();
    Ptr<FeatureDetector> feature_detector = SiftFeatureDetector::create();
    Ptr<DescriptorExtractor> descriptor_extractor = SiftDescriptorExtractor::create();

    void train(vector<Mat> train_images, int dictionary_size) {
        Mat unclustered_features = extractFeatures(train_images);

        // Compute codewords
        TermCriteria tc(TermCriteria::EPS | TermCriteria::MAX_ITER,100,0.001);
        BOWKMeansTrainer bow_trainer(dictionary_size, tc, 1, KMEANS_PP_CENTERS);

        Mat dict = bow_trainer.cluster(unclustered_features);

        if (dict.empty())
            throw runtime_error("BagOfLeaves - error learning dictionary");

        this->dictionary = dict;
    }

    BOWImgDescriptorExtractor getBowExtractor() {
        if (dictionary.empty())
            throw logic_error("BagOfLeaves - error creating bow extractor: empty dictionary");

        // Fast Library Approximate Nearest Neighbor matcher - todo check other matchers
        Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

        BOWImgDescriptorExtractor bow_extractor(matcher);
        bow_extractor.setVocabulary(this->dictionary);

        return bow_extractor;
    }

    Mat computeBowDescriptorFromImage(const Mat& img) {
        BOWImgDescriptorExtractor bow_extractor = getBowExtractor();
        vector<KeyPoint> keypoints;
        Mat descriptors;
        Mat bow_desc;

        feature_detector->detect(img, keypoints);
        descriptor_extractor->compute(img, keypoints, descriptors);

        return computeBowDescriptor(descriptors);
    }

    Mat computeBowDescriptor(const Mat& descriptors) {
        Mat bow_desc;

        BOWImgDescriptorExtractor bow_extractor = getBowExtractor();
        bow_extractor.compute(descriptors, bow_desc);

        return bow_desc;
    }

    void saveBagOfLeaves() {
        FileStorage fs(DICT_PATH, FileStorage::WRITE);
        fs << "vocabulary" << this->dictionary;
        fs.release();
    }

    // Returns empty bag of leaves if not found
    static BagOfLeaves loadBagOfLeaves() {
        BagOfLeaves bow;

        if (!fileExist(DICT_PATH.c_str())) {
            cout << "Bag of leaves - cannot load dictionary: dictionary not found." << endl;
            return bow;
        }

        Mat dictionary;
        FileStorage fs(DICT_PATH, FileStorage::READ);
        fs["vocabulary"] >> dictionary;
        fs.release();

        bow.dictionary = dictionary;

        return bow;
    }

    bool isTrained() {
        return !dictionary.empty();
    }

private:

    Mat extractFeatures(const vector<Mat>& images) {
        Mat all_features;

        for (const auto& img : images) {
            Mat descriptor = extractFeatures(img);
            if (!descriptor.empty()) {
                all_features.push_back(descriptor);
            }
        }

        return all_features;
    }

    Mat extractFeatures(const Mat& img) {
        Mat descriptor;
        vector<KeyPoint> keypoints;

        feature_detector->detect(img, keypoints);
        descriptor_extractor->compute(img, keypoints, descriptor);

        return descriptor;
    }

    static bool fileExist(const char* filename) {
        ifstream infile(filename);
        return infile.good();
    }


};

#endif //TREEFINDER_BAGOFLEAVES_H