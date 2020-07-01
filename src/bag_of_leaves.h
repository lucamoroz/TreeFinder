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

    void train(vector<Mat> train_images, int dictionary_size) {
        Mat unclustered_features = extractFeatures(train_images);

        // Compute codewords
        TermCriteria tc(TermCriteria::EPS | TermCriteria::MAX_ITER,100,0.001);
        BOWKMeansTrainer bow_trainer(dictionary_size, tc, 1, KMEANS_PP_CENTERS);
        Mat dictionary = bow_trainer.cluster(unclustered_features);

        if (dictionary.empty())
            throw runtime_error("BagOfLeaves - error learning dictionary");

        this->dictionary = dictionary;
    }

    Mat computeBowDescriptor(const Mat& img) {
        if (dictionary.empty())
            throw logic_error("BagOfLeaves - error computing descriptor: empty dictionary");

        // todo move this such that is done once - or try to uniform feature extraction and descr computation
        // Fast Library Approximate Nearest Neighbor matcher - todo check other matchers
        Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
        Ptr<FeatureDetector> detector(SIFT::create());
        Ptr<DescriptorExtractor> extractor(SIFT::create());

        BOWImgDescriptorExtractor bow_extractor(extractor, matcher);
        bow_extractor.setVocabulary(this->dictionary);

        vector<KeyPoint> keypoints;
        Mat bow_desc;

        detector->detect(img, keypoints);
        // Delegate to BoW descriptors computation, codewords matching and final bow desc. calculation
        bow_extractor.compute(img, keypoints, bow_desc);

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

    static Mat extractFeatures(const vector<Mat>& images) {
        // todo change for just one image
        Ptr<SIFT> extractor = SIFT::create();
        Mat all_features;

        for (const auto& img : images) {
            vector<KeyPoint> keypoints;
            Mat descriptor;

            extractor->detect(img, keypoints);
            extractor->compute(img, keypoints, descriptor);

            all_features.push_back(descriptor);
        }

        return all_features;
    }

    static bool fileExist(const char* filename) {
        ifstream infile(filename);
        return infile.good();
    }


};

#endif //TREEFINDER_BAGOFLEAVES_H