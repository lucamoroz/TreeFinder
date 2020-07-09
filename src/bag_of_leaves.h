#ifndef TREEFINDER_BAGOFLEAVES_H
#define TREEFINDER_BAGOFLEAVES_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const string DICT_PATH = "../data/dictionary.yml";
const Size PREPROCESS_SIZE(1920, 960);

class BagOfLeaves {

public:

    Mat dictionary = Mat();
    Ptr<FeatureDetector> feature_detector = SiftFeatureDetector::create();
    Ptr<DescriptorExtractor> descriptor_extractor = SiftDescriptorExtractor::create();

    /***
     * Extract features and compute a vocabulary from a set of images.
     * @param images_path
     * @param dictionary_size
     */
    void train(const vector<string> &images_path, int dictionary_size) {
        cout << "BagOfLeaves - extracting features..." << endl;
        Mat unclustered_features = extractFeatures(images_path);

        // Extract vocabulary codewords
        TermCriteria tc(TermCriteria::EPS | TermCriteria::MAX_ITER,100,0.001);
        BOWKMeansTrainer bow_trainer(dictionary_size, tc, 1, KMEANS_PP_CENTERS);

        cout << "BagOfLeaves - extracting codewords..." << endl;
        Mat dict = bow_trainer.cluster(unclustered_features);

        if (dict.empty())
            throw runtime_error("BagOfLeaves - error learning dictionary");

        this->dictionary = dict;
    }

    /***
     * @return an object capable of extracting a BoW descriptor.
     * If not previously trained, throws a logic error.
     */
    BOWImgDescriptorExtractor getBowExtractor() {
        if (dictionary.empty())
            throw logic_error("BagOfLeaves - error creating bow extractor: empty dictionary");

        // Fast Library Approximate Nearest Neighbor matcher
        Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

        BOWImgDescriptorExtractor bow_extractor(matcher);
        bow_extractor.setVocabulary(this->dictionary);

        return bow_extractor;
    }

    /***
     * @param img input image
     * @return image's BoW descriptor
     */
    Mat computeBowDescriptorFromImage(const Mat& img) {
        Mat descriptors = extractFeatureDescriptors(img);
        return computeBowDescriptor(descriptors);
    }

    /**
     * @param feature_descriptors a set of feature descriptors consistent with the feature extractor
     * used for computing the vocabulary.
     * @return image's BoW descriptor
     */
    Mat computeBowDescriptor(const Mat& feature_descriptors) {
        Mat bow_desc;

        if (feature_descriptors.empty())
            return bow_desc;

        BOWImgDescriptorExtractor bow_extractor = getBowExtractor();
        bow_extractor.compute(feature_descriptors, bow_desc);

        return bow_desc;
    }

    /***
     * Saves the current state of the vocabulary so it can be loaded later.
     */
    void saveBagOfLeaves() {
        FileStorage fs(DICT_PATH, FileStorage::WRITE);
        fs << "vocabulary" << this->dictionary;
        fs.release();
    }

    /***
     * @return the previously saved vocabulary. If the vocabulary doesn't exists, returns an empty BagOfLeaves.
     */
    static BagOfLeaves loadBagOfLeaves() {
        BagOfLeaves bow;

        if (!fileExist(DICT_PATH.c_str())) {
            cout << "BagOfLeaves - cannot load dictionary: dictionary not found." << endl;
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

    /**
     * @param img input image
     * @param keypoints features extracted according with the feature extractor
     * @return descriptors of the extracted keypoints
     */
    Mat extractFeatureDescriptors(const Mat& img, vector<KeyPoint> &keypoints) {
        Mat descriptor;

        Mat prep = preprocessImg(img);

        this->feature_detector->detect(prep, keypoints);
        this->descriptor_extractor->compute(prep, keypoints, descriptor);

        return descriptor;
    }

    /***
     * @param img input image
     * @return descriptors found in the image, according with the feature & descriptor extractors.
     */
    Mat extractFeatureDescriptors(const Mat& img) {
        vector<KeyPoint> unused;
        return extractFeatureDescriptors(img, unused);
    }

    static Mat preprocessImg(const Mat &img) {
        Mat res;
        // Resize using default bilinear interpolation
        resize(img, res, PREPROCESS_SIZE);
        return res;
    }

private:

    /**
     * Util method that extracts a Mat of features from a folder of images.
     */
    Mat extractFeatures(const vector<string> images_path) {
        Mat all_features;
        Mat descriptor;
        Mat img;

        for (const auto& path : images_path) {
            img = imread(path);
            descriptor = extractFeatureDescriptors(img);
            if (!descriptor.empty()) {
                all_features.push_back(descriptor);
            }

            img.release();
        }

        return all_features;
    }

    static bool fileExist(const char* filename) {
        ifstream infile(filename);
        return infile.good();
    }


};

#endif //TREEFINDER_BAGOFLEAVES_H