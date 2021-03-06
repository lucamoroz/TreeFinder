#ifndef TREEFINDER_SVM_BINARY_CLASSIFIER_H
#define TREEFINDER_SVM_BINARY_CLASSIFIER_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::ml;

const char* SVM_PATH = "../data/svm_classifier.yml";
const SVM::KernelTypes DEFAULT_SVM_KERNEL = ml::SVM::RBF;

class SvmBinaryClassifier {
public:

    Ptr<SVM> svm = SVM::create();

    /***
     * Auto-train an SVM binary classifier with default param-grids.
     * @param training_data
     * @param labels must be 0 or 1
     */
    void train(const Mat& training_data, const Mat& labels) {
        for (int i = 0; i < labels.rows; i++) {
            int label = labels.at<int>(i,0);

            if ( !(label == 1 xor label == 0) )
                throw runtime_error("SvmBinaryClassifier - valid labels are '0' or '1'");
        }

        Ptr<SVM> s = SVM::create();
        s->setType(SVM::C_SVC);
        s->setKernel(DEFAULT_SVM_KERNEL);

        TermCriteria tc(TermCriteria::MAX_ITER, 300, 1e-6);
        s->setTermCriteria(tc);

        Ptr<TrainData> train_data = TrainData::create(training_data, ROW_SAMPLE, labels);
        s->trainAuto(train_data);

        if (!s->isTrained())
            throw runtime_error("SvmBinaryClassifier - Error training SVM");

        this->svm = s;
    }

    /**
     * Get the class predicted from the SVM and the output confidence with the prediction.
     * @param input
     * @param out_confidence
     * @return predicted class
     */
    int getClass(const Mat& input, float &out_confidence) {
        if (svm.empty() || !svm->isTrained()) {
            throw logic_error("SvmBinaryClassifier - cannot predict class: emtpy SVM");
        }

        float distance = svm->predict(input, noArray(), StatModel::Flags::RAW_OUTPUT);

        float conf = 1.0 / (1.0 + exp(-distance));
        out_confidence = conf;

        return this->svm->predict(input);
    }

    /***
     * Save current SVM state so it can be loaded.
     */
    void saveSvmBinaryClassifier() {
        this->svm->save(SVM_PATH);
    }

    /***
     * @return a previously saved SvmBinaryClassifier. If not found, an empty classifier is returned.
     */
    static SvmBinaryClassifier loadSvmBinaryClassifier() {
        SvmBinaryClassifier classifier;

        if (!fileExist(SVM_PATH)) {
            cout << "SvmBinaryClassifier - cannot load SVM: file not found" << endl;
            return classifier;
        }

        try {
            classifier.svm = SVM::load(SVM_PATH);
        } catch (Exception e) {
            cout << "SvmBinaryClassifier - error loading svm: " << e.msg << " ---- Returning empty classifier." << endl;
            return SvmBinaryClassifier();
        }

        return classifier;
    }

    bool isTrained() {
        return !svm.empty() && svm->isTrained();
    }

private:

    static bool fileExist(const char* filename) {
        ifstream infile(filename);
        return infile.good();
    }

};

#endif //TREEFINDER_SVM_BINARY_CLASSIFIER_H
