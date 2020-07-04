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

class SvmBinaryClassifier {

public:

    Ptr<SVM> svm = SVM::create();

    void train(const Mat& training_data, const Mat& labels, SVM::KernelTypes kernel_type) {
        for (int i = 0; i < labels.rows; i++) {
            int label = labels.at<int>(i,0);

            if ( !(label == 1 xor label == 0) )
                throw runtime_error("SvmBinaryClassifier - valid labels are '0' or '1'");
        }

        Ptr<SVM> s = SVM::create();
        s->setType(SVM::C_SVC);
        s->setKernel(kernel_type);

        TermCriteria tc(TermCriteria::MAX_ITER, 100, 1e-6);
        s->setTermCriteria(tc);

        Ptr<TrainData> train_data = TrainData::create(training_data, ROW_SAMPLE, labels);
        s->trainAuto(train_data);

        if (!s->isTrained())
            throw runtime_error("SvmBinaryClassifier - Error training SVM");

        this->svm = s;
    }

    int getClass(const Mat& input, float &out_confidence) {
        if (svm.empty() || !svm->isTrained()) {
            throw logic_error("SvmBinaryClassifier - cannot predict class: emtpy SVM");
        }

        float distance = svm->predict(input, noArray(), StatModel::Flags::RAW_OUTPUT);

        float conf = 1.0 / (1.0 + exp(-distance));
        out_confidence = conf;

        return this->svm->predict(input);
    }

    void saveSvmBinaryClassifier() {
        this->svm->save(SVM_PATH);
    }

    // Returns empty SvmBinaryClassifier if not found
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
