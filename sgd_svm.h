#ifndef _SGD_SVM_H
#define _SGD_SVM_H

#include <vector>

using namespace std;

struct Param {
    uint32_t iteration;
    double lambda;
    double eta;
    uint32_t step;
    double loss_hinge;
    double loss_l2;
    uint32_t right;
};

struct Feature {
    uint32_t idx;
    double val;
};

struct Instance {
    double y_label;
    vector<Feature> f_vec;
};

struct Dataset {
    vector<Instance> ins_vec;
};

struct Weight {
    vector<double> w_vec;
};

int load_line(
        const char* line, 
        Instance &instance,
        uint32_t &max_feature);

double inner_product(
        const Instance &instance, 
        const Weight &weight);

#endif
