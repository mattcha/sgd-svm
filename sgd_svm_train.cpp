#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cmath>

#include "sgd_svm.h"

using namespace std;

int load_data(const char* filename,
        Dataset &dataset,
        Weight &weight)
{
    if (filename == NULL) {
        fprintf(stderr, "filename is null\n", filename);
        return -1;
    }

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "can't open filename [%s]\n", filename);
        return -1;
    }

    char *buff = NULL;
    size_t buff_len = 0;

    size_t lineno = 0;
    uint32_t max_feature = 0;

    for (ssize_t read = 0; 
            (read = getline(&buff, &buff_len, fp)) != -1; 
            ++lineno) {
        if(read > 1 && buff[read-2] == '\r') {
            buff[read-2] = '\0';
        } else if (read > 0 && (buff[read-1] == '\r' || buff[read-1] == '\n')) {
            buff[read-1] = '\0';
        }

        Instance instance;
        int ret = load_line(buff, instance, max_feature);
        if (ret != 0) {
            fprintf(stderr, "load line [%s] fail\n", buff);
            break;
        }
        dataset.ins_vec.push_back(instance);
    }

    fprintf(stderr, "Line number: [%u]\n", lineno);
    fprintf(stderr, "Max feature: [%u]\n", max_feature);

    for (size_t i = 0; i <= max_feature; ++i) {
        weight.w_vec.push_back(0.0);
    }

    if (fp)
        fclose(fp);
    if (buff)
        free(buff);

    return 0;
}

int step_train(const Instance &instance, Weight &weight, Param &param)
{
    uint32_t fea_size = instance.f_vec.size();
    // update eta
    param.eta = 1.0 / (param.lambda * sqrt(param.step));
    
    double y_score = inner_product(instance, weight);
    double y_label = instance.y_label;

    double loss = 1 - y_label * y_score;

    for (uint32_t fi=0; fi<fea_size; ++fi) {
        const Feature &fea = instance.f_vec[fi];
        double w_i = weight.w_vec[fea.idx];

        // compute l2 loss
        param.loss_l2 += param.lambda * w_i * w_i;

        // wrong classfy
        // iterate all instance feature
        // w = w - eta*(lambda*w - y_label*x)
        if (loss > 0) {
            w_i = w_i - param.eta*(param.lambda*w_i - y_label*fea.val);
        }
        // right classfy
        // as loss > 0
        // w = w - eta*lambda*w
        if (loss < 0) {
            w_i = w_i - param.eta*param.lambda*w_i;
        }

        weight.w_vec[fea.idx] = w_i;
    }

    if (loss > 0) {
        param.loss_hinge += loss;
    }

    if (loss <= 0) 
        return 1;
    return 0;
}

int train(Dataset &dataset, Weight &weight, Param &param)
{
    uint32_t ins_num = dataset.ins_vec.size();

    for (uint32_t i_iter = 0; i_iter < param.iteration; ++i_iter) {
        param.right = 0;
        param.loss_hinge = 0;
        param.loss_l2 = 0;

        for (uint32_t i_ins = 0; i_ins < ins_num; ++i_ins) {
            ++ param.step;

            int r_idx = rand() % (ins_num - i_ins);
            Instance inst = dataset.ins_vec[i_ins + r_idx];
            dataset.ins_vec[i_ins + r_idx] = dataset.ins_vec[i_ins];
            dataset.ins_vec[i_ins] = inst;

            param.right += step_train(inst, weight, param);
        }
        fprintf(stderr, "iter : %-5u   accurate : %-8f  hinge loss : %-8f   l2 loss : %-8f\n", 
                i_iter, param.right*100.0/ins_num, param.loss_hinge/ins_num, param.loss_l2/ins_num);
    }

    fprintf(stderr, "finish training\n");

    return 0;
}

int save_model(const char* modelname, const Weight &weight, const Param &param)
{
    if (modelname == NULL) {
        fprintf(stderr, "model name is null\n");
        return -1;
    }

    FILE *fp = fopen(modelname, "w");
    if (fp == NULL) {
        fprintf(stderr, "open model file fail [%s]\n", modelname);
        return -1;
    }

    uint32_t w_size = weight.w_vec.size();
    fprintf(fp, "%f %u %u\n", param.lambda, param.iteration, w_size);

    for (uint32_t i=0; i<w_size; ++i) {
        if (weight.w_vec[i]==0) {
            continue;
        }
        fprintf(fp, "%u\t%f\n", i, weight.w_vec[i]);
    }

    fprintf(stderr, "save model succ\n");
    return 0;
}

int main(int argc, char** argv)
{
    Dataset dataset;
    Weight weight;
    Param param;

    if (argc != 5) {
        fprintf(stderr, 
                "Usage : %s iteration lambda trainname modelname\n", argv[0]);
        return 1;
    }

    param.iteration = atoi(argv[1]);
    param.lambda = atof(argv[2]);

    char *trainname = argv[3];
    char *modelname = argv[4];

    param.step = 0;

    int ret = load_data(trainname, dataset, weight);
    if (ret != 0) {
        fprintf(stderr, "load data [%s] fail\n", argv[1]);
        return -1;
    }

    ret = train(dataset, weight, param);
    if (ret != 0) {
        fprintf(stderr, "train fail\n");
        return -1;
    }

    ret = save_model(modelname, weight, param);
    if (ret != 0) {
        fprintf(stderr, "save model fail\n");
        return -1;
    }
    
    return 0;
}

