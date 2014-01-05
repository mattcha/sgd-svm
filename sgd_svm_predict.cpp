#include <iostream>
#include <fstream>
#include <vector>

#include "sgd_svm.h"

using namespace std;

int load_model(const char* modelname, Weight &weight)
{
    if (modelname == NULL) {
        fprintf(stderr, "modelname is null\n");
        return -1;
    }

    double lambda = 0.0;
    uint32_t iteration = 0;
    uint32_t w_size = 0;

    ifstream fin;
    fin.open(modelname);
    fin >> lambda >> iteration >> w_size;
    for (uint32_t wi=0; wi<w_size; ++wi) {
        weight.w_vec.push_back(0.0);
    }

    uint32_t idx = 0;
    double val = 0.0;
    while (fin) {
        fin >> idx >> val;
        weight.w_vec[idx] = val;
    }
    
    fprintf(stderr, "load model succ\n");
    return 0;
}

int predict(const char* filename, const Weight &weight)
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

    int eval_tp = 0;
    int eval_fn = 0;
    int eval_tn = 0;
    int eval_fp = 0;

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

        double result = inner_product(instance, weight);
        // output format : predict_result label score 

        if (instance.y_label >= 0 && result >= 0) 
            eval_tp++;
        if (instance.y_label < 0 && result < 0)
            eval_fn++;
        if (instance.y_label >= 0 && result < 0)
            eval_tn++;
        if (instance.y_label < 0 && result >= 0)
            eval_fp++;

        printf("%s\t%f\t%f\n", (result*instance.y_label > 0)?"right":"wrong", 
                instance.y_label, result);
    }

    fprintf(stderr, " 1 recall %f, precision %f\n", 
            eval_tp/(double)(eval_tp+eval_tn), eval_tp/(double)(eval_tp+eval_fp));
    fprintf(stderr, "-1 recall %f, precision %f\n",
            eval_fn/(double)(eval_fn+eval_tn), eval_fn/(double)(eval_fn+eval_fp));

    return 0;
}

int main(int argc, char** argv) 
{
    int ret;
    if (argc != 3) {
        fprintf(stderr, "Usage : %s modelname testfile\n", argv[0]);
        return 1;
    }

    char *modelname = argv[1];
    char *testfile = argv[2];

    Weight weight;
    ret = load_model(modelname, weight);
    if (ret != 0) {
        fprintf(stderr, "load model [%s] fail\n", modelname);
        return -1;
    }

    ret = predict(testfile, weight);
    if (ret != 0) {
        fprintf(stderr, "predict [%s] fail\n", testfile);
        return -1;
    }

    return 0;
}
