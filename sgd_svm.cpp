#include <sstream>
#include <iostream>
#include "sgd_svm.h"

using namespace std;

double inner_product(const Instance &instance, const Weight &weight)
{
    uint32_t fea_size = instance.f_vec.size();
    double product = 0.0;

    for (uint32_t i = 0; i < fea_size; ++i) {
        const Feature &fea = instance.f_vec[i];
        product += fea.val * weight.w_vec[fea.idx];
    }

    return product;
}

int load_line(
        const char* line,
        Instance &instance,
        uint32_t &max_feature)
{
    double label;
    uint32_t fea_idx;
    double fea_val;

    stringstream ss(line);
    string field;
    if (getline(ss, field, ' ')) 
        label = atof(field.c_str());    
    else {
        fprintf(stderr, "read label fail [%s]\n",
                field.c_str());
        return -1;
    }
    instance.y_label = label;
    //cout << label << " ";

    while (true) {
        if (getline(ss, field, ':')) {
            fea_idx = atoi(field.c_str());
            //cout << fea_idx << ":";
        }
        else
            break;

        if (fea_idx > max_feature)
            max_feature = fea_idx;

        if (getline(ss, field, ' ')) {
            fea_val = atof(field.c_str());
            //cout << fea_val << " ";
        }
        else
            break;

        Feature fea;
        fea.idx = fea_idx;
        fea.val = fea_val;
        instance.f_vec.push_back(fea);
    }

    //cout << endl;
    return 0;
}

