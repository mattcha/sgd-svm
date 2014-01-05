make
./sgd_svm_train 20 0.001 train.txt model
./sgd_svm_predict model train.txt > result
