
all:
	g++ -c *.cpp 
	g++ -o sgd_svm_train sgd_svm_train.o sgd_svm.o 
	g++ -o sgd_svm_predict sgd_svm_predict.o sgd_svm.o
clean:
	rm *.o
	rm sgd_svm_train sgd_svm_predict
