#include <iostream>
#include <fstream>
#include <random>
#include <cassert>
#include <unistd.h>

#include "header.h"
#include "neural.h"

int main(){

    int n_samp = 2000, in_d = 1, in_v = 28, in_h = 28;

    double **** x;
    x = new double***[n_samp];
    for(int s=0;s<n_samp;s++) x[s] = new double**[in_d];
    for(int s=0;s<n_samp;s++) for(int i=0;i<in_d;i++) x[s][i] = new double*[in_v];
    for(int s=0;s<n_samp;s++) for(int i=0;i<in_d;i++) for(int j=0;j<in_v;j++) x[s][i][j] = new double[in_h];
    double ** y;
    y = new double*[n_samp];
    for(int i=0;i<n_samp;i++){
        y[i] = new double[10];
        for(int j=0;j<10;j++) y[i][j] = 0;
    }

    ifstream file("mnist_train.csv");
    string str;
    for(int line=0;line<n_samp;line++){
        getline(file, str, ',');
        for(int i=0;i<10;i++) y[line][i] = 0;
        y[line][stoi(str)] = 1;
        for(int i=0;i<28*28;i++){
            getline(file, str, ',');
            x[line][0][(i-(i%28))/28][i%28] = stod(str)/255;
        }
    }
    file.close();

    int n_tr = 1000;
    double **** x_te;
    double ** y_te;
    x_te = new double***[n_samp-n_tr]; y_te = new double*[n_samp-n_tr];
    for(int i=0;i<n_samp-n_tr;i++){
        x_te[i] = new double**[in_d];
        for(int j=0;j<in_d;j++) x_te[i][j] = new double*[in_v];
        for(int j=0;j<in_d;j++) for(int k=0;k<in_v;k++) x_te[i][j][k] = new double[in_h];
        y_te[i] = new double[10];
    }
    for(int i=0;i<n_samp-n_tr;i++){
        for(int j=0;j<in_d;j++) for(int k=0;k<in_v;k++) for(int n=0;n<in_h;n++) x_te[i][j][k][n] = x[i+n_tr][j][k][n];
        for(int j=0;j<10;j++) y_te[i][j] = y[i+n_tr][j];
    }


    int depth = 5;
    layer L[depth];
    L[0].set("conv",3); L[0].f_v = 7; L[0].f_h = 7;
    L[1].set("conv",6); L[1].f_v = 5; L[1].f_h = 5;
    L[2].set("dense",10);
    L[3].set("dense",10); L[3].activ = id;
    L[4].set("softmax");

    network N({in_d,in_v,in_h},L,depth);
    N.loss_fnc = log_like;
    N.randomize();
    //N.read_from_file();

    N.train(x, y, n_tr, /*batch_size=*/50, /*learning_rate=*/.02, /*num_of_epochs=*/10);
    //N.print_to_file();

    double correct = 0, all = 0;
    int max = 0;
    for(int s=0;s<n_samp-n_tr;s++){
        all++;
        N.forward(x_te[s]);
        for(int i=0;i<10;i++) max = N.arch[depth-1].p_d[i]>N.arch[depth-1].p_d[max] ? i:max;
        correct += y_te[s][max];
    }
    cout << "Accuracy: " << 100*correct/all << "%\n";

    return 0;
}