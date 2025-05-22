#include <iostream>
#include <fstream>
#include <random>
#include <cassert>
#include <chrono>
#include <unistd.h>

#include "header.h"
#include "neural_new.h"

int main(){

    int n_samp = 60000, in_d = 1, in_v = 28, in_h = 28;

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

    int n_tr = 55000;
    double **** x_tr, **** x_te;
    double ** y_tr, ** y_te;
    x_tr = new double***[n_tr]; y_tr = new double*[n_tr];
    for(int i=0;i<n_tr;i++){
        x_tr[i] = new double**[in_d];
        for(int j=0;j<in_d;j++) x_tr[i][j] = new double*[in_v];
        for(int j=0;j<in_d;j++) for(int k=0;k<in_v;k++) x_tr[i][j][k] = new double[in_h];
        y_tr[i] = new double[10];
    }
    x_te = new double***[n_samp-n_tr]; y_te = new double*[n_samp-n_tr];
    for(int i=0;i<n_samp-n_tr;i++){
        x_te[i] = new double**[in_d];
        for(int j=0;j<in_d;j++) x_te[i][j] = new double*[in_v];
        for(int j=0;j<in_d;j++) for(int k=0;k<in_v;k++) x_te[i][j][k] = new double[in_h];
        y_te[i] = new double[10];
    }
    for(int i=0;i<n_tr;i++){
        for(int j=0;j<in_d;j++) for(int k=0;k<in_v;k++) for(int n=0;n<in_h;n++) x_tr[i][j][k][n] = x[i][j][k][n];
        for(int j=0;j<10;j++) y_tr[i][j] = y[i][j];
    }
    for(int i=0;i<n_samp-n_tr;i++){
        for(int j=0;j<in_d;j++) for(int k=0;k<in_v;k++) for(int n=0;n<in_h;n++) x_te[i][j][k][n] = x[i+n_tr][j][k][n];
        for(int j=0;j<10;j++) y_te[i][j] = y[i+n_tr][j];
    }


    int depth = 5;
    layer L[depth];
    L[0].set("conv",3); L[0].f_v = 7; L[0].f_h = 7;
    L[1].set("conv",6); L[1].f_v = 5; L[1].f_h = 5;
    //L[2].set("conv",9); L[2].f_v = 3; L[2].f_h = 3;
    L[2].set("dense",10);
    L[3].set("dense",10); L[3].activ = id;
    //L[3].set("dense",10); L[3].activ = id;
    L[4].set("softmax");

    network N({in_d,in_v,in_h},L,depth);
    N.loss_fnc = log_like;
    //N.randomize();
    N.read_from_file();

    steady_clock::time_point begin = steady_clock::now();
    N.train(x_tr, y_tr, n_tr, 500, .02, 2);
    steady_clock::time_point end = steady_clock::now();
    cout << "Training time: " << duration_cast<milliseconds>(end - begin).count() << " ms\n" << endl;
    N.print_to_file();

    double correct = 0, all = 0;
    int max = 0;
    for(int s=0;s<n_samp-n_tr;s++){
        all++;
        N.forward(x_te[s]);
        for(int i=0;i<10;i++) max = N.arch[depth-1].p_d[i]>N.arch[depth-1].p_d[max] ? i:max;
        correct += (y_te[s][max] == 1);
    }
    cout << "Accuracy: " << 100*correct/all << "%\n";

    cout << "set_mv_to_zero: " << t_mv/1000000000. << " s" << endl;
    cout << "set_j_to_zero: " << t_j/1000000000. << " s" << endl;
    cout << "forward: " << t_forw/1000000000. << " s" << endl;
    //cout << "backward: " << t_back/1000000000. << " s" << endl;
    cout << "update_derivatives: " << t_update/1000000000. << " s" << endl;
    cout << "run_adam: " << t_adam/1000000000. << " s" << endl;




    return 0;
}

int main_old(){

    int n_samp = 20, i_d = 2, i_v = 7, i_h = 9, n_o = 3;
    double **** x;
    x = new double***[n_samp];
    for(int i=0;i<n_samp;i++) x[i] = new double**[i_d];
    for(int i=0;i<n_samp;i++) for(int j=0;j<i_d;j++) x[i][j] = new double*[i_v];
    for(int i=0;i<n_samp;i++) for(int j=0;j<i_d;j++) for(int k=0;k<i_v;k++) x[i][j][k] = new double[i_h];
    for(int i=0;i<n_samp;i++) for(int j=0;j<i_d;j++) for(int k=0;k<i_v;k++) for(int l=0;l<i_h;l++) x[i][j][k][l] = rand_U(2);


    double ** y;
    y = new double*[n_samp];
    for(int i=0;i<n_samp;i++) y[i] = new double[n_o];
    for(int i=0;i<n_samp;i++) for(int j=0;j<n_o;j++) y[i][j] = 1+rand_U(1);



    int depth = 5;
    layer L[depth];
    L[0].set("conv",3); L[0].f_v = 2; L[0].stride_h = 2; 
    L[1].set("conv",3); L[1].stride_v = 3;
    L[2].set("dense",4);
    L[3].set("dense",3);
    L[4].set("softmax");

    network N({i_d,i_v,i_h},L,depth);
    N.loss_fnc = least_sq;
    N.randomize();

    N.train(x,y,n_samp,n_samp,.1,100);



    return 0;
}