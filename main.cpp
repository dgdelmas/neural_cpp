#include <iostream>
#include <fstream>
#include <random>
#include <cassert>
#include <unistd.h>

#include "header.h"
#include "neural.h"

int main(){

    int n_samp = 10000, in_d = 3, in_v = 32, in_h = 32;

    double **** x;
    x = new double***[n_samp];
    for(int s=0;s<n_samp;s++) x[s] = new double**[in_d];
    for(int s=0;s<n_samp;s++) for(int i=0;i<in_d;i++) x[s][i] = new double*[in_v];
    for(int s=0;s<n_samp;s++) for(int i=0;i<in_d;i++) for(int j=0;j<in_v;j++) x[s][i][j] = new double[in_h];
    double ** y;
    y = new double*[n_samp];
    for(int i=0;i<n_samp;i++) y[i] = new double[10];

    ifstream file1("myfile.txt");
    string str;
    for(int line=0;line<n_samp;line++){
        for(int j=0;j<in_d;j++){
            for(int i=0;i<in_v*in_h;i++){
                getline(file1, str, ',');
                x[line][j][(i-(i%in_h))/in_v][i%in_h] = stod(str)/255.;
            }
        }
    }
    file1.close();
    ifstream file2("myfile_label.txt");
    for(int line=0;line<n_samp;line++){
        getline(file2, str, ',');
        for(int i=0;i<10;i++) y[line][i] = 0;
        y[line][stoi(str)] = 1;
    }
    file2.close();

    // cout << "{";
    // for(int i=0;i<3;i++){
    //     cout << "{";
    //     for(int j=0;j<32;j++){
    //         cout << "{";
    //         for(int k=0;k<32;k++){
    //             cout << x[1][i][j][k] << (k<31?",":"}");
    //         }
    //         cout << (j<31?",":"}");
    //     }
    //     cout << (i<2?",":"}");
    // }
    // cout << endl;
    //for(int i=0;i<10;i++) cout << y[30][i] << ",";

    int depth = 5;
    layer L[depth];
    L[0].set("conv",3); L[0].f_v = 3; L[0].f_h = 3;
    L[1].set("conv",4); L[1].f_v = 6; L[1].f_h = 6; L[1].stride_h = 2; L[1].stride_v = 2;
    //L[1].set("conv",6); L[1].f_v = 5; L[1].f_h = 5;
    //L[2].set("dense",10);
    //L[3].set("dense",10); L[3].activ = id;
    //L[4].set("softmax");
    //L[0].set("dense",10);
    //L[0].set("dense",60);
    L[2].set("dense",30);
    L[3].set("dense",10); L[3].activ = id;
    L[4].set("softmax");

    network N({in_d,in_v,in_h},L,depth);
    N.loss_fnc = log_like;
    N.randomize();
    //N.read_from_file();

    int n_tr = 9000;
    N.train(x, y, n_tr, /*batch_size=*/300, /*learning_rate=*/.01, /*num_of_epochs=*/6);
    N.print_to_file();

    double correct = 0, all = 0;
    int max = 0;
    for(int s=0;s<n_samp-n_tr;s++){
        all++;
        N.forward(x[s+n_tr]);
        for(int i=0;i<10;i++) max = N.arch[depth-1].p_d[i]>N.arch[depth-1].p_d[max] ? i:max;
        correct += y[s+n_tr][max];
        //for(int i=0;i<10;i++) cout << N.arch[depth-1].p_d[i] << ","; cout << endl;
        //for(int i=0;i<10;i++) cout << y[s+n_tr][i] << ","; cout << endl;
    }
    cout << "Accuracy: " << 100*correct/all << "%\n";

    return 0;
}