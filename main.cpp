#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <time.h>

#include "header.h"
#include "neural.h"


int main(){
    int n_samp = 30000, in_d = 3, in_v = 32, in_h = 32;

    double **** x;
    x = new double***[n_samp];
    for(int s=0;s<n_samp;s++) x[s] = new double**[in_d];
    for(int s=0;s<n_samp;s++) for(int i=0;i<in_d;i++) x[s][i] = new double*[in_v];
    for(int s=0;s<n_samp;s++) for(int i=0;i<in_d;i++) for(int j=0;j<in_v;j++) x[s][i][j] = new double[in_h];
    double ** y;
    y = new double*[n_samp];
    for(int i=0;i<n_samp;i++) y[i] = new double[10];

    ifstream file1("cifar10.txt");
    assert(file1);
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
    ifstream file2("cifar10_y.txt");
    assert(file2);
    for(int line=0;line<n_samp;line++){
        getline(file2, str, ',');
        for(int i=0;i<10;i++) y[line][i] = 0;
        y[line][stoi(str)] = 1;
    }
    file2.close();


    int depth = 7;
    layer L[depth];
    L[0].set("conv",8);
    L[1].set("maxpool"); L[1].set_filter_size(2);
    L[2].set("conv",12); L[2].set_filter_size(4);
    L[3].set("maxpool"); L[3].set_filter_size(2);
    L[4].set("conv",16); L[4].set_filter_size(5);
    //L[5].set("dense",36); L[5].activ = relu;
    L[5].set("dense",10); L[5].activ = id;
    L[6].set("softmax");

    network N({in_d,in_v,in_h},L,depth);
    N.loss_fnc = log_like;
    //N.randomize();
    N.read_from_file();

    int n_tr = 29000;


    clock_t start, end;
    start = clock();
    N.train(x, y, n_tr, /*batch_size=*/500, /*learning_rate=*/.01, /*num_of_epochs=*/5);
    end = clock();
    cout << "Train time: " << double(end-start)/CLOCKS_PER_SEC << endl;
    N.print_to_file();


    double correct = 0, all = 0;
    int max = 0;
    for(int s=0;s<n_tr;s++){
        all++;
        N.forward(x[s]);
        for(int i=0;i<10;i++) max = N.arch[depth-1].p_d[i]>N.arch[depth-1].p_d[max] ? i:max;
        correct += y[s][max];
    }
    cout << "Train accuracy: " << 100*correct/all << "%" << endl;

    correct = 0, all = 0;
    max = 0;
    for(int s=0;s<n_samp-n_tr;s++){
        all++;
        N.forward(x[s+n_tr]);
        for(int i=0;i<10;i++) max = N.arch[depth-1].p_d[i]>N.arch[depth-1].p_d[max] ? i:max;
        correct += y[s+n_tr][max];
        //for(int i=0;i<10;i++) cout << N.arch[depth-1].p_d[i] << ","; cout << endl;
        //for(int i=0;i<10;i++) cout << y[s+n_tr][i] << ","; cout << endl;
    }
    cout << "Test accuracy: " << 100*correct/all << "%" << endl;
}

int main_old_2(){
    int depth = 4, inp_v = 12, inp_h = 12;
    layer L[depth];
    L[0].set("conv",2);
    L[1].set("maxpool"); L[1].set_filter_size(4,4); L[1].set_pool_overlap(0,1);
    L[2].set("conv",1); L[2].set_filter_size(2,2);
    L[3].set("dense",1);
    network N({1,inp_v,inp_h},L,depth);
    N.randomize();
    //cout << L[1] << endl;
    //cout << N << endl;


    double *** x;
    x = new double**[1];
    x[0] = new double*[inp_v];
    for(int i=0;i<inp_v;i++) x[0][i] = new double[inp_h];
    for(int i=0;i<inp_v;i++) for(int j=0;j<inp_h;j++) x[0][i][j] = rand_U(1);
    //for(int i=0;i<inp_v;i++) {cout << "{"; for(int j=0;j<inp_h;j++) cout << x[0][i][j] << ","; cout << "}" << endl;}

    N.forward(x);

    // for(int s=0;s<depth-1;s++){
    //     for(int i=0;i<L[s].get_numb_of_out_channels();i++){
    //         for(int j=0;j<L[s].get_out_im_size('v');j++){
    //             for(int k=0;k<L[s].get_out_im_size('h');k++){
    //                 cout << L[s].p_c[i][j][k] << " ";
    //             }
    //             cout << endl;
    //         }
    //         cout << endl;
    //     }
    // cout << "--------------------------------------------------------------------\n";
    // }
    // cout << L[depth-1].p_d[0] << ",";


    double * y; y = new double[1]; y[0] = .23;
    cout << N.loss_fnc.f(L[depth-1].p_d[0],y[0]) << endl;
    N.set_j_to_zero();
    N.update_derivatives(x,y,1);

    cout << N.jb_d[0][0] << endl;
    //for(int i=0;i<2;i++) for(int j=0;j<3;j++) for(int k=0;k<3;k++) cout << N.jf_c[0][i][0][j][k] << " ";
    for(int i=0;i<1;i++) for(int u=0;u<2;u++) for(int j=0;j<2;j++) for(int k=0;k<2;k++) cout << N.jf_c[2][i][u][j][k] << " "; cout << endl;
    for(int i=0;i<2;i++) for(int u=0;u<1;u++) for(int j=0;j<3;j++) for(int k=0;k<3;k++) cout << N.jf_c[0][i][u][j][k] << " ";



}


int main_old(){
    int depth = 6; // number of layers
    layer L[depth]; // array of neurons
    L[0].set("dense", 32); // first layer, with 32 neurons
    L[1].set("dense", 16); // second layer, with 16 neurons
    L[2].set("dense", 8);  // third layer, with 8 neurons
    L[3].set("dense", 4);  // fourth layer, with 4 neurons
    L[4].set("dense", 2);  // fifth layer, with 2 neurons
    L[5].set("softmax");

    L[1].activ = relu;
    L[4].activ = id;

    int n_in = 5; // input size
    network N({n_in}, L, depth);
    N.randomize(/*scale = */ 0.7);
    //cout << N << endl;

    int n_samp = 7776; // number of samples
    double ** X, ** Y;
    
    X = new double*[n_samp]; // declare predictor
    Y = new double*[n_samp]; // declare response
    for(int i=0;i<n_samp;i++) X[i] = new double[n_in];
    for(int i=0;i<n_samp;i++) Y[i] = new double[2];
    
    // uniform grid in [0,1]^5
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][0] = i1/5.;
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][1] = i2/5.;
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][2] = i3/5.;
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][3] = i4/5.;
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][4] = i5/5.;

    // model to be learned
    for(int s=0;s<n_samp;s++) Y[s][0] = sin(X[s][0]+X[s][1]+X[s][2]+X[s][3]+X[s][4]);
    for(int s=0;s<n_samp;s++) Y[s][1] = 0.2*(cos(X[s][0]-X[s][1]+X[s][2]-X[s][3]+X[s][4])+X[s][2]-X[s][4]);

    N.train(X, Y, n_samp, /*batch_size=*/n_samp, /*learning_rate=*/.02, /*num_of_epochs=*/20);

    // check that it worked
    for(int s=1000;s<1000+6;s++) cout << "(" << Y[s][0] << "," << Y[s][1] << ") "; cout << endl;
    cout << "vs" << endl;

    for(int s=1000;s<1000+6;s++){
        N.forward(X[s]);
        cout << "(" << N[depth-1][0] << "," << N[depth-1][1] << ") ";
    }
    cout << endl;

    return 0;
}
