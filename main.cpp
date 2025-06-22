#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <time.h>

#include "header.h"
#include "neural.h"

void dense();
void mnist();
void cifar();

int main(){
    //dense();
    //mnist();
    cifar();

    return 0;
}

void dense(){
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
}

void mnist(){
    int n_samp = 60000, in_v = 28, in_h = 28;

    double *** x;
    x = new double**[n_samp];
    for(int s=0;s<n_samp;s++) x[s] = new double*[in_v];
    for(int s=0;s<n_samp;s++) for(int i=0;i<in_v;i++) x[s][i] = new double[in_h];
    double ** y;
    y = new double*[n_samp];
    for(int i=0;i<n_samp;i++) y[i] = new double[10];
    for(int i=0;i<n_samp;i++) for(int j=0;j<10;j++) y[i][j] = 0;

    ifstream file("mnist.csv"); // change path to your local file
    assert(file); // check that file loaded correctly
    string str;
    for(int line=0;line<n_samp;line++){
        getline(file, str, ',');
        y[line][stoi(str)] = 1;
        for(int i=0;i<in_v*in_h;i++){
            getline(file, str, ',');
            x[line][(i-(i%in_h))/in_v][i%in_h] = stod(str)/255.;
        }
    }
    file.close();

    int depth = 4; // number of layers
    layer L[depth];
    L[0].set("conv", 5); L[0].activ = relu; // first conv layer, with five filters
    L[1].set("conv", 8); L[1].activ = relu; // second conv layer, with eight filters
    L[2].set("dense", 10); L[2].activ = id; // one dense layer, with 10 neurons
    L[3].set("softmax");                    // a softmax layer at the end

    L[1].set_filter_size(5); // change filter size
    L[1].set_stride(3); // change stride

    network N({in_v,in_h}, L, depth);
    N.loss_fnc = log_like;
    N.randomize();

    int n_tr = 58000;
    N.train(x, y, n_tr, /*batch_size=*/50, /*learning_rate=*/.01, /*num_of_epochs=*/2);


    double correct = 0, all = 0;
    int max = 0;
    for(int s=0;s<n_tr;s++){
        all++;
        N.forward(x[s]);
        for(int i=0;i<10;i++) max = N.arch[depth-1].p_d[i]>N.arch[depth-1].p_d[max] ? i:max;
        correct += y[s][max];
    }
    cout << "Train accuracy: " << 100*correct/all << "%\n";

    correct = 0;
    all = 0;
    max = 0;
    for(int s=0;s<n_samp-n_tr;s++){
        all++;
        N.forward(x[s+n_tr]);
        for(int i=0;i<10;i++) max = N.arch[depth-1].p_d[i]>N.arch[depth-1].p_d[max] ? i:max;
        correct += y[s+n_tr][max];
    }
    cout << "Test accuracy: " << 100*correct/all << "%\n";

}

void cifar(){
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


    int depth = 5;
    layer L[depth];
    L[0].set("conv",5);
    L[1].set("maxpool"); L[1].set_filter_size(2);
    L[2].set("conv",8);
    L[3].set("dense",10); L[3].activ = id;
    L[4].set("softmax");

    network N({in_d,in_v,in_h},L,depth);
    N.loss_fnc = log_like;
    N.randomize();
    //N.read_from_file();

    int n_tr = 28800;
    double correct = 0, all = 0;
    int max = 0;

    N.train(x, y, n_tr, /*batch_size=*/64, /*learning_rate=*/.005, /*num_of_epochs=*/35);

    correct = 0, all = 0;
    max = 0;
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
    }
    cout << "Test accuracy: " << 100*correct/all << "%" << endl << endl;

}