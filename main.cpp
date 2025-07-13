#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <random>
#include <iomanip>

using namespace std;

mt19937 rng(1);
uniform_int_distribution<int> dist(0, 32767);

#include "neural.h"

void fully_connected();
void mnist();
void cifar();

int main(){

    fully_connected();
    //mnist();
    //cifar();

    return 0;
}


void fully_connected(){

    input i(5);
    dense d1(32), d2(16), d3(8), d4(4), d5(2);
    network N{&i,&d1,&d2,&d3,&d4,&d5};
    d2.activ = relu;
    d5.activ = id;

    N.randomize(/*scale = */ 0.7);
    //N.print();

    double * x; // input to pass through the network
    x = new double[5];
    for(int i=0;i<5;i++) x[i] = (i+1)/10.;

    N.forward(x); // perform forward pass.

    for(int l=0;l<6;l++){ // print activations (hidden and output layers)
        cout << "activations of layer " << l << ": ";
        for(int i=0;i<N.output_size(l);i++) cout << N.activation(l,i) << " ";
        cout << endl;
    }

    //cout << "activations of last layer: " << N.activation(0) << " " << N.activation(1) << endl; // print output layer directly

    int n_samp = 7776; // number of samples
    double ** X, ** Y; // predictor and response

    X = new double*[n_samp];
    Y = new double*[n_samp];
    for(int i=0;i<n_samp;i++) X[i] = new double[5];
    for(int i=0;i<n_samp;i++) Y[i] = new double[2];

    // uniform grid in [0,1]^5
    for (int i=0;i<n_samp;i++){
        int k = i;
        for(int j=0;j<5;j++){
            X[i][j] = (k%6)/5.;
            k /= 6;
        }
    }

    // model to be learned
    for(int s=0;s<n_samp;s++) Y[s][0] = sin(X[s][0]+X[s][1]+X[s][2]+X[s][3]+X[s][4]);
    for(int s=0;s<n_samp;s++) Y[s][1] = 0.2*(cos(X[s][0]-X[s][1]+X[s][2]-X[s][3]+X[s][4])+X[s][2]-X[s][4]);

    shuffle_samples(X,Y,n_samp);

    int n_tr = 7000; // training set

    cout << "mean L2 error before training: " << N.cost(X,Y,n_samp) << endl;

    SGD optim(&N); // declare stochastic gradient descent optimizer (with momentum)
    optim.train(X, Y, n_tr, /*batch_size=*/1000, /*learning_rate=*/1, /*num_of_epochs=*/100, /*progress_bar=*/ false);

    cout << "mean L2 error after training, training set: " << N.cost(X,Y,n_tr) << endl;
    cout << "mean L2 error after training, testing set: " << N.cost(X+n_tr,Y+n_tr,n_samp-n_tr) << endl;

    // check that it worked
    for(int s=7000;s<7000+6;s++) cout << "(" << Y[s][0] << "," << Y[s][1] << ") "; cout << endl;
    cout << "vs" << endl;

    for(int s=7000;s<7000+6;s++){
        N.forward(X[s]);
        cout << "(" << N.activation(0) << "," << N.activation(1) << ") ";
    }
    cout << endl;

    delete[] x;
    delete[] X;
    delete[] Y;

}

void mnist(){

    input i(28,28); // declare input size
    conv c1(5), c2(8,5,5,3,3); // two conv layers, the first one with default size and stride, the second one 5x5 with a stride of `(s_v,s_h) = (3,3)`
    dense d(10); // dense layer with ten outputs, for the 10 classes
    softmax sm;
    network N{&i,&c1,&c2,&d,&sm};

    // change activation functions
    c1.activ = relu;
    c2.activ = relu;
    d.activ = id;

    N.randomize();
    //N.print();


    int n_samp = 60000; // number of samples in the mnist dataset
    double ** x, ** y;
    x = new double*[n_samp];
    y = new double*[n_samp];

    ifstream file("mnist.csv"); // change path to your local file
    assert(file); // check that file loaded correctly
    string str;
    for(int line=0;line<n_samp;line++){ // load data (details depend on how your mnist file is formatted)
        x[line] = new double[28*28];
        y[line] = new double[10];
        getline(file, str, ',');
        for(int i=0;i<10;i++) y[line][i] = 0;
        y[line][stoi(str)] = 1;
        for(int i=0;i<28*28;i++){
            getline(file, str, ',');
            x[line][i] = stod(str)/255.;
        }
    }
    file.close();

    int n_tr = 58000; // training set

    Adam optim(&N); // we use an Adam optimizer
    optim.loss_fnc = log_like;
    optim.train(x, y, n_tr, /*batch_size=*/40, /*learning_rate=*/.02, /*num_of_epochs=*/1);

    // check accuracy:
    double correct = 0, all = 0;
    int max = 0;
    for(int s=0;s<n_tr;s++){
        all++;
        N.forward(x[s]);
        for(int i=0;i<10;i++) max = N.activation(i)>N.activation(max) ? i:max;
        correct += y[s][max];
    }
    cout << "Train accuracy: " << 100*correct/all << "%\n";

    correct = 0, all = 0;
    max = 0;
    for(int s=0;s<n_samp-n_tr;s++){
        all++;
        N.forward(x[s+n_tr]);
        for(int i=0;i<10;i++) max = N.activation(i)>N.activation(max) ? i:max;
        correct += y[s+n_tr][max];
    }
    cout << "Test accuracy: " << 100*correct/all << "%\n";


    delete[] x;
    delete[] y;

}

void cifar(){

    input i(3,32,32);
    conv c1(5), c2(8,5,5,2,2);
    maxpool mp;
    dense d(10);
    softmax sm;
    network N{&i,&c1,&mp,&c2,&d,&sm};
    d.activ = id;

    N.randomize();

    int n_samp = 30000;
    double ** x;
    x = new double*[n_samp];
    for(int s=0;s<n_samp;s++) x[s] = new double[3*32*32];
    double ** y;
    y = new double*[n_samp];
    for(int i=0;i<n_samp;i++) y[i] = new double[10];

    ifstream file1("cifar10.txt");
    assert(file1);
    string str;
    for(int line=0;line<n_samp;line++){
        for(int j=0;j<3*32*32;j++){
            getline(file1, str, ',');
            x[line][j] = stod(str)/255.;
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


    int n_tr = 28800; // training set

    Adam optim(&N); // we use an Adam optimizer
    optim.loss_fnc = log_like;
    optim.train(x, y, n_tr, /*batch_size=*/64, /*learning_rate=*/.005, /*num_of_epochs=*/35);

    // check accuracy:
    double correct = 0, all = 0;
    int max = 0;
    for(int s=0;s<n_tr;s++){
        all++;
        N.forward(x[s]);
        for(int i=0;i<10;i++) max = N.activation(i)>N.activation(max) ? i:max;
        correct += y[s][max];
    }
    cout << "Train accuracy: " << 100*correct/all << "%\n";

    correct = 0, all = 0;
    max = 0;
    for(int s=0;s<n_samp-n_tr;s++){
        all++;
        N.forward(x[s+n_tr]);
        for(int i=0;i<10;i++) max = N.activation(i)>N.activation(max) ? i:max;
        correct += y[s+n_tr][max];
    }
    cout << "Test accuracy: " << 100*correct/all << "%\n";



}