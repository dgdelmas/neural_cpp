#include <iostream>
#include <fstream>
#include <random>
#include <cassert>
#include <unistd.h>

#include "header.h"
#include "neural.h"


int main_2(){

    int i_d = 1, i_v = 28, i_h = 28;

    int n_samp = 60000;
    double **** x, ** y;

    x = new double***[n_samp];
    y = new double*[n_samp];
    x = new double***[n_samp];
    for(int s=0;s<n_samp;s++) x[s] = new double**[i_d];
    for(int s=0;s<n_samp;s++) for(int i=0;i<i_d;i++) x[s][i] = new double*[i_v];
    for(int s=0;s<n_samp;s++) for(int i=0;i<i_d;i++) for(int j=0;j<i_v;j++) x[s][i][j] = new double[i_h];
    for(int i=0;i<n_samp;i++) y[i] = new double[10];

    ifstream file("C:\\Users\\Diego\\Documents\\cpp\\sudoku\\digits\\mnist_train.csv");
    assert(file);
    string str;
    for(int line=0;line<n_samp;line++){
        getline(file, str, ',');
        for(int i=0;i<10;i++) y[line][i] = 0;
        y[line][stoi(str)] = 1;
        for(int i=0;i<i_v*i_h;i++){
            getline(file, str, ',');
            x[line][0][(i-(i%i_h))/i_v][i%i_h] = stod(str)/255.;
        }
    }
    file.close();


    int depth = 4;
    layer L[depth];
    L[0].set("conv", 5); L[0].activ = relu;
    L[1].set("conv", 8); L[1].activ = relu;
    L[2].set("dense", 10); L[2].activ = id;
    L[3].set("softmax");

    L[1].f_v = L[1].f_h = 5;
    L[1].stride_h = L[1].stride_v = 3;

    network N({i_d,i_v,i_h}, L, depth);

    N.randomize();

    N.loss_fnc = log_like;

    int n_tr = 58000;
    N.train(x, y, n_tr, /*batch_size=*/50, /*learning_rate=*/.005, /*num_of_epochs=*/3);


    double correct = 0, all = 0;
    int max = 0;
    for(int s=0;s<n_samp-n_tr;s++){
        all++;
        N.forward(x[s+n_tr]);
        for(int i=0;i<10;i++) max = N.arch[depth-1].p_d[i]>N.arch[depth-1].p_d[max] ? i:max;
        correct += y[s+n_tr][max];
    }
    cout << "Accuracy: " << 100*correct/all << "%\n";

}


int main(){
    int depth = 5; // number of layers
    layer L[depth]; // array of neurons
    L[0].set("dense", 32); // first layer, with 32 neurons
    L[1].set("dense", 16); // second layer, with 16 neurons
    L[2].set("dense", 8); // third layer, with 8 neurons
    L[3].set("dense", 4);  // fourth layer, with 4 neurons
    L[4].set("dense", 2);  // fifth layer, with 2 neurons

    L[1].activ = relu;
    L[4].activ = id;

    int n_in = 5; // input size
    network N({n_in}, L, depth);
    N.randomize(/*scale = */ 0.7);

    int n_samp = 7776; // number of samples
    double ** X, ** Y;
    
    X = new double*[n_samp];
    Y = new double*[n_samp];
    for(int i=0;i<n_samp;i++) X[i] = new double[n_in];
    for(int i=0;i<n_samp;i++) Y[i] = new double[2];
    
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][0] = i1/5.;
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][1] = i2/5.;
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][2] = i3/5.;
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][3] = i4/5.;
    for(int i1=0;i1<6;i1++) for(int i2=0;i2<6;i2++) for(int i3=0;i3<6;i3++) for(int i4=0;i4<6;i4++) for(int i5=0;i5<6;i5++) X[i1+6*i2+36*i3+216*i4+1296*i5][4] = i5/5.;

    for(int s=0;s<n_samp;s++) Y[s][0] = sin(X[s][0]+X[s][1]+X[s][2]+X[s][3]+X[s][4]);
    for(int s=0;s<n_samp;s++) Y[s][1] = 0.2*(cos(X[s][0]-X[s][1]+X[s][2]-X[s][3]+X[s][4])+X[s][2]-X[s][4]);



    N.train(X, Y, n_samp, /*batch_size=*/n_samp, /*learning_rate=*/.02, /*num_of_epochs=*/500);



    for(int s=1000;s<1000+6;s++) cout << Y[s][0] << " ";
    cout << endl << "vs" << endl;

    for(int s=1000;s<1000+6;s++){
        N.forward(X[s]);
        cout << N.arch[depth-1].p_d[0] << " ";
    }
    cout << endl;

    //for(double s = 0;s<=1;s+=.2){
    //    cout << "{" << s << "," << sin(s) << "},";
    //}
    //cout << endl;
    //double * x;
    //x = new double[5];
    //x[0] = x[1] = x[2] = x[3] = 0;
    //for(double s = 0;s<=1;s+=.2){
    //    x[4] = s;
    //    N.forward(x);
    //    cout << "{" << s << "," << N.arch[depth-1].p_d[0] << "},";
    //}
    //cout << endl;

    //N.train(X, Y, n_samp, /*batch_size=*/n_samp, /*learning_rate=*/.01, /*num_of_epochs=*/10000,0);
    //for(double s = 0;s<=1;s+=.2){
    //    x[4] = s;
    //    N.forward(x);
    //    cout << "{" << s << "," << N.arch[depth-1].p_d[0] << "},";
    //}

    return 0;
}

//0 .2 .4 .6 .8 1 === 6

int main_old(){

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
    L[0].set("conv",8); L[0].f_v = 3; L[0].f_h = 3;
    L[1].set("conv",16); L[1].f_v = 5; L[1].f_h = 5; //L[1].stride_h = 2; L[1].stride_v = 2;
    L[2].set("conv",24); L[2].f_v = 7; L[2].f_h = 7;
    //L[1].set("conv",6); L[1].f_v = 5; L[1].f_h = 5;
    //L[2].set("dense",10);
    //L[3].set("dense",10); L[3].activ = id;
    //L[4].set("softmax");
    //L[0].set("dense",10);
    //L[0].set("dense",60);
    //L[3].set("dense",30);
    L[3].set("dense",10); L[3].activ = id;
    L[4].set("softmax");

    network N({in_d,in_v,in_h},L,depth);
    N.loss_fnc = log_like;
    //N.randomize();
    N.read_from_file();

    int n_tr = 28000;
    N.train(x, y, n_tr, /*batch_size=*/400, /*learning_rate=*/.0001, /*num_of_epochs=*/3);
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