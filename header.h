double rand_U(double width = 1.){ // uniform random number
    return width*(2*(((double)dist(rng))/32767.)-1);
}

struct activation_func{
    /*struct that encapsulates a function f:R->R and its derivative df:R->R. Optional, the name of the function.
    Default f(x) = softsign(x) = x/(1+|x|) a.k.a soft tanh.
    */
    string name = "";
    double (*f)(double);
    double (*df)(double);

    activation_func(){
        f = [](double x)->double {return x/(1+abs(x));};
        df = [](double x)->double {return 1/((1+abs(x))*(1+abs(x)));};
        name = "soft tanh";
    }

    activation_func(double(*activ)(double), double(*D_activ)(double), string s = ""){
        f = activ;
        df = D_activ;
        name = s;
    }
};

struct loss{
    /*struct that encapsulates a function f:RxR->R and its derivative df = d_1f:RxR->R. Optional, the name of the function.
    Default: L(x,y) = least squares(x,y) = (1/2)(x-y)^2.
    */
    string name = "";
    double (*f)(double,double);
    double (*df)(double,double);

    loss(){
        f = [](double x, double y)->double {return 0.5*(x-y)*(x-y);};
        df = [](double x, double y)->double {return x-y;};
        name = "least squares";
    }

    loss(double(*loss_fnc)(double,double), double(*D_loss_fnc)(double,double), string s = ""){
        f = loss_fnc;
        df = D_loss_fnc;
        name = s;
    }
};

//examples of activation functions (one unnamed). Note:  "->double" can be removed
const activation_func softsign([](double x)->double {return x/(1.+abs(x));}, [](double x)->double {return 1./((1.+abs(x))*(1.+abs(x)));}, "soft sign");
const activation_func relu([](double x)->double {return x>0. ? x : 0.;}, [](double x)->double {return x>0. ? 1. : 0.;}, "ReLU");
const activation_func id([](double x)->double {return x;}, [](double x)->double {return 1.;}, "identity");
const activation_func sigmoid([](double x)->double {return 1./(1.+exp(-x));}, [](double x)->double {return 1./((1.+exp(-x))*(1.+exp(x)));}, "logistic");
const activation_func quad([](double x)->double {return x*sqrt(x*x+1.);}, [](double x)->double {return (2.*x*x+1.)/sqrt(1.+x*x);});

//activation_func LReLU(double a){ // untested
//    activation_func act([](double x)->double {return x>0. ? x : a*x;}, [](double x)->double {return x>0. ? 1. : a;}, "LReLU");
//    return act;
//}

//examples of loss functions
const loss least_sq([](double x,double y)->double {return 0.5*(x-y)*(x-y);}, [](double x, double y)->double {return x-y;}, "least squares");
const loss log_like([](double x,double y)->double {return -y*log(x+.000000001)-(1-y)*log(1-x+.000000001);}, [](double x, double y)->double {return -y/(x+.000000001)+(1-y)/(1-x+.000000001);}, "cross-entropy");
//TODO: define a version of `log_like` that takes logits directly, so one can drop the softmax layer


void to_binary(int i, double * y, int L){ // L = len(y); y = binary representation of i
    y[L-1] = i%2;
    for(int k=L-2;k>=0;k--){
        y[k] = i;
        for(int n=L-1;n>k;n--) y[k] -= (y[n]*(2<<(L-n-1)))/2;
        y[k] = (((int)y[k])/(2<<(L-k-2)))%2;
    }
}

void progress_bar_before(int i, int epochs, int max_bar = 20){ // displays a progress bar while training
    if(i < (epochs/max_bar)*max_bar){
        cout << "[";
        for(int j=0;j<i % max_bar;j++) cout << "#";
        for(int j=i % max_bar;j<min(epochs,max_bar)-1;j++) cout << " ";
        cout << "] + " << (epochs/max_bar) - (i/max_bar);
        for(int j=0;j<int(log10(epochs/max_bar))+1;j++) cout << " ";
    }
    else{
        cout << "[";
        for(int j=0;j<i % max_bar;j++) cout << "#";
        for(int j=i % max_bar;j< epochs% max_bar-1;j++) cout << " ";
        cout << "]    ";
        for(int j=epochs% max_bar;j<min(epochs,max_bar)-1;j++) cout << " ";
        for(int j=0;j<int(log10(epochs/max_bar))+1;j++) cout << " ";
    }
}
void progress_bar_after(int i, int epochs, int max_bar = 20){
    cout << "\b\b";
    for(int j=0;j<min(epochs,max_bar)-1;j++) cout << "\b";
    cout << "\b\b\b\b";
    for(int j=0;j<int(log10((epochs/max_bar) - (i/max_bar)))+3; j++) cout << "\b";
    for(int j=0;j<int(log10(epochs/max_bar))+1;j++) cout << "\b";
    if(i == epochs-2){
        cout << "[";
        for(int j=0;j<min(epochs,max_bar)-1;j++) cout << "#";
        cout << "]";
    }
}

// double shuffle(double * x, int size, int i){
//     int pos = i;
//     for (int j=size-1;j>pos;j--){
//         int k = rand()%(j+1);
//         if(k == pos) pos = j;
//         else if(j == pos) pos = k;
//     }
//     return x[pos];
// }

void shuffle_samples(double ** x, double ** y, int n_samp){
    //TODO: change rand() into dist(rng)
    double * z;
    int j;
    for(int i=0;i<n_samp-1;i++){
        j = i + (rand()%(n_samp-i));
        z = x[i];
        x[i] = x[j];
        x[j] = z;
        z = y[i];
        y[i] = y[j];
        y[j] = z;
    }
}

void full_print(double * M, int n, ostream & os = cout){ // prints one-dimensional array
    os << "{";
    for(int i=0;i<n;i++){
        os << fixed << setprecision(5) << (M[i]<0.?"":" ") << M[i] << (i<n-1?",":"}");
    }
}
void full_print(double * M, int n_v, int n_h, ostream & os = cout){ // prints two-dimensional array
    os << "{";
    for(int i=0;i<n_v;i++){
        full_print(M+i,n_h,os);
        os << (i<n_v-1?",\n ":"}");
    }
}
void full_print(double * M, int n_d, int n_v, int n_h, ostream & os = cout){ // prints three-dimensional array
    os << "{";
    for(int i=0;i<n_d;i++){
        full_print(M+i,n_v,n_h,os);
        os << (i<n_d-1?",\n ":"}");
    }
}

void short_print(double * M, int n, ostream & os = cout){ // prints a one-dimensional array, omitting entries in the middle
    int max_n = 3;
    if(n <= max_n){
        full_print(M,n,os);
    }
    else{
        os << "{";
        for(int i=0;i<max_n-1;i++){
            os << fixed << setprecision(5) << (M[i]<0.?"":" ") << M[i] << ",";
        }
        os << " ... ," << (M[n-1]<0.?"":" ") << M[n-1] << "}";
    }
}
void short_print(double * M, int n_v, int n_h, ostream & os = cout){ // prints a two-dimensional array, omitting entries in the middle
    int max_n = 3;
    os << "{";
    if(n_v <= max_n){
        for(int j=0;j<n_v;j++){
            short_print(M+j,n_h,os);
            os << (j<n_v-1?",\n ":"}");
        }
    }
    else{
        for(int j=0;j<max_n-1;j++){
            short_print(M+j,n_h,os);
            os << "," << endl << " ";
        }
        if(n_h > max_n){
            for(int j=0;j<max_n-1;j++) os << "      |  ";
            os << "            |  \n ";
        }
        else{
            for(int j=0;j<n_h;j++) os << "      |  ";
            os << "\n ";
        }
        short_print(M+n_v-1,n_h,os);
        os << "}";
    }
}
void short_print(double * M, int n_d, int n_v, int n_h, ostream & os = cout){ // prints a three-dimensional array, omitting entries in the middle
    int max_n = 3;
    os << "{";
    if(n_d <= max_n){
        for(int j=0;j<n_d;j++){
            short_print(M+j,n_v,n_h,os);
            os << (j<n_d-1?",\n ":"}");
        }
    }
    else{
        for(int j=0;j<max_n-1;j++){
            short_print(M+j,n_v,n_h,os);
            os << "," << endl << " ";
        }
        if(n_h > max_n){
            for(int j=0;j<max_n-1;j++) os << "     ||| ";
            os << "           ||| \n ";
        }
        else{
            for(int j=0;j<n_h;j++) os << "     ||| ";
            os << "\n ";
        }
        short_print(M+n_d-1,n_v,n_h,os);
        os << "}";
    }
}