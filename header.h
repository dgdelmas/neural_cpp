using namespace std;

void here(int i){cout << "here " << i << endl; cout.flush();} // for debugging

double rand_U(double width){ // uniform random number
    return width*(2*(((double)rand())/RAND_MAX)-1);
}

struct activation{
    /*struct that encapsulates a function f:R->R and its derivative df:R->R. Optional, the name of the function.
    Default f(x) = softsign(x) = x/(1+|x|) a.k.a soft tanh.
    */
    string name = "";
    double (*f)(double);
    double (*df)(double);

    activation(){
        f = [](double x)->double {return x/(1+abs(x));};
        df = [](double x)->double {return 1/((1+abs(x))*(1+abs(x)));};
        name = "soft tanh";
    }

    activation(double(*activ)(double), double(*D_activ)(double), string s = ""){
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
const activation softsign([](double x)->double {return x/(1.+abs(x));}, [](double x)->double {return 1./((1.+abs(x))*(1.+abs(x)));}, "soft tanh");
const activation relu([](double x)->double {return x>0. ? x : 0.;}, [](double x)->double {return x>0. ? 1. : 0.;}, "ReLU");
const activation id([](double x)->double {return x;}, [](double x)->double {return 1.;}, "identity");
const activation sigmoid([](double x)->double {return 1./(1.+exp(-x));}, [](double x)->double {return 1./((1.+exp(-x))*(1.+exp(x)));}, "logistic");
const activation quad([](double x)->double {return x*sqrt(x*x+1.);}, [](double x)->double {return (2.*x*x+1.)/sqrt(1.+x*x);});


//examples of loss functions
const loss least_sq([](double x,double y)->double {return 0.5*(x-y)*(x-y);}, [](double x, double y)->double {return x-y;}, "least squares");
const loss log_like([](double x,double y)->double {return -y*log(x+.000000001)-(1-y)*log(1-x+.000000001);}, [](double x, double y)->double {return -y/(x+.000000001)+(1-y)/(1-x+.000000001);}, "cross-entropy");

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

bool softpad(int L, int l, int o, int & m, int & n){ // finds minimal m\in[0,l) such that L == n*l+m(2*l-m-1)-(n+2m)*o
    for(m=0;m<l;m++){
        n = L+m-2*l*m+m*m-o+2*m*o;
        if( n % (l-o) == 0){
            n /= l-o;
            return true;
        }
    }
    return false;
}

void softpad_range(int & start, int & end, int a, int l, int o, int m, int n){
    if(a < m){
        start = a*(l-m-o)+(a*(a-1))/2;
        end = a*(l-m-o)+l-m+(a*(a+1))/2;
    }
    else if(a >= m && a < m + n ){
        start = a*(l-o)-(m*(m+1))/2;
        end = l+a*(l-o)-(m*(m+1))/2;
    }
    else{
        start = a*(l+m+n-o)-m*(m+n)-(a*(a+1)+n*(n-1))/2;
        end = a*(l+m+n-o)+l-1+o-m*(m+n-1)-(a*(a+3)+n*(n-3))/2;
    }
}