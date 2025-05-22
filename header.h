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