#include "header.h"

/*
Base layer class, with default methods. You can create your own layer type by defining a subclass and overriding the various methods.
The `network` class (see below) can take any layer type, as long as the various methods have been defined properly.

All layers carry an activation function (although it is sometimes not used). The default activation is "soft sign", i.e., x/(1+|x|).
Other options are `ReLU` (sigma(x) = max(0,x)), `id` (short for "identity", namely sigma(x) = x), `sigmoid` (sigma(x) = 1/(1+exp(x)).
User-defined activations are allowed too, use the `activation_func` struct (see header file) to define other activations.

The methods one can (and should) override are:
++ print(bool,ostream). Prints layer details to stream (default = cout). The bool controls short vs long print, i.e., how much detail
    we want to see.
++ init(), only for the very first layer, i.e., an input layer. This initializes the layer, without using information from previous
    layers (because there are none).
++ init(int*,int), for the rest of layers. This initializes the layer, using information from the previous layer. The first argument
    is a pointer to the `output_shape` of the previous layer, and the second argument is an integer equal to the `output_rank` of the
    previous layer. For now, `output_rank` can only be 1,2,3, but it should be easy to rewrite the code if one needs higher rank tensors.
    `output_shape` is an array of `output_rank` integers, specifying the dimensions of the tensor. For example, a 2x3x4 tensor has
    `output_rank = 3`, `output_shape[0] = 2, output_shape[1] = 3, output_shape[2] = 4`.
++ randomize(double). Randomizes all learnable parameters in the layer (some layer types have none), using the argument as, say,
    the scale (e.g., the width of a uniform distribution, or of a gaussian, etc.)
++ forward(double*). Performs a forward pass, taking as input the output of the previous layer.
++ backward(double*). Performs a backward pass, taking as input the gradients of the following layer.
++ zero_grad(). Sets all derivatives of the loss function with respect to all learnable parameters, to zero.
++ update_derivatives(double*,double*). Adds to the current derivative of the loss function with respect to the learnable parameters
    an update coming from a single sample. (So derivatives accumulate over a mini-batch, and we reset it to zero using `zero_grad` at the
    end of the mini-batch). The first input to `update_derivatives` is a pointer to the output of the previous layer, and the second is a
    poiter to the gradients of the following layer.

Default attributes:
++ input_rank and output_rank. The rank of the tensor that the layer takes as input and output, respectively. For now, only ranks 1,2,3
    are implemented but it should be easy to generalize, should the need arise.
++ input_size and output_size. The total dimension of the input and output (e.g., `input_size = input_shape[0]*input_shape[1]*input_shape[2]`)
++ input_shape and output_shape. A list of integers that specify the dimensions of input and output tensors. For example, if the input
    is a 2x3x4 tensor, then `input_shape[0] = 2, input_shape[1] = 3, input_shape[2] = 4`, and `input_size = 24`
++ p. This is the output of the layer, a list of doubles of length `output_size`. Note that `p` is always a one-dimensional array, even
    when the output is a higher-rank tensor (i.e., we always flatten all arrays and store them linearly).
++ q. This is not always used. When `p = f(...)`, then `q` is defined as `q = f'(...)`, i.e., `q` is the derivative of the output with
    respect to (typically) logits. Although not strictly necessary, storing this variable during the forward pass makes the backward
    pass a little more efficient.
++ theta. These are all the learnable parameters (some layer types have none). For example, a dense layer has weights and biases. We
    flatten these and store them in the linear array `theta`.
++ Z. These are the gradients, namely `Z[i] = d(loss)/dp[i]`.
++ dL. These are the derivatives of the loss function with respect to the learnable parametres, `dL[i] = d(loss)/dtheta[i]`.
++ n_param. This is the number of learnable parameters, i.e., the length of `theta` (of course also the length of `dL`).
++ initQ. This is a bool that remembers whether the layer has been initialized (via the `init` method) or not. A layer cannot be used
    before it has been initialized. (The `network` class runs `init` automatically so we never initialize layers by hand; this is just
    a safety measure, it should never matter in practice unless someone does something we did not account for).
*/
class layer{
    friend class network;   // gathers a list of `layer`s together and defines various methods thereon
    friend class optimizer; // trains a network using some form of gradient descent
    friend class SGD;       // a specific optimizer that uses stochastic gradient descent
    friend class Adam;      // another specific optimizer, this one using the Adam method

public:
    activation_func activ = softsign; // activation function (default = softsign(x) = x/(1+|x|))
    virtual int print(bool, ostream & os = cout){
        os << "default print(bool,ostream)." << endl;
        return 0;
    }
    virtual ~layer(){}

protected:
    virtual void init(){
        cout << "default init()." << endl;
    }
    virtual void init(int*,int){
        cout << "default init(int*,int)." << endl;
    }
    virtual void randomize(double){
        cout << "default randomize(double)." << endl;
    }
    virtual void forward(double*){
        cout << "default forward(double*)." << endl;
    }
    virtual void backward(){} // should this be removed?
    virtual void backward(double*){}

    virtual void zero_grad(){
        cout << "default zero_grad()." << endl;
    }
    virtual void update_derivatives(double*,double*){
        cout << "defaul update_derivatives(double*,double*)." << endl;
    }

    int input_rank = -1, output_rank = -1, input_size = -1, output_size = -1;
    int * input_shape, * output_shape;
    double * p, * q, * theta; // output, output', learnable parameters
    double * Z; // d(loss)/dp
    double * dL; // d(loss)/d(theta)
    int n_param = 0; // length of theta == length of dL == number of learnable parameters
    bool initQ = false;
};


/*
Here we define our first layer type, an input layer. This  takes an array and passes it to the first actual layer in the network. The
main purpose of this layer is to specify the size of the input tensor, it doesn't do much more. The size of a rank-3 array is specified
in the following order: (depth, vertical size, horizontal size).

Declare as
    input i(3);
for an input layer that takes a one-dimensional array of length 3, or as
    input i(4,5);
for an input layer that takes a two-dimensional array of length 4x5, or as
    input i(6,7,8);
for three-dimensional arrays. We do not allow for higher-dimensional arrays but it is easy to rewrite the code if such arrays turn out
to be useful for some applications. Note: input(1,3,3) and input(3,3) are essentially identical: all rank-2 arrays are secretly treated
as rank-3 internally, just with length-1 depth.
*/
class input: public layer{
public:
    input(int n){
        assert(n >= 1);
        output_rank = 1;
        output_shape = new int[3];
        output_shape[0] = n;
        output_shape[1] = 1;
        output_shape[2] = 1;
        output_size = n;
    }
    input(int n_v, int n_h){
        assert(n_v >=1 && n_h >= 1);
        output_rank = 3;
        output_shape = new int[3];
        output_shape[0] = 1;
        output_shape[1] = n_v;
        output_shape[2] = n_h;
        output_size = n_v*n_h;
    }
    input(int n_d, int n_v, int n_h){
        assert(n_d >= 1 && n_v >=1 && n_h >= 1);
        output_rank = 3;
        output_shape = new int[3];
        output_shape[0] = n_d;
        output_shape[1] = n_v;
        output_shape[2] = n_h;
        output_size = n_d*n_v*n_h;
    }

    int print(bool longQ = false, ostream & os = cout) override {
        assert(initQ);
        os << "input layer, size: ";
        if(output_rank == 1) os << output_shape[0] << endl;
        else os << output_shape[0] << "x" << output_shape[1] << "x" << output_shape[2] << endl;
        return 0;
    }
    ~input() override{
        delete[] output_shape;
    }

protected:
    void init() override {
        assert(!initQ);
        initQ = true;
    }
    void forward(double * x) override {
        p = x;
    }
};



/*
Our next layer type is a dense layer, namely a fully connected, all-to-all layer.

Although in principle one could feed a dense layer into a convolutional one, there is no canonical way of doing so, since a dense layer
is always rank-1. We could simply declare by hand that this one-dimensional array should be reshaped into a higher-dimensional one, but
this is not a very natural operation. For now, I do not allow for this, but it should be easy to define a `reshape` layer that does this
for us should the need arise.

Declare as
    dense(4);
for a dense layer with 4 neurons.

We override all the default methods in the `layer` class so that, for example, the forward pass takes the usual form `p = sigma(w.x+b)`.
The weights and biases are stored as `w[i][j] = theta[input_size*i + j]` and `b[i] = theta[input_size*output_size + i]`.
*/
class dense: public layer{
public:
    dense(int n){
        assert(n >= 1);
        output_rank = 1;
        output_shape = new int[1];
        output_shape[0] = n;
        output_size = n;
    }
    int print(bool longQ = false, ostream & os = cout) override {
        assert(initQ);
        os << "dense layer, input size: " << input_size << ", output size: " << output_size << endl;
        if(longQ){
            if(activ.name.size() > 0) os << "activation function: " << activ.name << endl;
            os << "weights:\n";
            full_print(theta,output_size,input_size,os);
            os << "\nbiases:\n";
            full_print(theta+input_size*output_size,output_size,os);
            os << "\nnumber of parameters: " << n_param << endl;
        }
        else{
            os << "weights:\n";
            short_print(theta,output_size,input_size,os);
            os << "\nbiases:\n";
            short_print(theta+input_size*output_size,output_size,os);
            os << endl;
        }
        return n_param;
    }

    ~dense() override {
        delete[] output_shape;
        if(initQ){
            delete[] p;
            delete[] q;
            delete[] theta;
            delete[] Z;
            delete[] dL;
        }
    }
protected:
    void init(int * prev_shape, int prev_rank) override {
        assert(!initQ);
        input_size = 1;
        for(int i=0;i<prev_rank;i++) input_size *= prev_shape[i];
        p = new double[output_size];
        q = new double[output_size];
        n_param = input_size*output_size + output_size;
        theta = new double[n_param];
        dL = new double[n_param];
        Z = new double[input_size];
        initQ = true;
    }
    void randomize(double scale = 1.) override {
        assert(initQ);
        for(int i=0;i<output_size;i++){
            theta[input_size*output_size+i] = 0.1*rand_U(scale);
            for(int j=0;j<input_size;j++) theta[input_size*i+j] = rand_U(scale);
        }
    }
    void forward(double * x) override {
        for(int i=0;i<output_size;i++){
            double aux = 0.;
            for(int j=0;j<input_size;j++) aux += theta[input_size*i + j]*x[j];
            aux += theta[input_size*output_size + i];
            p[i] = activ.f(aux);
            q[i] = activ.df(aux);
        }
    }

    void backward(double * x) override {
        for(int i=0;i<input_size;i++){
            double aux = 0.;
            for(int j=0;j<output_size;j++) aux += x[j]*q[j]*theta[input_size*j + i];
            Z[i] = aux;
        }
    }

    void zero_grad() override {
        for(int i=0;i<n_param;i++) dL[i] = 0.;
    }

    void update_derivatives(double * x, double * z) override {
        for(int a=0;a<input_size*output_size;a++){
            double aux = 0.;
            int min_i = max(0,a/input_size);
            int max_i = min(1+a/input_size,input_size);
            for(int i=min_i;i<max_i;i++) aux += z[i]*q[i]*x[a-input_size*i];
            dL[a] += aux;
        }
        for(int i=0;i<output_size;i++){
            dL[input_size*output_size+i] += z[i]*q[i];
        }
    }
};

/*
We now define a (2d) convolutional layer.

Declare as
    conv(4);
for a convolutional layer with 4 filters. This uses the default filter size `(f_v,f_h) = (3,3)` and default stride `(s_v,s_h) = (1,1)`.
A different filter size can be declared as
    conv(4,5,6);
for 4 filters of size 5x6, or as
    conv(4,5,6,2,3);
for 4 filters of size 5x6 and stride 2x3. Note: we do not allow for padding so the stride should fit the input properly. If padding is
required, one should define a `padding` layer type that simply resizes its input into the proper size, and does nothing else.

We override all the default methods in the `layer` class so that, for example, the forward pass takes the usual form `p = sigma(w*x+b)`.
Note that the weights are a rank-4 array of size `o_d x i_d x f_v x f_h`, where `(o_d,o_v,o_h)` are the dimensions of the output tensor,
`(i_d,i_v,i_h)` are the dimensions of the input tensor. The weights and biases are stored as
`w[i][j][k][l] = theta[i_d*f_v*f_h*i + f_v*f_h*j + f_h*k + l]` and `b[i] = theta[o_d*i_d*f_v*f_h + i]`.
*/
class conv: public layer{
//TODO: Z is sparse...
public:
    conv(int n){
        assert(n >= 1);
        input_rank = output_rank = 3;
        input_shape = new int[3];
        output_shape = new int[3];
        output_shape[0] = n;
        f_v = 3;
        f_h = 3;
        s_v = 1;
        s_h = 1;
    }
    conv(int n, int fv, int fh, int sv = 1, int sh = 1){
        assert(n >= 1 && fv >= 1 && fh >= 1 && sv >= 1 && sh >= 1);
        input_rank = output_rank = 3;
        input_shape = new int[3];
        output_shape = new int[3];
        output_shape[0] = n;
        f_v = fv;
        f_h = fh;
        s_v = sv;
        s_h = sh;
    }

    int print(bool longQ = false, ostream & os = cout) override {
        assert(initQ);
        int i_d = input_shape[0], i_v = input_shape[1], i_h = input_shape[2],
            o_d = output_shape[0], o_v = output_shape[1], o_h = output_shape[2];
        os << "conv layer, input size: " << i_d << "x" << i_v << "x" << i_h
           << ", output size: " << o_d << "x" << o_v << "x" << o_h << endl;
        if(longQ){
            if(activ.name.size() > 0) os << "activation function: " << activ.name << endl;
            for(int i=0;i<o_d;i++){
                switch(i){
                case 0: os << "first kernel-\n"; break;
                case 1: os << "second kernel-\n"; break;
                case 2: os << "third kernel-\n"; break;
                default: os << i+1 << "th kernel-\n";
                }

                os << "weights:\n";
                full_print(theta,i_d,f_v,f_h,os);
                os << "\nbias: " << theta[o_d*i_d*f_v*f_h + i] << (i<o_d-1 ? "\n\n" : "\n");
            }
            if(s_v != 1 || s_h != 1){
                if(s_v == s_h) os << "(Stride: " << s_h << ")\n";
                else os << "(Vertical stride: " << s_v << ", Horizontal stride: " << s_h << ")\n";
            }
            os << "\nnumber of parameters: " << n_param << endl;
        }
        else{
            for(int i=0;i<o_d;i++){
                if(i == 0 || i == o_d-1){
                    if(i == 0) os << "first kernel-\n";
                    if(i == o_d-1) os << o_d << "th kernel-\n";
                    os << "weights:\n";
                    short_print(theta,i_d,f_v,f_h,os);
                    os << "\nbias: " << theta[o_d*i_d*f_v*f_h + i] << (i<o_d-1 ? "\n\n" : "\n");
                }
                else{
                    os << "..." << endl;
                }
            }
            if(s_v != 1 || s_h != 1){
                if(s_v == s_h) os << "(Stride: " << s_h << ")\n";
                else os << "(Vertical stride: " << s_v << ", Horizontal stride: " << s_h << ")\n";
            }
        }
        return n_param;
    }

    ~conv() override {
        delete[] input_shape;
        delete[] output_shape;
        if(initQ){
            delete[] p;
            delete[] q;
            delete[] theta;
            delete[] Z;
            delete[] dL;
        }
    }

    // conv * set_filter_size(int n){
    //     assert(n >= 1);
    //     f_v = f_h = n;
    //     return this;
    // }
    // conv * set_filter_size(int n_v, int n_h){
    //     assert(n_v >= 1 && n_h >= 1);
    //     f_v = n_v;
    //     f_h = n_h;
    //     return this;
    // }
protected:
    void init(int * prev_shape, int prev_rank) override {
        assert(!initQ);
        assert(prev_rank == 3);
        input_shape[0] = prev_shape[0];
        input_shape[1] = prev_shape[1];
        input_shape[2] = prev_shape[2];
        input_size = input_shape[0]*input_shape[1]*input_shape[2];
        if( (input_shape[1]-f_v) % s_v != 0 || (input_shape[2]-f_h) % s_h != 0){
            cerr << "stride doesn't fit, no padding allowed.\n";
            assert((input_shape[1]-f_v) % s_v != 0);
            assert((input_shape[2]-f_h) % s_h != 0);
        }
        if( f_v > input_shape[1] || f_h > input_shape[2]){
            cerr << "filter too large or input too small.\n";
            assert(f_v > input_shape[1]);
            assert(f_h > input_shape[2]);
        }
        output_shape[1] = (input_shape[1]-f_v)/s_v + 1;
        output_shape[2] = (input_shape[2]-f_h)/s_h + 1;
        assert(output_shape[1] >= 1 && output_shape[2] >= 1);
        output_size = output_shape[0]*output_shape[1]*output_shape[2];
        p = new double[output_size];
        q = new double[output_size];
        n_param = output_shape[0]*input_shape[0]*f_v*f_h + output_shape[0];
        theta = new double[n_param];
        Z = new double[input_size];
        dL = new double[n_param];
        initQ = true;
    }
    void randomize(double scale = 1.) override {
        assert(initQ);
        int i_d = input_shape[0], i_v = input_shape[1], i_h = input_shape[2],
            o_d = output_shape[0], o_v = output_shape[1], o_h = output_shape[2];
        for(int i=0;i<o_d;i++) for(int j=0;j<i_d;j++) for(int k=0;k<f_v;k++) for(int l=0;l<f_h;l++) theta[i_d*f_v*f_h*i + f_v*f_h*j + f_h*k + l] = rand_U(scale);
        for(int i=0;i<o_d;i++) theta[o_d*i_d*f_v*f_h + i] = rand_U(0.1*scale);
    }
    void forward(double * x) override {
        int i_d = input_shape[0], i_v = input_shape[1], i_h = input_shape[2],
            o_d = output_shape[0], o_v = output_shape[1], o_h = output_shape[2];
        //The code below is equivalent to the following, but slightly more efficient:
        // for(int a=0;a<o_d;a++){
        //     for(int b=0;b<o_v;b++){
        //         for(int c=0;c<o_h;c++){
        //             double aux = 0.;
        //             for(int i=0;i<i_d;i++) for(int j=0;j<f_v;j++) for(int k=0;k<f_h;k++) aux += theta[i_d*f_v*f_h*a + f_v*f_h*i + f_h*j + k]*x[i_v*i_h*i + i_h*(j+b*s_v) + (k+c*s_h)];
        //             aux += theta[o_d*i_d*f_v*f_h + a];
        //             p[o_v*o_h*a + o_h*b + c] = activ.f(aux);
        //             q[o_v*o_h*a + o_h*b + c] = activ.df(aux);
        //         }
        //     }
        // }
        int C1 = i_v*i_h;
        int C7 = o_v*o_h;
        int C8 = f_v*f_h;
        for(int a=0;a<o_d;a++){
            int C2 = i_d*C8*a;
            for(int b=0;b<o_v;b++){
                for(int c=0;c<o_h;c++){
                    int C9 = C7*a + o_h*b + c;
                    double aux = 0.;
                    for(int i=0;i<i_d;i++){
                        int C3 = C8*i;
                        int C4 = C1*i;
                        for(int j=0;j<f_v;j++){
                            int C5 = f_h*j;
                            int C6 = i_h*(j+b*s_v);
                            for(int k=0;k<f_h;k++){
                                aux += theta[C2+C3+C5+k]*x[C4+C6+(k+s_h*c)];
                            }
                        }
                    }
                    aux += theta[o_d*i_d*C8 + a];
                    p[C9] = activ.f(aux);
                    q[C9] = activ.df(aux);
                }
            }
        }
    }
    void backward(double * x) override {
        int i_d = input_shape[0], i_v = input_shape[1], i_h = input_shape[2],
            o_d = output_shape[0], o_v = output_shape[1], o_h = output_shape[2];
        //The code below is equivalent to this, but slightly more efficient:
        // for(int a=0;a<i_d;a++){
        //     for(int b=0;b<i_v;b++){
        //         int min_j = max(0,(b-f_v+s_v)/s_v);
        //         int max_j = min(o_v,1+b/s_v);
        //         for(int c=0;c<i_h;c++){
        //             int min_k = max(0,(c-f_h+s_h)/s_h);
        //             int max_k = min(o_h,1+c/s_h);
        //             double aux = 0.;
        //             for(int i=0;i<o_d;i++){
        //                 for(int j=min_j;j<max_j;j++){
        //                     for(int k=min_k;k<max_k;k++){
        //                         aux += x[o_v*o_h*i + o_h*j + k]*q[o_v*o_h*i + o_h*j + k]*theta[i_d*f_v*f_h*i + f_v*f_h*a + f_h*(b-j*s_v) + (c-k*s_h)];
        //                     }
        //                 }
        //             }
        //             Z[i_v*i_h*a + i_h*b + c] = aux;
        //         }
        //     }
        // }
        int C3 = o_v*o_h;
        int C4 = f_v*f_h;
        for(int b=0;b<i_v;b++){
            int min_j = max(0,(b-f_v+s_v)/s_v);
            int max_j = min(o_v,1+b/s_v);
            for(int c=0;c<i_h;c++){
                int min_k = max(0,(c-f_h+s_h)/s_h);
                int max_k = min(o_h,1+c/s_h);
                for(int a=0;a<i_d;a++){
                    double aux = 0.;
                    for(int i=0;i<o_d;i++){
                        int C1 = C3*i;
                        int C2 = C4*(i_d*i+a);
                        for(int j=min_j;j<max_j;j++){
                            int C5 = C1+o_h*j;
                            int C6 = f_h*(b-j*s_v);
                            for(int k=min_k;k<max_k;k++){
                                aux += x[C5+k]*q[C5+k]*theta[C2+C6+(c-k*s_h)];
                            }
                        }
                    }
                    Z[i_v*i_h*a + i_h*b + c] = aux;
                }
            }
        }
    }

    void zero_grad() override{
        for(int i=0;i<n_param;i++) dL[i] = 0.;
    }

    void update_derivatives(double * x, double * z) override {
        int i_d = input_shape[0], i_v = input_shape[1], i_h = input_shape[2],
            o_d = output_shape[0], o_v = output_shape[1], o_h = output_shape[2];
        for(int i=0;i<o_d;i++) for(int j=0;j<i_d;j++) for(int k=0;k<f_v;k++) for(int l=0;l<f_h;l++){
            double aux = 0.;
            for(int b=0;b<o_v;b++) for(int c=0;c<o_h;c++) aux += z[o_v*o_h*i + o_h*b + c]*q[o_v*o_h*i + o_h*b + c]*x[i_v*i_h*j + i_h*(k+b*s_v) + (l+c*s_h)];
            dL[i_d*f_v*f_h*i + f_v*f_h*j + f_h*k + l] += aux;
        }
        for(int i=0;i<o_d;i++){
            double aux = 0.;
            for(int b=0;b<o_v;b++) for(int c=0;c<o_h;c++) aux += z[o_v*o_h*i + o_h*b + c]*q[o_v*o_h*i + o_h*b + c];
            dL[o_d*i_d*f_v*f_h+i] += aux;
        }
    }

    int f_v = -1, f_h = -1, s_v = -1, s_h = -1;
};


/*
Next, we define a (2d) maxpool layer.

Declare as
    maxpool;
for a maxpool layer, using the default size `(f_v,f_h) = (2,2)`, or as
    maxpool(3,4);
for a layer with size 3x4. Note: again, we do not allow for padding so the filter size should fit the input properly. See the notes in
the `conv` layer if padding is required. Note also: we do not allow for overlaps either: the stride is always equal to the filter size.
*/
class maxpool: public layer{
public:
    maxpool(){
        input_rank = output_rank = 3;
        input_shape = new int[3];
        output_shape = new int[3];
        f_v = 2;
        f_h = 2;
    }
    maxpool(int fv, int fh){
        assert(fv >= 1 && fh >= 1);
        input_rank = output_rank = 3;
        input_shape = new int[3];
        output_shape = new int[3];
        f_v = fv;
        f_h = fh;
    }
    int print(bool longQ = false, ostream & os = cout) override {
        //TODO: print info about filter size
        os << "maxpool layer, output size: " << output_shape[0] << "x" << output_shape[1] << "x" << output_shape[2] << endl;
        return n_param;
    }

    ~maxpool() override {
        delete[] input_shape;
        delete[] output_shape;
        if(initQ){
            delete[] p;
            delete[] mp_v;
            delete[] mp_h;
            delete[] Z;
        }
    }

protected:
    void init(int * prev_shape, int prev_rank) override {
        assert(!initQ);
        assert(prev_rank == 3);
        input_shape[0] = prev_shape[0];
        input_shape[1] = prev_shape[1];
        input_shape[2] = prev_shape[2];
        input_size = input_shape[0]*input_shape[1]*input_shape[2];


        assert(input_shape[1] % f_v == 0);
        assert(input_shape[2] % f_h == 0);

        output_shape[0] = prev_shape[0];
        output_shape[1] = input_shape[1]/f_v;
        output_shape[2] = input_shape[2]/f_h;
        output_size = output_shape[0]*output_shape[1]*output_shape[2];

        p = new double[output_size];
        mp_v = new int[output_size];
        mp_h = new int[output_size];
        Z = new double[input_size];
        initQ = true;
    }
    void randomize(double scale = 1.) override {}
    void forward(double * x) override {
        int i_d = input_shape[0], i_v = input_shape[1], i_h = input_shape[2],
            o_d = output_shape[0], o_v = output_shape[1], o_h = output_shape[2];
        for(int a=0;a<o_d;a++){
            for(int b=0;b<o_v;b++){
                for(int c=0;c<o_h;c++){
                    int abc = o_v*o_h*a + o_h*b + c;
                    mp_v[abc] = b*f_v;
                    mp_h[abc] = c*f_h;
                    double aux = x[i_v*i_h*a + i_h*b*f_v + c*f_h];
                    for(int j=0;j<f_v;j++){
                        for(int k=0;k<f_h;k++){
                            if(x[i_v*i_h*a + i_h*(j+b*f_v) + (k+c*f_h)] > aux){
                                aux = x[i_v*i_h*a + i_h*(j+b*f_v) + (k+c*f_h)];
                                mp_v[abc] = j+b*f_v;
                                mp_h[abc] = k+c*f_h;
                            }
                        }
                    }
                    p[abc] = aux;
                }
            }
        }
    }
    void backward(double * x) override {
        int i_d = input_shape[0], i_v = input_shape[1], i_h = input_shape[2],
            o_d = output_shape[0], o_v = output_shape[1], o_h = output_shape[2];
        for(int i=0;i<i_d;i++) for(int j=0;j<i_v;j++) for(int k=0;k<i_h;k++){
            Z[i_v*i_h*i + i_h*j + k] = 0.;
        }
        for(int a=0;a<o_d;a++) for(int b=0;b<o_v;b++) for(int c=0;c<o_h;c++) Z[i_v*i_h*a + i_h*mp_v[o_v*o_h*a + o_h*b + c] + mp_h[o_v*o_h*a + o_h*b + c]] = x[o_v*o_h*a + o_h*b + c];
    }

    void zero_grad() override {}
    void update_derivatives(double*, double*) override {}

    int f_v = -1, f_h = -1;
    int * mp_v, * mp_h; // vertical and horizontal position of the max element
};

/*
Finally, we define a softmax layer.

Declare as
    softmax;
for a softmax layer, using the default weight `w = 1.`, or as
    maxpool(0.8);
for a layer with weight `w = 0.8`
*/
class softmax: public layer{
public:
    softmax(){
        weight = 1.;
    }
    softmax(double w){
        assert(w > 0.);
        weight = w;
    }
    int print(bool longQ = false, ostream & os = cout) override {
        if(weight == 1.) os << "softmax layer." << endl;
        else os << "softmax layer, weight: " << weight << endl;
        return 0;
    }

    ~softmax() override {
        if(initQ){
            delete[] p;
            delete[] Z;
        }
    }

protected:
    void init(int * prev_shape, int prev_rank) override {
        assert(!initQ);
        input_size = 1;
        for(int i=0;i<prev_rank;i++) input_size *= prev_shape[i];
        output_size = input_size;
        p = new double[output_size];
        Z = new double[input_size];
        initQ = true;
    }
    void randomize(double) override {}
    void forward(double * x) override {
        double max = x[0];
        for(int i=0;i<output_size;i++) max = max > x[i] ? max : x[i];
        double den = 0.;
        for(int i=0;i<output_size;i++) p[i] = exp(weight*(x[i]-max));
        for(int i=0;i<output_size;i++) den += p[i];
        for(int i=0;i<output_size;i++) p[i] /= den;
    }
    void backward(double * x) override {
        for(int i=0;i<input_size;i++){
            double aux = 0.;
            for(int j=0;j<output_size;j++) aux += x[j]*weight*p[i]*((i==j)-p[j]);
            Z[i] = aux;
        }

    }

    void zero_grad() override {}
    void update_derivatives(double*,double*) override {}

    double weight;
};



/*
We can now define a network class. It takes a list of pointers to various `layer`s, which can be any subclass, either the predefined
ones above (input, dense, conv, etc.), or entirely new ones. As long as the various methods (init, forward, backward, etc.) have been
specified, any layer type should work.

The attributes of a network are simply its depth ( = number of layers) and its architecture ( = list of pointers to all its layers,
sequentially).

The methods are the usual (`forward`, which performs a complete forward pass, `backward`, which performs a complete backward pass, etc.)
plus a few other utilities, such as `activation(int,int)`, which allows one to access the value of the activation ( = output) of hidden
layers, as well as `cost(double**,double**,int,loss)`, which computes the average loss across a list of samples, according to some choice
of loss function (default = least square).
*/
class network{
public:
    const long unsigned int depth;
    layer ** arch;

    network(initializer_list<layer*> lyrs):depth{lyrs.size()}{
        assert(depth > 0);
        arch = new layer*[depth];
        copy(lyrs.begin(),lyrs.end(),arch);

        bool no_layer_sharing = true; // check that all layers are distinct
        for(int l=0;l<depth;l++) for(int r=l+1;r<depth;r++) no_layer_sharing *= arch[l] != arch[r];
        assert(no_layer_sharing);

        arch[0]->init(); // initializes first layer
        for(int l=1;l<depth;l++) arch[l]->init(arch[l-1]->output_shape,arch[l-1]->output_rank); // initializes the rest of layers
                                                                                                // using the size of the previous one
    }
    layer * operator [](int i){ // returns pointer to `l`-th layer. TODO: perhaps return the layer itself?
        assert(i >= 0 && i < depth);
        return arch[i];
    }
    double activation(int l, int i){ // returns the `i`-th component of the output of the `l`-th layer
        assert(l >= 0 && l < depth);
        assert(i >= 0 && i < arch[l]->output_size);
        return arch[l]->p[i];
    }
    double activation(int i){ // returns the `i`-th component of the output of the last layer, i.e., of the network itself
        assert(i >= 0 && i<arch[depth-1]->output_size);
        return arch[depth-1]->p[i];
    }
    int output_size(int l){ // returns the size of the output of the `l`-th layer
        return arch[l]->output_size;
    }

    int print(bool longQ = false, ostream & os = cout){ // prints network details to stream, either in short or long format
        int n_param = 0;
        os << "------------------------------------------------------------------------------------------\n";
        for(int l=0;l<depth;l++){
            n_param += arch[l]->print(longQ,os);
            os << "------------------------------------------------------------------------------------------\n";
        }
        if(longQ) os << "total number of parameters: " << n_param << endl;
        return n_param;
    }

    void randomize(double scale = 1.){
        //TODO: add option for xavier-he
        for(int l=1;l<depth;l++) arch[l]->randomize(scale);
    }

    void forward(double * x){ // performs forward pass, each layer gets as input the output of the previous one
        //TODO: perhaps add optional int parameter `l_max` to stop the forward pass early (if last few activations are not needed)
        arch[0]->forward(x);
        for(int l=1;l<depth;l++) arch[l]->forward(arch[l-1]->p);
    }
    void backward(double * L){ // performs backward pass, each layer gets as input the gradients of the next one
        //TODO: perhaps add optional int parameter `l_min` to stop the backward pass early (if first few gradients are not needed)
        arch[depth-1]->backward(L);
        for(int l=depth-2;l>0;l--) arch[l]->backward(arch[l+1]->Z);
    }

    void zero_grad(){
        for(int l=1;l<depth;l++) arch[l]->zero_grad();
    }

    void update_derivatives(double * L){ // updates derivatives for a single sample. Each layer gets as input the output of the
                                         // previous layer, and the gradients of the next one.
        for(int l=1;l<depth-1;l++) arch[l]->update_derivatives(arch[l-1]->p, arch[l+1]->Z);
        arch[depth-1]->update_derivatives(arch[depth-2]->p, L);
    }

    double cost(double ** x, double ** y, int sample_size, loss loss_fnc = least_sq){ // computes the current cost, using some loss function
        double C = 0;
        for(int s=0;s<sample_size;s++){
            forward(x[s]);
            for(int a=0;a<arch[depth-1]->output_size;a++) C += loss_fnc.f(arch[depth-1]->p[a],y[s][a])/(sample_size*(arch[depth-1]->output_size));
        }
        return C;
    }

    ~network(){
        delete[] arch;
    }

    void clear(){ // if declared using `new` (not recommended).
        for(int l=0;l<depth;l++) delete arch[l];
    }
};


/*
The last ingredient is an optimizer, namely a class that takes a network and a list of samples, and iteratively performs forward and
backward passes using the derivatives to minimize the loss.

Here we define a base class, while different choices of optimization methods are handled via subclasses that override the `step` method
(which concretely specifies what to do with the derivatives computed after a given forward/backward pass).

We define stochastic gradient descent (SGD) and Adam below. User-defined optimizers are allowed too, simply inherit this class and
override the `step` method (and the constructor, of course).
*/
class optimizer{
public:
    loss loss_fnc = least_sq; // default loss function
    virtual ~optimizer() {}
    void train(double ** x, double ** y, int sample_size, int batch_size, double LR, int epochs, bool prog_bar = true, bool shuffle = false){

            if(sample_size % batch_size != 0) cerr << "warning, mini-batches don't fit!\n";
            cout << "Training parameters:\n-Number of samples: " << sample_size << "\n-Size of mini-batch: " << batch_size << "\n-Learning rate: " << LR << "\n-Number of epochs: " << epochs << endl;

            cout << "[training...]" << endl;
            for(int ep=0;ep<epochs;ep++){
                if(shuffle) shuffle_samples(x,y,sample_size); // TODO: perhaps, no need to shuffle the whole array, just the minibatches...
                //cout << " " << N->cost(x,y,sample_size,loss_fnc) << ",";
                //cout.flush();
                double C = 0; // estimate of the cost after each epoch. Note: not the exact loss, since parameters are updated after each mini-batch. Uncomment previous lines if exact loss is needed (will slow down training).
                for(int t=0;t<sample_size/batch_size;t++){
                    if(prog_bar) {progress_bar_before(t,sample_size/batch_size+1,80); cout.flush();}
                    N->zero_grad();
                    for(int u=0;u<batch_size;u++){
                        N->forward(x[t*batch_size+u]);
                        for(int i=0;i<output_size;i++) L[i] = loss_fnc.df(N->arch[depth-1]->p[i],y[t*batch_size+u][i])/(output_size*batch_size);
                        N->backward(L);
                        N->update_derivatives(L);
                        if(prog_bar) for(int i=0;i<output_size;i++) C += loss_fnc.f(N->arch[depth-1]->p[i],y[t*batch_size+u][i])/(output_size*sample_size);
                    }
                    step(LR);
                    if(prog_bar) progress_bar_after(t,sample_size/batch_size+1,80);
                }
                if(prog_bar) cout << " cost = " << C << endl;
            }
            cout << "[done]" << endl;
        }

protected:
    int depth, output_size;
    double * L;
    network * N;
    virtual void step(double){}
};

/*
Stochastic gradient descent optimizer, with optional momentum (on by default).

The basic step is theta -= LR*dL, where LR is the learning rate, and dL is the derivative.

With momentum, m = beta*m+(1-beta)*dL and theta -= LR*m, where beta is a hyperparameter (default = 0.9).

*/
class SGD: public optimizer{
public:
    SGD(network * ntwrk, bool mmntm = true, double b = 0.9){
        N = ntwrk;
        depth = N->depth;
        momentumQ = mmntm;
        if(momentumQ){
            m = new double*[depth];
            for(int l=1;l<depth;l++) m[l] = new double[N->arch[l]->n_param];
            for(int l=1;l<depth;l++) for(int i=0;i<N->arch[l]->n_param;i++) m[l][i] = 0.;
        }
        output_size = N->arch[depth-1]->output_size;
        L = new double[output_size];
        beta = b;
    }
    ~SGD() override {
        delete[] L;
        if(momentumQ){
            for(int l=1;l<depth;l++) delete[] m[l];
            delete[] m;
        }
    }

private:
    void step(double LR) override {
        for(int l=1;l<depth;l++){
            for(int i=0;i<N->arch[l]->n_param;i++){
                if(momentumQ){
                    m[l][i] = beta*m[l][i]+(1-beta)*N->arch[l]->dL[i];
                    N->arch[l]->theta[i] -= LR*m[l][i];
                }
                else N->arch[l]->theta[i] -= LR*N->arch[l]->dL[i];
            }
        }
    }

    double ** m;
    double beta;
    bool momentumQ;

};

/*
Adam optimizer.

The basic step is m = beta1*m + (1-beta1)*dL, v = beta2*v + (1-beta2)*dL**2, theta -= LR*m/sqrt(v), for some hyperparameters
beta1 and beta2 (default = 0.9 and 0.999, respectively).

We actually use unbiased estimates so we replace m by m/(1-beta1**n) and v by v/(1-beta2**n) in the last step, where n is the
current number of iterations.
*/
class Adam: public optimizer{
public:
    Adam(network * ntwrk, double b1 = 0.9, double b2 = 0.999){
        N = ntwrk;
        depth = N->depth;
        m = new double*[depth];
        for(int l=1;l<depth;l++) m[l] = new double[N->arch[l]->n_param];
        for(int l=1;l<depth;l++) for(int i=0;i<N->arch[l]->n_param;i++) m[l][i] = 0.;
        v = new double*[depth];
        for(int l=1;l<depth;l++) v[l] = new double[N->arch[l]->n_param];
        for(int l=1;l<depth;l++) for(int i=0;i<N->arch[l]->n_param;i++) v[l][i] = 0.;
        
        output_size = N->arch[depth-1]->output_size;
        L = new double[output_size];
        beta1 = beta1_n = b1;
        beta2 = beta2_n = b2;
    }
    ~Adam() override {
        delete[] L;
        for(int l=1;l<depth;l++) delete[] m[l];
        delete[] m;
        for(int l=1;l<depth;l++) delete[] v[l];
        delete[] v;
    }

private:
    void step(double LR) override {
        for(int l=1;l<depth;l++){
            for(int i=0;i<N->arch[l]->n_param;i++){
                m[l][i] = beta1*m[l][i]+(1-beta1)*(N->arch[l]->dL[i]);
                v[l][i] = beta2*v[l][i]+(1-beta2)*(N->arch[l]->dL[i])*(N->arch[l]->dL[i]);
                N->arch[l]->theta[i] -= LR*(m[l][i]/(1-beta1_n))/(sqrt((v[l][i])/(1-beta2_n))+.000000001); // small shift to prevent division by zero
            }
        }
        beta1_n *= beta1; beta2_n *= beta2;
    }

    double ** m, ** v;
    double beta1, beta2;
    double beta1_n, beta2_n;
};