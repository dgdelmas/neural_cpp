# Neural Network in C++

`neural.h` is a straightforward implementation of netural networks in C++. For now, only dense and convolutional layers have been implemented.

Also included a `header.h` file with some utility functions, to be loaded before `neural.h`.

See `nncpp.pdf` for an explanation of the math behind the code.

Example of usage:

Say we want a network with five dense layers, with `32`, `16`, `8`, `4`, `2` neurons, respectively. First, we declare the layers as follows:

    int depth = 5; // number of layers
    layer L[depth]; // array of layers
    L[0].set("dense", 32); // first layer, with 32 neurons
    L[1].set("dense", 16); // second layer, with 16 neurons
    L[2].set("dense", 8);  // third layer, with 8 neurons
    L[3].set("dense", 4);  // fourth layer, with 4 neurons
    L[4].set("dense", 2);  // fifth layer, with 2 neurons

The output to the network will be an array of length `2`; more generally, the output size is always equal to the number of neurons in the last layer.

The default activation is a soft version of the sign function. We can choose other activations as well. For example, say we want the second layer to be a ReLU; then, we can redefine

    L[1].activ = relu;

User-defined activations are allowed too; see `header.h` for a few examples. Another useful activation is `id`, namely `sigma(z) = z`. This just spits out the logit as is.
It is sometimes useful to use this for the last layer:

    L[4].activ = id;

For now, all these layers are independent. We will assamble them into a network next. Say that the input is an array of length `5`. Then, we declare the network as follows:

    int n_in = 5; // input size
    network N({n_in}, L, depth);

The layers are now part of a fully connected network. We can initialize the weights and biases to random numbers using

    N.randomize(/*scale = */ 0.7);

The network is ready to use. For example, say we want to pass the input `x = {.1,.2,.3,.4,.5}` through the network. This is done as follows:

    double * x;
    x = new double[n_in]; // declare length-5 array
    for(int i=0;i<n_in;i++) x[i] = (i+1)/10.;

    N.forward(x); // perform forward pass.

The activation of the `l`-th layer is accessed via `p_d[l]` (where `d` stands for "dense"; there is also `p_c[l]` for conv layers). For example, say we want the output of the second layer; this is done via

    int l = 1; // layer we wish to look at
    for(int i=0;i<N.arch[l].n_out;i++) cout << N.arch[l].p_d[i] << " ";

This will print `16` floats to the console. Note that they are non-negative, since this layer is a ReLU. If we want the output of the last layer, we simply look at `l = depth-1`.

Say that, instead, we have `6**5 = 7776` samples `(X,Y)` and we wish to train the network so that its output for a given `X` is as close as possible to the corresponding `Y`.
Assume that "close" here means least squares (which is the default loss function). As before, `X` is 5-dimensional, and assume that `0 <= X[i] <= 1`.
Assume also that the "real model" that we wish to learn is `Y[0] = sin(X[0]+X[1]+X[2]+X[3]+X[4])`, `Y[1] = 0.2*(cos(X[0]-X[1]+X[2]-X[3]+X[4])+X[2]-X[4])`.
Then we can proceed as follows:

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

    N.train(X, Y, n_samp, /*batch_size=*/n_samp, /*learning_rate=*/.02, /*num_of_epochs=*/500);

This will print the progress to the console; if we wish to suppress this we can run instead `N.train(..., /*prog_bar = */ false);`.
In any case, the final cost is around `10^-4`, which is roughly the average error for each sample. We can see that the model learned.
To check this further, let us print the expected output for a few concrete samples, say, those in the range `[1000,1006)`, and the corresponding prediction:

    // check that it worked
    for(int s=1000;s<1000+6;s++) cout << "(" << Y[s][0] << "," << Y[s][1] << ") "; cout << endl;
    cout << "vs" << endl;

    for(int s=1000;s<1000+6;s++){
        N.forward(X[s]);
        cout << "(" << N.arch[depth-1].p_d[0] << "," << N.arch[depth-1].p_d[1] << ") ";
    }
    cout << endl;

This prints `(0.14112,0.316013) (-0.0583741,0.32) (0.675463,0.192472) (0.515501,0.22806) (0.334988,0.259341) (0.14112,0.285067)` vs `(0.133904,0.298009) (-0.0786117,0.303425) (0.675002,0.206057) (0.493398,0.237544) (0.320523,0.26093) (0.133001,0.275874)`. It is reasonably close.

Let us move on to convolutional networks. These follow the exact same rules as before. Say we want two conv layers, followed by a dense layer, and finally a soft-max layer.
The default filter size is `3x3`, with a stride of `1`. Say we want the second layer to be `5x5` with a stride of `3`. Then, we can call

    int depth = 4; // number of layers
    layer L[depth];
    L[0].set("conv", 5); L[0].activ = relu; // first conv layer, with five filters
    L[1].set("conv", 8); L[1].activ = relu; // second conv layer, with eight filters
    L[2].set("dense", 10); L[2].activ = id; // one dense layer, with 10 neurons
    L[3].set("softmax");                    // a softmax layer at the end

    L[1].f_v = L[1].f_h = 5; // change filter size (v = vertical, h = horizontal)
    L[1].stride_h = L[1].stride_v = 3; // change stride

where we changed the activations into a pair of relu's and one identity.

We can assamble this into a network. For example, assume that the input will be a `28x28` picture in black and white (so only one channel). Then, the input size is

    int i_d = 1, i_v = 28, i_h = 28;
    
and we can declare the network as

    network N({i_d,i_v,i_h}, L, depth);

As before, we can initialize the weights at random:

    N.randomize();

Assume that we will use this network to identify handwritten digits, say, from the MNIST dataset. Then, given that this is categorical data, it makes more sense to replace the loss function by cross-entropy instead of least squares:

    N.loss_fnc = log_like;

Next, we load the data. I have a local csv file with `60000` lines, each of which contains, first, an integer from `0` to `9`, and then `28x28 = 784` integers from `0` to `255`. I load this data as follows:

    int n_samp = 60000; // number of samples in the mnist dataset
    ifstream file("mnist.csv"); // change path to your local file
    assert(file); // check that file loaded correctly
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

Note that we are using a one-hot representation for `y`, where `y=(0,...,0,1,0,...,0)`.

Let us use `n_tr = 58000` samples for training, and the remaining `n_samp - n_tr = 2000` for testing. Let us also use mini-batches of size `50`. We can train as usual:

    int n_tr = 58000;
    N.train(x, y, n_tr, /*batch_size=*/50, /*learning_rate=*/.01, /*num_of_epochs=*/2);

We can finally test the model:

    double correct = 0, all = 0;
    int max = 0;
    for(int s=0;s<n_samp-n_tr;s++){
        all++;
        N.forward(x[s+n_tr]);
        for(int i=0;i<10;i++) max = N.arch[depth-1].p_d[i]>N.arch[depth-1].p_d[max] ? i:max;
        correct += y[s+n_tr][max];
    }
    cout << "Accuracy: " << 100*correct/all << "%\n";


This returns `97.8%` accuracy. Pretty decent.