# Neural Network in C++

`neural.h` is a straightforward implementation of neural networks in C++. The usual basic layer types (dense, convolutional, maxpool, softmax) are implemented. The code is structured in such a way that adding new layer types should be relatively straightforward.

The basic object is the `layer` class, which has several default methods such as `init` (for "initialize"), `forward` (for the forward pass), `backward` (for the backward pass), etc. Defining a new layer type is done via subclasses, which should override these methods (and add new ones, if necessary):

    class MyLayer: public layer{
        void init() override {
            // your code goes here
        }
    };

The `network` class works with any subclass inherited from `layer`, as long as the various default methods have been overridden appropriately. See the predefined options (`dense`, `conv`, etc.) for practical examples. See also the companion pdf for the explanation of some of the math behind the code.


## Examples:

### Fully connected

Say we want a network that takes a `5`-dimensional array as input, and passes it through five dense layers, with `32`, `16`, `8`, `4`, `2` neurons, respectively. First, we declare the layers as follows:

    input i(5); // input layer, which specifies the size of the input
    dense d1(32), d2(16), d3(8), d4(4), d5(2); // five dense layers, which specify the number of neurons

The output to the network will be an array of length `2`; more generally, the output size is always equal to the number of neurons in the last layer.

The default activation is a soft version of the sign function. We can choose other activations as well. For example, say we want the second layer to be a ReLU; then, we can redefine

    d2.activ = relu;

User-defined activations are allowed too; see `header.h` for a few examples. Another useful activation is `id`, namely `sigma(z) = z`. This just spits out the logit as is.
It is sometimes useful to use this for the last layer:

    d5.activ = id;

For now, all these layers are independent. We will assemble them into a network next. We do this by feeding the addresses of these layers to an instance of the `network` class:

    network N{&i,&d1,&d2,&d3,&d4,&d5};

The layers are now part of a fully connected network. We can initialize the weights and biases to random numbers using

    N.randomize(/*scale = */ 0.7);

The network is ready to use. Note: we can print the details of the network to the console using `N.print()` (or `N.print(\*longQ=*\true)` for a long print, with all weights and biases).

Say we want to pass the input `x = {.1,.2,.3,.4,.5}` through the network. This is done as follows:

    double * x;
    x = new double[n_in]; // declare length-5 array
    for(int i=0;i<n_in;i++) x[i] = (i+1)/10.;

    N.forward(x); // perform forward pass.

The activation of the `l`-th layer is accessed via the `activation(int,int)` method, where the first input specifies the layer we are interested in, and the second one the specific component of the output of that layer. So for example, the following prints all activations, hidden or otherwise:

    for(int l=0;l<6;l++){ // print activations (hidden and output layers)
        cout << "activations of layer " << l << ": ";
        for(int i=0;i<N.output_size(l);i++) cout << N.activation(l,i) << " ";
        cout << endl;
    }

Note: if we are interested in the output of the last layer only, then we can omit the `l` argument, namely we use `N.activation(i)` for the `i`-th component.

Say that, instead, we have `6**5 = 7776` samples `(X,Y)` and we wish to train the network so that its output for a given `X` is as close as possible to the corresponding `Y`.
Assume that "close" here means least squares (which is the default loss function). As before, `X` is `5`-dimensional, and assume that `0 <= X[i] <= 1`.
Assume also that the "real model" that we wish to learn is `Y[0] = sin(X[0]+X[1]+X[2]+X[3]+X[4])`, `Y[1] = 0.2*(cos(X[0]-X[1]+X[2]-X[3]+X[4])+X[2]-X[4])`.
Then we can proceed as follows:

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

Next, we shuffle the samples and take the first `7000` as training samples, and the rest as testing samples:

    shuffle_samples(X,Y,n_samp); // shuffle the samples

    int n_tr = 7000; // training set

    cout << "mean L2 error before training: " << N.cost(X,Y,n_samp) << endl;

This prints `0.121097`, which is the average L2 error before training (when all weights and biases are random). Finally, in order to train the network, we create an optimizer and pass the address to the network as input, and then run the `train` method thereon. For concreteness we choose a simple optimizer, namely vanilla stochastic gradient descent:

    SGD optim(&N); // declare stochastic gradient descent optimizer (with momentum)
    optim.train(X, Y, n_tr, /*batch_size=*/1000, /*learning_rate=*/1, /*num_of_epochs=*/100, /*progress_bar=*/ false);

    cout << "mean L2 error after training, training set: " << N.cost(X,Y,n_tr) << endl;
    cout << "mean L2 error after training, testing set: " << N.cost(X+n_tr,Y+n_tr,n_samp-n_tr) << endl;

This will print an average error of `0.00136129` for the training set, and of `0.00181505` for the testing set. It seems that the model has learned the shape of `Y`. To check this further, let us print the expected output for a few concrete samples, say, those in the range `[7000,7006)`, and the corresponding prediction:

    // check that it worked
    for(int s=7000;s<7000+6;s++) cout << "(" << Y[s][0] << "," << Y[s][1] << ") "; cout << endl;
    cout << "vs" << endl;

    for(int s=7000;s<7000+6;s++){
        N.forward(X[s]);
        cout << "(" << N.activation(0) << "," << N.activation(1) << ") ";
    }
    cout << endl;

This prints `(0.334988,0.259341) (-0.0583741,0.344212) (-0.255541,0.196013) (0.14112,-0.0519395) (0.909297,0.36) (0.334988,0.299341)` vs `(0.354843,0.266417) (-0.0896823,0.385939) (-0.22518,0.184985) (0.133247,-0.040837) (0.810816,0.306458) (0.378517,0.280032)`. It is reasonably close.

Note: the actual numbers you get might be different from mine, since your random number generator might produce different numbers from mine (it might be implementation dependent, although it actually shouldn't...)

### Convolutional

Let us move on to convolutional networks. These follow the exact same rules as before. As a concrete example, we will use the MNIST dataset, which consists of `60000` samples of `28x28` pixel images (one color channel, so grayscale) containing handwritten digits in the range `0-9` (so ten classes).

We decide that we want two conv layers, followed by a dense layer, and finally a soft-max layer.
The default filter size is `3x3`, with a stride of `1`. Say we want the second layer to be `5x5` with a stride of `(3,3)`. Then, we can call

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

where we also changed the activations into a pair of relu's and one identity.

Next, we load the data. I have a local csv file with `60000` lines, each of which contains, first, an integer from `0` to `9`, and then `28x28 = 784` integers from `0` to `255`. I load this data as follows:

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

Note that we are using a one-hot representation for `y`, where `y=(0,...,0,1,0,...,0)`.

Let us use `n_tr = 58000` samples for training, and the remaining `n_samp - n_tr = 2000` for testing. Instead of stochastic gradient descent, let us speed up the training by switching into the Adam optimizer. Given that this dataset is relatively easy to learn, we can use a very small mini-batch, say, a size `40`, and, consequently, a single pass should be enough (i.e., a single epoch). So we can run

    int n_tr = 58000; // training set

    Adam optim(&N); // we use an Adam optimizer
    optim.loss_fnc = log_like;
    optim.train(x, y, n_tr, /*batch_size=*/40, /*learning_rate=*/.02, /*num_of_epochs=*/1);

Note that, unlike before, here we do not turn off the progress bar (i.e., we didn't add the line `/*progress_bar=*/ false`). So here we see the progress on the console. Also, we have changed the default loss function from the default "least squares" into "cross entropy", which is more useful when dealing with categorical data.

In any case, in just shy of a minute, training is done. We can test the result by checking the accuracy on both the training and the testing sets:

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


This returns `94.6%` and `96.9` accuracy, respectively. Pretty decent.

### A harder example

Finally, we consider a more complicated example, the CIFAR10 dataset. This is famously a harder dataset, so we use more or less the same architecture, but we add a pooling layer in between the convolutional layers to speed up the process.

So our network looks like so:

    input i(3,32,32); // this dataset has three color channels
    conv c1(5), c2(8,5,5,2,2);
    maxpool mp; // add maxpool layer
    dense d(10);
    softmax sm;
    network N{&i,&c1,&mp,&c2,&d,&sm};
    d.activ = id;

We load the samples as usual (I only use `30000` samples to make things easier for me; this is for illustration purposes only).

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
            x[line][j] = stod(str)/255.; // load predictor
        }
    }
    file1.close();
    ifstream file2("cifar10_y.txt");
    assert(file2);
    for(int line=0;line<n_samp;line++){
        getline(file2, str, ',');
        for(int i=0;i<10;i++) y[line][i] = 0;
        y[line][stoi(str)] = 1; // load response
    }
    file2.close();

We use `28800` samples for training and the rest for testing. We use a smaller learning rate than before, a slightly larger mini-batch, and more epochs:

    int n_tr = 28800; // training set

    Adam optim(&N); // we use an Adam optimizer
    optim.loss_fnc = log_like;
    optim.train(x, y, n_tr, /*batch_size=*/64, /*learning_rate=*/.005, /*num_of_epochs=*/35);

After around 25 minutes training is done, and we find a `50%` accuracy rate on both the training and the testing sets. Not bad for such a small model, although one could definitely do better, by e.g. using the full dateset instead of only half of it, a larger/deeper network, and a longer training process. We leave this to the user if they want to explore further.
