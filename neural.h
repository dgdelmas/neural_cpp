class layer{
    /*Auiary class that encapsulates layers. For now, the only implemented types are "dense", "conv", and "softmax", for fully connected, convolutional, and
    softmax, respectively. Declare as `layer L{"dense",3};` for a three-neuron layer, as `layer L{"conv",4};` for a four-filter conv layer, and as `layer L{"softmax"}`;
    alternatively, use empty declaration `layer L;` and set type and size afterwards, via `L.set("dense",3);`.

    The default activation is `softsign`, i.e., x/(1+abs(x)), other predefined options are `relu` and `sigmoid`, while user-defined activations are allowed,
    handled via the `activation` struct defined in `header.h`. For a softmax layer, the activation is irrelevant.

    Other parameters are: `n_in` and `n_out` for dense layers, corresponding to the size of the input and output, respectively; `inp_d`, `inp_v`, `inp_h` for
    conv layers, corresponding to the depth/vertical size/horizontal size of the input, respectively, and `out_d`, `out_v`, `out_h` for the same parameters in the
    output; `stride_v` and `stride_h` for the stride of conv layers and `f_v`, `f_h` for the vertical and horizontal size of the filter. For softmax, `n_out == `n_in`.
    The default filter size for conv layers is `f_v = f_h = 3`, and the default stride is `stride_v = stride_h = 1`.

    Before a layer can be used for anything, one must run `L.initialize()`, which reserves memory for the weights and biases, and other variables that the layer
    requires. Note: the `network` class automatically initializes its layers, so no need to do this manually if the layer is going to be used by a network.

    Use `L.randomize(s)` to set weights and biases to random values, taken from a uniform distribution from `-s` to `+s`. Default `s` is 1.

    Use `L.forward(x)` to run a forward pass with input `x`. This computes `p` and `q`, where `p := activation(weights . x + bias)` and `q := activation'(weights . x + bias)`.

    Finally, use `<<` to print layer details to a stream, e.g. `cout << L;`.
    */
    public:
        activation activ = softsign; // default = x/(1+abs(x))

        int n_in = 0, n_out = 0; // for dense
        int stride_h = 1, stride_v = 1; // for conv
        int inp_h = 0, inp_v = 0, inp_d = 0; // for conv
        int out_d, out_h, out_v; // for conv
        int f_h = 3, f_v = 3; // for conv

        double **** filter; // for conv
        double ** weights; // for dense
        double * biases;   // for conv and dense

        double * p_d; // p := activation(wx+b), for dense
        double * q_d; // q := activation'(wx+b), for dense
        double *** p_c; // p := activation(fx+b), for conv
        double *** q_c; // q := activation'(fx+b), for conv

        //Note: might want to unify d's and c's into a single object, by flattening the various arrays into 1-dimensional ones. Leave this for the future...
        friend class network;

        layer(){}
        layer(string tp){
            set(tp);
        }
        layer(string tp, int d){
            set(tp,d);
        }
        void set(string tp){
            assert(tp == "softmax");
            type = tp;
        }
        void set(string tp, int d){
            if(tp == "dense"){
                if(d <= 0) cerr << "Number of neurons must be positive!\n";
                assert(d > 0);
                n_out = d;
            }
            else if(tp == "conv"){
                if(d <= 0) cerr << "Filter size must be positive!\n";
                assert(d > 0);
                out_d = d;
            }
            else if(tp == "softmax"){
                if(d <= 0) cerr << "Number of inputs/outputs must be positive!\n";
                assert(d > 0);
                n_out = d;
            }
            else{
                cerr << "Unsupported layer type.\n";
                assert(false);
            }
            type = tp;
        }

        void initialize(){ // checks that parameters make sense and, if so, reserves memory for weights and biases. (Also, sets `initializedQ` to `true`)
            if(!(type == "dense" || type == "conv" || type == "softmax")){
                cerr << "Unknown layer type!\n";
                assert(type == "dense" || type == "conv" || type == "softmax");
            }
            if(type == "dense"){
                if(n_out <= 0 || n_in <= 0){
                    cerr << "Input/output size to dense layer must be a positive integer!\n";
                    assert(n_in > 0 && n_out > 0);
                }
                weights = new double*[n_out];
                biases = new double[n_out];
                for(int i=0;i<n_out;i++) weights[i] = new double[n_in];
                p_d = new double[n_out];
                q_d = new double[n_out];
            }
            else if(type == "conv"){
                if(inp_h <= 0 || inp_v <= 0 || inp_d <=0 || out_d <=0){
                    cerr << "Input/output size to conv layer must be a positive integer!\n";
                    assert(inp_h > 0 && inp_v > 0 && inp_d > 0 && out_d > 0);
                }
                if(f_h <=0 || f_v <=0 || stride_h < 1 || stride_v < 1){
                    cerr << "Filter size and stride must be positive integers!\n";
                    assert(f_h > 0 && f_v > 0 && stride_h >= 1 && stride_v >= 1);
                }
                if( (inp_v-f_v) % stride_v !=0 || (inp_h-f_h) % stride_h !=0){
                    cerr << "Stride doesn't fit! No padding allowed...\n";
                    assert(( (inp_v-f_v) % stride_v == 0 && (inp_h-f_h) % stride_h == 0));
                }
                if( f_v > inp_v || f_h > inp_h){
                    cerr << "filter too large or input too small!\n";
                    assert( f_v <= inp_v && f_h <= inp_h);
                }

                out_v = (inp_v-f_v)/stride_v + 1;
                out_h = (inp_h-f_h)/stride_h + 1;

                if(out_v <= 0 || out_h <= 0) cerr << "filter don't fit!\n";
                assert(out_v > 0 && out_h > 0);

                biases = new double[out_d];

                filter = new double***[out_d];
                for(int i=0;i<out_d;i++) filter[i] = new double**[inp_d];
                for(int i=0;i<out_d;i++) for(int j=0;j<inp_d;j++) filter[i][j] = new double*[f_v];
                for(int i=0;i<out_d;i++) for(int j=0;j<inp_d;j++) for(int k=0;k<f_v;k++) filter[i][j][k] = new double[f_h];

                p_c = new double**[out_d];
                for(int i=0;i<out_d;i++) p_c[i] = new double*[out_v];
                for(int i=0;i<out_d;i++) for(int j=0;j<out_v;j++) p_c[i][j] = new double[out_h];

                q_c = new double**[out_d];
                for(int i=0;i<out_d;i++) q_c[i] = new double*[out_v];
                for(int i=0;i<out_d;i++) for(int j=0;j<out_v;j++) q_c[i][j] = new double[out_h];
            }
            else if(type == "softmax"){
                p_d = new double[n_out];
            }
            else cerr << "Unclear instructions...\n";
            initializedQ = true;

        }
        ~layer(){ // if layer has been initialized, frees reserved memory.
            if(initializedQ){
                if(type == "dense"){
                    for(int i=0;i<n_out;i++) delete[] weights[i];   
                    delete[] weights;
                    delete[] biases;
                    delete[] p_d;
                    delete[] q_d;
                }
                else if(type == "conv"){
                    delete[] biases;
                    for(int i=0;i<out_d;i++) for(int j=0;j<inp_d;j++) for(int k=0;k<f_v;k++) delete[] filter[i][j][k];
                    for(int i=0;i<out_d;i++) for(int j=0;j<inp_d;j++) delete[] filter[i][j];
                    for(int i=0;i<out_d;i++) delete[] filter[i];
                    delete[] filter;
                    for(int i=0;i<out_d;i++) for(int j=0;j<out_v;j++) delete[] p_c[i][j];
                    for(int i=0;i<out_d;i++) delete[] p_c[i];
                    delete[] p_c;
                    for(int i=0;i<out_d;i++) for(int j=0;j<out_v;j++) delete[] q_c[i][j];
                    for(int i=0;i<out_d;i++) delete[] q_c[i];
                    delete[] q_c;
                }
                else if(type == "softmax") delete[] p_d;
                else cerr << "unknown error...\n";
                //initializedQ = false;
            }
        }

        void randomize(double scale = 1){
            if(!initializedQ){
                cerr << "Layer not initialized!\n";
                assert(initializedQ);
            }
            if(type == "dense"){
                for(int i=0;i<n_out;i++){
                    biases[i] = 0.1*rand_U(scale);
                    for(int j=0;j<n_in;j++) weights[i][j] = rand_U(scale);
                }
            }
            else if(type == "conv"){
                for(int i=0;i<out_d;i++) for(int j=0;j<inp_d;j++) for(int k=0;k<f_v;k++) for(int l=0;l<f_h;l++) filter[i][j][k][l] = rand_U(scale);
                for(int i=0;i<out_d;i++) biases[i] = rand_U(0.1*scale);
            }
            else if(type == "softmax"){
                cerr << "Warning: nothing to randomize in a softmax layer...\n";
            }
            else cerr << "unknown error...\n";
        }
        void forward(double * input){
            double aux;
            if(type == "dense"){
                for(int i=0;i<n_out;i++){
                    aux = 0;
                    for(int j=0;j<n_in;j++) aux += weights[i][j]*input[j];
                    aux += biases[i];
                    p_d[i] = activ.f(aux);
                    q_d[i] = activ.df(aux);

                }
            }
            else if(type == "softmax"){
                for(int i=0;i<n_out;i++) p_d[i] = exp(input[i]);
                aux = 0;
                for(int i=0;i<n_out;i++) aux += p_d[i];
                for(int i=0;i<n_out;i++) p_d[i] /= aux;
            }
            else if(type == "conv"){
                cerr << "Input to conv layer must be rank-3!\n";
                assert(false);
            }
            else cerr << "unknown error...\n";
        }
        void forward(double *** input){
            double aux;
            if(type == "conv"){
                for(int a=0;a<out_d;a++){
                    for(int b=0;b<out_v;b++){
                        for(int c=0;c<out_h;c++){
                            aux = 0;
                            for(int i=0;i<inp_d;i++) for(int j=0;j<f_v;j++) for(int k=0;k<f_h;k++) aux += filter[a][i][j][k]*input[i][j+b*stride_v][k+c*stride_h];
                            aux += biases[a];
                            p_c[a][b][c] = activ.f(aux);
                            q_c[a][b][c] = activ.df(aux);
                        }
                    }
                }
            }
            else cerr << "unknown error...\n";
        }

        friend ostream& operator<<(ostream& os, const layer& L){ //prints layer to stream
            if(!L.initializedQ) os << "empty layer.\n";
            else if(L.type == "dense"){
                os << "weights: {";
                for(int i=0;i<L.n_out;i++){
                    os << "{";
                    if(L.n_in == 0) os << "?}";
                    for(int j=0;j<L.n_in;j++){
                        os << L.weights[i][j] << (j<L.n_in-1 ? "," : "}");
                    }
                    os << (i<L.n_out-1 ? "," : "}");
                }
                os << "\nbiases: {";
                for(int i=0;i<L.n_out;i++) os << L.biases[i] << (i<L.n_out-1 ? "," : "}\n");
            }
            else if(L.type == "conv"){
                for(int i=0;i<L.out_d;i++){
                    switch(i){
                    case 0: os << "first kernel-\n"; break;
                    case 1: os << "second kernel-\n"; break;
                    case 2: os << "third kernel-\n"; break;
                    default: os << i+1 << "th kernel-\n";
                    }

                    os << "weights: {";
                    for(int j=0;j<L.inp_d;j++){
                        os << "{";
                        for(int k=0;k<L.f_v;k++){
                            os << "{";
                            for(int l=0;l<L.f_h;l++){
                                os << L.filter[i][j][k][l] << (l<L.f_h-1 ? ",":"}");
                            }
                            os << (k<L.f_v-1 ? ",":"}");
                        }
                        os << (j<L.inp_d-1 ? ",":"}");
                    }
                    os << "\nbias: " << L.biases[i] << (i<L.out_d-1 ? "\n\n" : "\n");
                }
                if(L.stride_v != 1 || L.stride_h != 1){
                    if(L.stride_v == L.stride_h) os << "(Stride: " << L.stride_h << ")\n";
                    else os << "(Vertical stride: " << L.stride_v << ", Horizontal stride: " << L.stride_h << ")\n";
                }
            }
            else if(L.type == "softmax") os << "Soft-max layer.\n";
            else cerr << "unknown error...\n";
            return os;
        }

    private:
        string type = "empty";
        bool initializedQ = false;
};


class network{
    /*Class for neural network. For now, only supports convolutional and dense layers, as well as softmax. Fully connected network is fine, i.e., conv layers need not
    appear at all, but if they do, they can never postcede a dense one, and last layer must always be dense (or dense plus softmax). If last layer is softmax, it is
    recommended to use `activ = id` (or `relu`) for the previous layer, so that the input to softmax is a logit instead of a probability.

    To do: option for maxpool, batch normalization, xavier-he initialization, etc.

    Declare as `network NN({input size}, pointer to list of layers, lenght of list of layers)`. For example, a network with 3x10x10 input, one conv layer with
    two filters, and one dense layer with nine neurons, is created as follows:
    int depth = 2;
    layer L[depth];
    L[0].set("conv",2);
    L[1].set("dense",9);
    network NN({3,10,10},L,depth);

    If all layers are dense, input size is allowed to be either 3-dim as above, or 1-dim. For example, if the input is 1-dim with length six, and the network consists
    of one dense layer with 11 neurons and one with 15 neurons, plus a softmax layer, then we can declare this as follows:
    int depth = 3;
    layer L[depth];
    L[0].set("dense",11);
    L[1].set("dense",15); L[1].activ = id;
    L[2].set("softmax");
    network NN({6},L,depth);

    The default loss function is least squares, loss(a,b) = .5*(a-b)*(a-b), user-defined functions are allowed, handled via the `loss` struct defined in the `header.h`
    file. For example, use `NN.loss_fnc = log_like;` for cross-entropy.

    Training is done using the Adam method, with default parameters `adam_b1 = 0.9` and `adam_b2 = 0.999`. Run `NN.train(x, y, sample_size, batch_size,
    learning_rate, number_of_epochs)` to run mini-batch gradient descent, minimizing `sum_s loss_fnc(p(x[s]),y[s])`.

    To do: allow for other grad-desc methods.

    Use `NN.randomize(s)` to set all weights and biases to random numbers between `-s` and `+s`. Use `print_to_file(filename)` to write weights and biases to textfile
    and `read_from_file(filename)` to load weights and biases from textfile.

    Use `NN.forward(x)` to perform forward passes, using `x` as input to the network. This computes `p` and `q`, where `p := activation(weights . x + bias)` and
    `q := activation'(weights . x + bias)`.

    */
    public:
        int depth = 0; // number of layers, not counting softmax (if it appears at all)
        int input_size_v = 0, input_size_h = 0, input_size_d = 0; // for conv
        int output_size_v, output_size_h, output_size_d; // for conv
        int input_size_dense, output_size_dense; // for dense
        layer * arch; // architecture i.e., list of layers
        loss loss_fnc = least_sq; // default loss function: least squares
        double adam_b1 = 0.9, adam_b2 = 0.999; // training parameters for Adam iteration

        network(initializer_list<int> inputDims, layer * list_of_layers, const int list_length, bool print_details = true){
            //TODO: check that, if last activation is not non-negative, then loss makes sense for negative arguments

            assert(list_length > 0);
            arch = list_of_layers;
            depth = list_length;

            for(int l=0;l<depth-1;l++){ // check that there are no intermediate softmax layers
                if(arch[l].type == "softmax") cerr << "Only last layer may be softmax!\n";
                assert(arch[l].type != "softmax");
            }
            if(arch[depth-1].type == "softmax"){ // check if last layer is softmax or not
                softmaxQ = true;
                depth--;
                if(depth == 0) cerr << "At least one dense layer required.\n";
                assert(depth > 0);
            }

            for(int l=0;l<depth;l++){ // count how many conv layers there are, and that they all appear at the beginning
                if(arch[l].type == "conv"){
                    num_of_conv++;
                    if(l>0 && arch[l-1].type != "conv"){
                        cerr << "Error, cannot have conv layer after a non-conv layer.\n";
                        assert(false);
                    }
                }
            }
            if(num_of_conv == depth){
                cerr << "Must have at least one dense layer.\n";
                assert(false);
            }
            if(arch[0].type == "conv"){ // initializes conv layers
                if(inputDims.size() != 3){
                    cerr << "Unclear initializer: first argument to network must be a list of three positive integers.\n";
                    assert(false);
                }
                input_size_d = *(inputDims.begin());
                input_size_v = *(inputDims.begin()+1);
                input_size_h = *(inputDims.begin()+2);
                assert(input_size_d > 0 && input_size_v > 0 && input_size_h);
                arch[0].inp_d = input_size_d;
                arch[0].inp_v = input_size_v;
                arch[0].inp_h = input_size_h;
                arch[0].initialize();
                for(int l=1;l<num_of_conv;l++){
                    arch[l].inp_d = arch[l-1].out_d;
                    arch[l].inp_v = arch[l-1].out_v;
                    arch[l].inp_h = arch[l-1].out_h;
                    arch[l].initialize();
                }
                output_size_d = arch[num_of_conv-1].out_d;
                output_size_v = arch[num_of_conv-1].out_v;
                output_size_h = arch[num_of_conv-1].out_h;
            }
            else if(arch[0].type == "dense"){ // initializes dense layers
                if(inputDims.size() == 1) input_size_dense = *(inputDims.begin());
                else if(inputDims.size() == 3){
                    input_size_d = *(inputDims.begin());
                    input_size_v = *(inputDims.begin()+1);
                    input_size_h = *(inputDims.begin()+2);
                    input_size_dense = input_size_d*input_size_v*input_size_h;
                }
                else{
                    cerr << "Unclear initializer: first argument to network must be a list of three positive integers, or one positive integer.\n";
                    assert(false);
                }
                //output_size_dense = ;
            }
            else cerr << "unclear instructions...\n";

            if(num_of_conv > 0) input_size_dense = output_size_d*output_size_v*output_size_h;

            arch[num_of_conv].n_in = input_size_dense;
            arch[num_of_conv].initialize();
            for(int i=num_of_conv+1;i<depth;i++){
                arch[i].n_in = arch[i-1].n_out;
                arch[i].initialize();
            }
            output_size_dense = arch[depth-1].n_out;

            if(softmaxQ){
                arch[depth].n_out = output_size_dense;
                arch[depth].initialize();
            }


            Xi_d = new double*[depth-num_of_conv]; // d(cost)/d(bias) for a single sample, for dense layers
            for(int l=0;l<depth-num_of_conv;l++) Xi_d[l] = new double[arch[l+num_of_conv].n_out];

            Xi_c = new double***[num_of_conv]; // d(cost)/d(bias) for a single sample, for conv layers
            for(int l=0;l<num_of_conv;l++) Xi_c[l] = new double**[arch[l].out_d];
            for(int l=0;l<num_of_conv;l++) for(int i=0;i<arch[l].out_d;i++) Xi_c[l][i] = new double*[arch[l].out_v];
            for(int l=0;l<num_of_conv;l++) for(int i=0;i<arch[l].out_d;i++) for(int j=0;j<arch[l].out_v;j++) Xi_c[l][i][j] = new double[arch[l].out_h];

            flatten = new double[input_size_dense]; // flattens rank-3 output into rank-1

            jf_c = new double****[num_of_conv]; // d(cost)/d(filter), for conv layer
            for(int l=0;l<num_of_conv;l++) jf_c[l] = new double***[arch[l].out_d];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) jf_c[l][a] = new double**[arch[l].inp_d];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) jf_c[l][a][i] = new double*[arch[l].f_v];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) for(int j=0;j<arch[l].f_v;j++) jf_c[l][a][i][j] = new double[arch[l].f_h];
            mf_c = new double****[num_of_conv]; // for Adam
            for(int l=0;l<num_of_conv;l++) mf_c[l] = new double***[arch[l].out_d];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) mf_c[l][a] = new double**[arch[l].inp_d];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) mf_c[l][a][i] = new double*[arch[l].f_v];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) for(int j=0;j<arch[l].f_v;j++) mf_c[l][a][i][j] = new double[arch[l].f_h];
            vf_c = new double****[num_of_conv]; // for Adam
            for(int l=0;l<num_of_conv;l++) vf_c[l] = new double***[arch[l].out_d];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) vf_c[l][a] = new double**[arch[l].inp_d];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) vf_c[l][a][i] = new double*[arch[l].f_v];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) for(int j=0;j<arch[l].f_v;j++) vf_c[l][a][i][j] = new double[arch[l].f_h];

            jb_c = new double*[num_of_conv]; // d(cost)/d(bias), for conv layer
            for(int l=0;l<num_of_conv;l++) jb_c[l] = new double[arch[l].out_d];
            mb_c = new double*[num_of_conv]; // for Adam
            for(int l=0;l<num_of_conv;l++) mb_c[l] = new double[arch[l].out_d];
            vb_c = new double*[num_of_conv]; // for Adam
            for(int l=0;l<num_of_conv;l++) vb_c[l] = new double[arch[l].out_d];

            jw_d = new double**[depth-num_of_conv]; // d(cost)/d(weight), for dense
            for(int i=0;i<depth-num_of_conv;i++) jw_d[i] = new double*[arch[i+num_of_conv].n_out];
            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) jw_d[i][j] = new double[arch[i+num_of_conv].n_in];
            mw_d = new double**[depth-num_of_conv]; // for Adam
            for(int i=0;i<depth-num_of_conv;i++) mw_d[i] = new double*[arch[i+num_of_conv].n_out];
            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) mw_d[i][j] = new double[arch[i+num_of_conv].n_in];
            vw_d = new double**[depth-num_of_conv]; // for Adam
            for(int i=0;i<depth-num_of_conv;i++) vw_d[i] = new double*[arch[i+num_of_conv].n_out];
            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) vw_d[i][j] = new double[arch[i+num_of_conv].n_in];

            jb_d = new double*[depth-num_of_conv]; // d(cost)/d(bias), for dense
            for(int i=0;i<depth-num_of_conv;i++) jb_d[i] = new double[arch[i+num_of_conv].n_out];
            mb_d = new double*[depth-num_of_conv]; // for Adam
            for(int i=0;i<depth-num_of_conv;i++) mb_d[i] = new double[arch[i+num_of_conv].n_out];
            vb_d = new double*[depth-num_of_conv]; // for Adam
            for(int i=0;i<depth-num_of_conv;i++) vb_d[i] = new double[arch[i+num_of_conv].n_out];

            if(print_details){
                cout << "Network initialized:\n";
                print_size();
                cout << endl;
            }

        }
        ~network(){ // frees allocated arrays

            for(int l=0;l<depth-num_of_conv;l++) delete[] Xi_d[l];
            delete[] Xi_d;

            for(int l=0;l<num_of_conv;l++) for(int i=0;i<arch[l].out_d;i++) for(int j=0;j<arch[l].out_v;j++) delete[] Xi_c[l][i][j];
            for(int l=0;l<num_of_conv;l++) for(int i=0;i<arch[l].out_d;i++) delete[] Xi_c[l][i];
            for(int l=0;l<num_of_conv;l++) delete[] Xi_c[l];
            delete[] Xi_c;

            delete[] flatten;
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) for(int j=0;j<arch[l].f_v;j++) delete[] jf_c[l][a][i][j];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) delete[] jf_c[l][a][i];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) delete[] jf_c[l][a];
            for(int l=0;l<num_of_conv;l++) delete[] jf_c[l];
            delete[] jf_c;
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) for(int j=0;j<arch[l].f_v;j++) delete[] mf_c[l][a][i][j];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) delete[] mf_c[l][a][i];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) delete[] mf_c[l][a];
            for(int l=0;l<num_of_conv;l++) delete[] mf_c[l];
            delete[] mf_c;
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) for(int j=0;j<arch[l].f_v;j++) delete[] vf_c[l][a][i][j];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) delete[] vf_c[l][a][i];
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) delete[] vf_c[l][a];
            for(int l=0;l<num_of_conv;l++) delete[] vf_c[l];
            delete[] vf_c;

            for(int l=0;l<num_of_conv;l++) delete[] jb_c[l];
            delete[] jb_c;
            for(int l=0;l<num_of_conv;l++) delete[] mb_c[l];
            delete[] mb_c;
            for(int l=0;l<num_of_conv;l++) delete[] vb_c[l];
            delete[] vb_c;

            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) delete[] jw_d[i][j];
            for(int i=0;i<depth-num_of_conv;i++) delete[] jw_d[i];
            delete[] jw_d;
            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) delete[] mw_d[i][j];
            for(int i=0;i<depth-num_of_conv;i++) delete[] mw_d[i];
            delete[] mw_d;
            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) delete[] vw_d[i][j];
            for(int i=0;i<depth-num_of_conv;i++) delete[] vw_d[i];
            delete[] vw_d;

            for(int i=0;i<depth-num_of_conv;i++) delete[] jb_d[i];
            delete[] jb_d;
            for(int i=0;i<depth-num_of_conv;i++) delete[] mb_d[i];
            delete[] mb_d;
            for(int i=0;i<depth-num_of_conv;i++) delete[] vb_d[i];
            delete[] vb_d;
        }
        void randomize(double scale = 1){ // randomizes weights and biases
            for(int i=0;i<depth;i++) arch[i].randomize(scale);
        }
        void forward(double * input){ // performs forward pass, with rank-1 input (fully connected networks only)
            if(num_of_conv != 0){
                cerr << "Error: input to conv layer must be a rank-3 array.\n";
                assert(false);
            }
            arch[0].forward(input);
            for(int i=1;i<depth+softmaxQ;i++) arch[i].forward(arch[i-1].p_d);
        }
        void forward(double *** input){ // performs forward pass, with rank-3 input
            if(num_of_conv == 0){
                if(input_size_h*input_size_v*input_size_d == 0) cerr << "Error: cannot read rank-3 data without size information!\n";
                assert(input_size_h*input_size_v*input_size_d > 0);
                for(int i=0;i<input_size_dense;i++) flatten[i] = input[(i-((i-i%input_size_h)/input_size_h % input_size_v))/(input_size_h*input_size_v)][(i-i%input_size_h)/input_size_h % input_size_v][i%input_size_h];
            }
            else{
                arch[0].forward(input);
                for(int i=1;i<num_of_conv;i++) arch[i].forward(arch[i-1].p_c);
                for(int i=0;i<input_size_dense;i++) flatten[i] = arch[num_of_conv-1].p_c[(i-((i-i%output_size_h)/output_size_h % output_size_v))/(output_size_h*output_size_v)][(i-i%output_size_h)/output_size_h % output_size_v][i%output_size_h]; //a=output_size_v*output_size_h*i+ output_size_h*j+k
            }
            arch[num_of_conv].forward(flatten);
            for(int i=num_of_conv+1;i<depth+softmaxQ;i++) arch[i].forward(arch[i-1].p_d);

        }
        double cost(double **** x, double ** y, int sample_size){ // computes C = sum_s loss(x[s],y[s])
            double C = 0;
            for(int s=0;s<sample_size;s++){
                forward(x[s]);
                for(int a=0;a<output_size_dense;a++) C += loss_fnc.f(arch[depth-1+softmaxQ].p_d[a],y[s][a])/(sample_size*output_size_dense);
            }
            return C;
        }
        double cost(double ** x, double ** y, int sample_size){
            if(num_of_conv != 0){
                cerr << "Error: input to conv layer must be a rank-3 array.\n";
                assert(false);
            }
            double C = 0;
            for(int s=0;s<sample_size;s++){
                forward(x[s]);
                for(int a=0;a<output_size_dense;a++) C += loss_fnc.f(arch[depth-1+softmaxQ].p_d[a],y[s][a])/(sample_size*output_size_dense);
            }
            return C;
        }

        void train(double **** x, double ** y, int sample_size, int batch_size, double LR, int epochs, bool prog_bar = true){ // trains network, using rank-3 input
            // TODO: option to shuffle mini batches

            if(sample_size % batch_size !=0) cerr << "warning, mini-batches don't fit!\n";
            cout << "[training...]\n"; cout.flush();
            double beta1, beta2;
            beta1 = adam_b1, beta2 = adam_b2;
            set_mv_to_zero(); // this and previous line: apparently, inside `ep` loop works better if `epochs` is small...

            for(int ep=0;ep<epochs;ep++){
                //cout << " " << cost(x,y,sample_size) << ",";
                //cout.flush();
                if(prog_bar) progress_bar_before(ep,epochs+1,40);
                for(int t=0;t<sample_size/batch_size;t++){
                    set_j_to_zero();
                    for(int u=0;u<batch_size;u++){
                        forward(x[t*batch_size+u]);
                        update_derivatives(x[t*batch_size+u],y[t*batch_size+u],batch_size*output_size_dense);
                    }
                    run_adam(beta1,beta2,LR);
                }
                if(prog_bar) {progress_bar_after(ep,epochs+1,40); cout.flush();}
            }
            cout << "\n[done; cost = " << cost(x,y,sample_size) << "]" << endl;
        }

        void train(double ** x, double ** y, int sample_size, int batch_size, double LR, int epochs){ // trains network, using rank-3 input
            if(num_of_conv != 0){
                cerr << "Error: input to conv layer must be a rank-3 array.\n";
                assert(false);
            }
            if(sample_size % batch_size !=0) cout << "warning, batches don't fit!\n";
            cout << "[training...]"; cout.flush();
            double beta1, beta2;
            beta1 = adam_b1, beta2 = adam_b2;
            set_mv_to_zero();

            for(int ep=0;ep<epochs;ep++){
                //cout << " " << cost(x,y,sample_size) << ",";
                //cout.flush();
                for(int t=0;t<sample_size/batch_size;t++){
                    set_j_to_zero();
                    for(int u=0;u<batch_size;u++){
                        forward(x[t*batch_size+u]);
                        update_derivatives(x[t*batch_size+u],y[t*batch_size+u],batch_size*output_size_dense);
                    }
                    run_adam(beta1,beta2,LR);
                }
            }
            cout << "\n[done; cost = " << cost(x,y,sample_size) << "]\n";
        }

        void print_to_file(string filename = ""){ // prints weights and biases to textfile
            string name_of_file;
            if(filename == "") name_of_file = "nn_data_"+to_string(input_size_d)+"_"+to_string(input_size_v)+"_"+to_string(input_size_h)+"_"+to_string(output_size_dense)+".txt";
            else name_of_file = filename;
            ofstream myfile;
            myfile.open(name_of_file);

            for(int l=0;l<num_of_conv;l++){
                for(int i=0;i<arch[l].out_d;i++){
                    for(int a=0;a<arch[l].inp_d;a++){
                        for(int b=0;b<arch[l].f_v;b++){
                            for(int c=0;c<arch[l].f_h;c++){
                                myfile << arch[l].filter[i][a][b][c] << ",\n";
                            }
                        }
                    }
                    myfile << arch[l].biases[i] << ",\n";
                }
            }
            for(int l=num_of_conv;l<depth;l++){
                for(int alpha=0;alpha<arch[l].n_out;alpha++){
                    for(int beta=0;beta<arch[l].n_in;beta++){
                        myfile << arch[l].weights[alpha][beta] << ",\n";
                    }
                    myfile << arch[l].biases[alpha] << ",\n";
                }
            }
            myfile.close();
        }
        void read_from_file(string filename = ""){ // loads weights and biases from textfile
            string name_of_file, str;
            if(filename == "") name_of_file = "nn_data_"+to_string(input_size_d)+"_"+to_string(input_size_v)+"_"+to_string(input_size_h)+"_"+to_string(output_size_dense)+".txt";
            else name_of_file = filename;
            ifstream myfile;
            myfile.open(name_of_file);

            for(int l=0;l<num_of_conv;l++){
                for(int i=0;i<arch[l].out_d;i++){
                    for(int a=0;a<arch[l].inp_d;a++){
                        for(int b=0;b<arch[l].f_v;b++){
                            for(int c=0;c<arch[l].f_h;c++){
                                getline(myfile, str, ',');
                                arch[l].filter[i][a][b][c]  = stod(str);
                            }
                        }
                    }
                    getline(myfile, str, ',');
                    arch[l].biases[i] = stod(str);
                }
            }
            for(int l=num_of_conv;l<depth;l++){
                for(int alpha=0;alpha<arch[l].n_out;alpha++){
                    for(int beta=0;beta<arch[l].n_in;beta++){
                        getline(myfile, str, ',');
                        arch[l].weights[alpha][beta] = stod(str);
                    }
                    getline(myfile, str, ',');
                    arch[l].biases[alpha] = stod(str);
                }
            }
            myfile.close();
        }

        friend ostream& operator<<(ostream& os, const network& netw){ //prints network details to stream
            os << "==========================================================================\n";
            int n_param = 0;
            if(netw.num_of_conv > 0){
                for(int l=0;l<netw.num_of_conv;l++) n_param += netw.arch[l].out_d + netw.arch[l].out_d*netw.arch[l].inp_d*netw.arch[l].f_v*netw.arch[l].f_h;
                os << "Convolutional layers: [Input size: " << netw.input_size_d << "x" << netw.input_size_v << "x" <<netw.input_size_h << ", Output size: " << netw.output_size_d << "x" << netw.output_size_v << "x" <<netw.output_size_h;
                os << ", Number of parameters: " << n_param <<"]\n";
                for(int i=0;i<netw.num_of_conv;i++){
                    switch(i){
                    case 0: os << "first layer:\n"; break;
                    case 1: os << "second layer:\n"; break;
                    case 2: os << "third layer:\n"; break;
                    default: os << i+1 << "th layer:\n";
                    }
                    if(netw.arch[i].activ.name.size() > 0) os << "activation: " << netw.arch[i].activ.name << endl;
                    os << netw.arch[i] << (i<netw.num_of_conv-1 ? "\n\n" : "");
                }
                os << "--------------------------------------------------------------------------\n";
            }
            n_param = 0;
            for(int i=netw.num_of_conv;i<netw.depth;i++) n_param += netw.arch[i].n_out + netw.arch[i].n_in*netw.arch[i].n_out;
            os << "Dense layers: [Input size: " << netw.input_size_dense << ", Output size: " << netw.output_size_dense;
            os << ", Number of parameters: " << n_param << "]\n";
            for(int i=netw.num_of_conv;i<netw.depth;i++){
                switch(i-netw.num_of_conv){
                case 0: os << "first layer:\n"; break;
                case 1: os << "second layer:\n"; break;
                case 2: os << "third layer:\n"; break;
                default: os << i+1 << "th layer:\n";
                }
                if(netw.arch[i].activ.name.size() > 0) os << "activation: " << netw.arch[i].activ.name << endl;
                os << netw.arch[i] << (i<netw.depth-1 ? "\n" : "");
            }
            if(netw.softmaxQ) os << "\nFinal layer: Soft-max.\n";

            os << "==========================================================================\n";
            return os;
        }

    private:
        bool softmaxQ = false; // is there a softmax layer or not
        int num_of_conv = 0; // number of conv layers
        double * flatten; // flatten rank-3 into linear array
        double ** Xi_d; // d(cost)/d(bias), for dense layers, for a single sample
        double **** Xi_c; // d(cost)/d(bias), for conv layers, for a single sample

        double ***** jf_c; // d(cost)/d(filter) for conv
        double ***** mf_c; // for momentum
        double ***** vf_c; // for adam

        double ** jb_c; // d(cost)/d(bias) for conv
        double ** mb_c; // for momentum
        double ** vb_c; // for adam

        double *** jw_d; // d(cost)/d(weight) for dense
        double *** mw_d; // for momentum
        double *** vw_d; // for adam

        double ** jb_d; // d(cost)/d(bias) for dense
        double ** mb_d; // for momentum
        double ** vb_d; // for adam

        void update_derivatives(double * x, double * y, double scale){
            double aux, tot;
            if(softmaxQ){
                tot = 0;
                for(int alpha=0;alpha<output_size_dense;alpha++) tot += arch[depth].p_d[alpha]*loss_fnc.df(arch[depth].p_d[alpha],y[alpha]);
                for(int beta=0;beta<output_size_dense;beta++) Xi_d[depth-1][beta] = arch[depth].p_d[beta]*arch[depth-1].q_d[beta]*(loss_fnc.df(arch[depth].p_d[beta],y[beta])-tot);
            }
            else for(int beta=0;beta<output_size_dense;beta++) Xi_d[depth-1][beta] = arch[depth-1].q_d[beta]*loss_fnc.df(arch[depth-1].p_d[beta],y[beta]);

            for(int l=depth-2;l>=0;l--){
                for(int beta=0;beta<arch[l].n_out;beta++){
                    aux = 0;
                    for(int gamma=0;gamma<arch[l+1].n_out;gamma++) aux += arch[l+1].weights[gamma][beta]*Xi_d[l+1][gamma];
                    Xi_d[l][beta] = arch[l].q_d[beta]*aux;
                }
            }
            for(int l=0;l<depth;l++) for(int i=0;i<arch[l].n_out;i++) jb_d[l][i] += Xi_d[l][i]/scale;
            for(int l=0;l<depth;l++) for(int i=0;i<arch[l].n_out;i++) for(int j=0;j<arch[l].n_in;j++) jw_d[l][i][j] += Xi_d[l][i]*( (l==0) ? x[j] : arch[l-1].p_d[j] )/scale;

        }
        void update_derivatives(double *** x, double * y, double scale){
            double aux, tot;
            if(softmaxQ){
                tot = 0;
                for(int alpha=0;alpha<output_size_dense;alpha++) tot += arch[depth].p_d[alpha]*loss_fnc.df(arch[depth].p_d[alpha],y[alpha]);
                for(int beta=0;beta<output_size_dense;beta++) Xi_d[depth-num_of_conv-1][beta] = arch[depth].p_d[beta]*arch[depth-1].q_d[beta]*(loss_fnc.df(arch[depth].p_d[beta],y[beta])-tot);
            }
            else for(int beta=0;beta<output_size_dense;beta++) Xi_d[depth-num_of_conv-1][beta] = arch[depth-1].q_d[beta]*loss_fnc.df(arch[depth-1].p_d[beta],y[beta]);

            for(int l=depth-num_of_conv-2;l>=0;l--){
                for(int beta=0;beta<arch[l+num_of_conv].n_out;beta++){
                    aux = 0;
                    for(int gamma=0;gamma<arch[l+num_of_conv+1].n_out;gamma++) aux += arch[l+num_of_conv+1].weights[gamma][beta]*Xi_d[l+1][gamma];
                    Xi_d[l][beta] = arch[l+num_of_conv].q_d[beta]*aux;
                }
            }
            for(int i=0;i<output_size_d;i++){
                for(int j=0;j<output_size_v;j++){
                    for(int k=0;k<output_size_h;k++){
                        aux = 0;
                        for(int beta=0;beta<arch[num_of_conv].n_out;beta++) aux += Xi_d[0][beta]*arch[num_of_conv].weights[beta][output_size_v*output_size_h*i+output_size_h*j+k];
                        Xi_c[num_of_conv-1][i][j][k] = aux*arch[num_of_conv-1].q_c[i][j][k];
                    }
                }
            }

            int minbb, maxbb, mincc, maxcc;
            for(int l=num_of_conv-2;l>=0;l--){
                    for(int j=0;j<arch[l].out_v;j++){
                        minbb = max(0,(j-arch[l+1].f_v+arch[l+1].stride_v)/arch[l+1].stride_v);
                        maxbb = min(j/arch[l+1].stride_v,arch[l+1].out_v-1);
                        for(int k=0;k<arch[l].out_h;k++){
                            mincc = max(0,(k-arch[l+1].f_h+arch[l+1].stride_h)/arch[l+1].stride_h);
                            maxcc = min(k/arch[l+1].stride_h,arch[l+1].out_h-1);
                            for(int i=0;i<arch[l].out_d;i++){
                                aux = 0;
                                for(int aa=0;aa<arch[l+1].out_d;aa++)
                                    for(int bb=minbb;bb<=maxbb;bb++)
                                        for(int cc=mincc;cc<=maxcc;cc++)
                                            aux += Xi_c[l+1][aa][bb][cc]*arch[l+1].filter[aa][i][j-bb*arch[l+1].stride_v][k-cc*arch[l+1].stride_h];
                                Xi_c[l][i][j][k] = aux*arch[l].q_c[i][j][k];
                            }
                        }
                    }
                }



            for(int l=0;l<depth-num_of_conv;l++) for(int i=0;i<arch[l+num_of_conv].n_out;i++) jb_d[l][i] += Xi_d[l][i]/scale;
            for(int l=0;l<depth-num_of_conv;l++) for(int i=0;i<arch[l+num_of_conv].n_out;i++) for(int j=0;j<arch[l+num_of_conv].n_in;j++) jw_d[l][i][j] += Xi_d[l][i]*( (l==0) ? flatten[j] : arch[l+num_of_conv-1].p_d[j] )/scale;

            for(int l=0;l<num_of_conv;l++) for(int i=0;i<arch[l].out_d;i++) for(int j=0;j<arch[l].out_v;j++) for(int k=0;k<arch[l].out_h;k++) jb_c[l][i] += Xi_c[l][i][j][k]/scale;
            for(int l=0;l<num_of_conv;l++)
                for(int i=0;i<arch[l].out_d;i++)
                    for(int j=0;j<arch[l].out_v;j++)
                        for(int k=0;k<arch[l].out_h;k++)
                            for(int a=0;a<arch[l].inp_d;a++)
                                for(int b=0;b<arch[l].f_v;b++) for(int c=0;c<arch[l].f_h;c++) jf_c[l][i][a][b][c] += Xi_c[l][i][j][k]*(l==0 ? x[a][b+j*arch[l].stride_v][c+k*arch[l].stride_h] : arch[l-1].p_c[a][b+j*arch[l].stride_v][c+k*arch[l].stride_h] )/scale;

        }

        void set_mv_to_zero(){ // sets Adam parameters m_t and v_t to zero, for all layers
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) for(int j=0;j<arch[l].f_v;j++) for(int k=0;k<arch[l].f_h;k++) mf_c[l][a][i][j][k] = 0;
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) for(int j=0;j<arch[l].f_v;j++) for(int k=0;k<arch[l].f_h;k++) vf_c[l][a][i][j][k] = 0;
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) mb_c[l][a] = 0;
            for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) vb_c[l][a] = 0;
            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) for(int k=0;k<arch[i+num_of_conv].n_in;k++) mw_d[i][j][k] = 0;
            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) for(int k=0;k<arch[i+num_of_conv].n_in;k++) vw_d[i][j][k] = 0;
            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) mb_d[i][j] = 0;
            for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) vb_d[i][j] = 0;
        }
        void set_j_to_zero(){ // sets all d(cost)/d(parameter) to zero
                for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) for(int i=0;i<arch[l].inp_d;i++) for(int j=0;j<arch[l].f_v;j++) for(int k=0;k<arch[l].f_h;k++) jf_c[l][a][i][j][k] = 0;
                for(int l=0;l<num_of_conv;l++) for(int a=0;a<arch[l].out_d;a++) jb_c[l][a] = 0;
                for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) for(int k=0;k<arch[i+num_of_conv].n_in;k++) jw_d[i][j][k] = 0;
                for(int i=0;i<depth-num_of_conv;i++) for(int j=0;j<arch[i+num_of_conv].n_out;j++) jb_d[i][j] = 0;
        }
        void run_adam(double & beta1, double & beta2, double LR){ // runs Adam for a single iteration
            for(int l=0;l<num_of_conv;l++){
                for(int a=0;a<arch[l].out_d;a++){
                    for(int i=0;i<arch[l].inp_d;i++){
                        for(int j=0;j<arch[l].f_v;j++){
                            for(int k=0;k<arch[l].f_h;k++){
                                mf_c[l][a][i][j][k] = adam_b1*mf_c[l][a][i][j][k]+(1-adam_b1)*jf_c[l][a][i][j][k];
                                vf_c[l][a][i][j][k] = adam_b2*vf_c[l][a][i][j][k]+(1-adam_b2)*jf_c[l][a][i][j][k]*jf_c[l][a][i][j][k];
                                arch[l].filter[a][i][j][k] -= LR*(mf_c[l][a][i][j][k]/(1-beta1))/(sqrt((vf_c[l][a][i][j][k])/(1-beta2))+.000000001);;
                            }
                        }
                    }
                    mb_c[l][a] = adam_b1*mb_c[l][a]+(1-adam_b1)*jb_c[l][a];
                    vb_c[l][a] = adam_b2*vb_c[l][a]+(1-adam_b2)*jb_c[l][a]*jb_c[l][a];
                    arch[l].biases[a] -= LR*(mb_c[l][a]/(1-beta1))/(sqrt((vb_c[l][a])/(1-beta2))+.000000001);;
                }
            }
            for(int l=0;l<depth-num_of_conv;l++){
                for(int b=0;b<arch[l+num_of_conv].n_out;b++){
                    for(int c=0;c<arch[l+num_of_conv].n_in;c++){
                        mw_d[l][b][c] = adam_b1*mw_d[l][b][c]+(1-adam_b1)*jw_d[l][b][c];
                        vw_d[l][b][c] = adam_b2*vw_d[l][b][c]+(1-adam_b2)*jw_d[l][b][c]*jw_d[l][b][c];
                        arch[l+num_of_conv].weights[b][c] -= LR*(mw_d[l][b][c]/(1-beta1))/(sqrt((vw_d[l][b][c])/(1-beta2))+.000000001);
                    }
                    mb_d[l][b] = adam_b1*mb_d[l][b]+(1-adam_b1)*jb_d[l][b];
                    vb_d[l][b] = adam_b2*vb_d[l][b]+(1-adam_b2)*jb_d[l][b]*jb_d[l][b];
                    arch[l+num_of_conv].biases[b] -= LR*(mb_d[l][b]/(1-beta1))/(sqrt((vb_d[l][b])/(1-beta2))+.000000001);
                }
            }
            beta1 *= adam_b1; beta2 *= adam_b2;

            // for(int l=0;l<depth;l++){
            //     for(int b=0;b<arch[l].n_out;b++){
            //         for(int c=0;c<arch[l].n_in;c++){
            //             //mw_d[l][b][c] = 0.9*mw_d[l][b][c]+0.1*jw_d[l][b][c];
            //             arch[l].weights[b][c] -= LR*jw_d[l][b][c];
            //         }
            //         //mb_d[l][b] = 0.9*mb_d[l][b]+0.1*jb_d[l][b];
            //         arch[l].biases[b] -= LR*jb_d[l][b];
            //     }
            // }


        }

        void print_size(){ // prints input/output size to `cout`, as well as number of trainable parameters
            int n_param = 0;
            if(num_of_conv > 0){
                for(int l=0;l<num_of_conv;l++) n_param += arch[l].out_d + arch[l].out_d*arch[l].inp_d*arch[l].f_v*arch[l].f_h;
                cout << "Convolutional layers: [Input size: " << input_size_d << "x" << input_size_v << "x" << input_size_h << ", Output size: " << output_size_d << "x" << output_size_v << "x" << output_size_h;
                cout << ", Number of parameters: " << n_param <<"]\n";
            }
            n_param = 0;
            for(int i=num_of_conv;i<depth;i++) n_param += arch[i].n_out + arch[i].n_in*arch[i].n_out;
            cout << "Dense layers: [Input size: " << input_size_dense << ", Output size: " << output_size_dense;
            cout << ", Number of parameters: " << n_param << "]\n";
        }

};