# README #

This repository lists the files that were used to develop a spiking neural network for supervsied learning 
based application for handwritten digit classification form the MNIST data-set. The network is trained in a manner similar
to stochastic gradient descent, where the weights are updated at the end of every presentation of the image.

The neurons used in the SNN are simple leaky-integrate-and-fire neurons.

The supervised SNN training algorithm of NormAD is described in this paper:
N. Anwani and B. Rajendran, "NormAD - Normalized Approximate Descent based supervised learning rule for spiking neurons," 2015 
International Joint Conference on Neural Networks (IJCNN), Killarney, 2015, pp. 1-8.

The CUDA implementation of the three layered SNN using the NormAD algorithm is described in this paper. Please cite the following if you are using our codes in your work.

S. R. Kulkarni, J. M. Alexiades and B. Rajendran, "Learning and real-time classification of hand-written digits with spiking neural networks," 2017 24th IEEE International Conference on Electronics, Circuits and Systems (ICECS), Batumi, 2017, pp. 128-131.
doi: 10.1109/ICECS.2017.8292015
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8292015&isnumber=8291994

Arxiv link at:
S. R. Kulkarni, J. Alexiades and B. Rajendran, "Learning and Real-time Classification of Hand-written Digits With Spiking Neural Networks",
arXiv preprint arXiv:1711.03637, 2017.

Detailed explanation of the different optimization studies can be found here:
Kulkarni, S. R., & Rajendran, B. (2018). Spiking neural networks for handwritten digit recognition—Supervised learning and network optimization, Neural Networks, Volume 103, 2018, Pages 118-127, ISSN 0893-6080,
https://doi.org/10.1016/j.neunet.2018.03.019.
(http://www.sciencedirect.com/science/article/pii/S0893608018301126)

A live interactive demo of this work was also implemented. We made use of OpenCV to preprocess the captured image and display the spike maps for visualization. The description is available here:
S. R. Kulkarni, J. M. Alexiades and B. Rajendran, "Live Demonstration: Image Classification Using Bio-inspired Spiking Neural Networks," 2018 IEEE International Symposium on Circuits and Systems (ISCAS), Florence, Italy, 2018, pp. 1-1.
doi: 10.1109/ISCAS.2018.8351810
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8351810&isnumber=8350884

### What is this repository for? ###

* Quick summary
This repository consists of the following files:

1. snn_train.cu - CUDA code for network simulation and NormAD based training
Takes in input from mnist_trn.csv (reduced training set for hyperparameter
optimization) or from mnist_train.csv (for full 60k training)

This file reads in images from mnist_train.csv.
This line has to be accordingly changed in the code depending on where the file is located.
The corresponding file handle is F_train.

2. snn_test.cu - CUDA code for network's inference on the test data set.
Performs only the forward pass
Takes input from mnist_test.csv (test set).
This line has to be changed in the code, depending on where the file is placed in the users' directory.
The corresponding file handle is F_test.

3. kernels_3x3.csv - The convolution kernel weights
These values are scaled by 5 nS so that the current to the neuron lies in the
range of few nA

4. pixel_spks.csv - Spike trains for layer 1 neurons
The file has 256 x 1000 sized binary array created from a matlab file.

5. lif_pixel_spike_converter.m: For generating layer 1 neuron spikes in response to 
currents coming in from image pixels. The spikes of the 1st layer correspond to 
outputs of LIF neurons being activated by spikes from image pixels 0 to 255 and are stored in the file
pixel_spks.csv


### How do I get set up? ###

* Summary of set up
The CUDA codes for training and testing the network train using 60k images from the MNIST training set 
and the network's accuracy is reported on the performance on the MNIST test set with 10k images.

Each of the CUDA file takes in the starting index of the image from the respective datasets as a runtime parameter.

The MNIST files need to be provded in .csv format.
They are available here:
https://pjreddie.com/projects/mnist-in-csv/

which is the same files provided in the original MNIST dataset (found at http://yann.lecun.com/exdb/mnist/)

The images present in the above files are vectorized per row. Hence, there are 10,000 rows each having 1st element as the the label of the image followed by 784 pixels.

* Codes - compilation
To compile the CUDA code:
nvcc snn_test.cu -o runtest

Similarly the code to train the network can be compiled and run.

Additionally depeding on the GPU architecture you are using you can give the option of -arch=sm_37, or any different number based on the compute capability of your GPU.

* Running the simulations:
Here, runtest is the executable and can be runs as:
./runtest <starting_img_index>

<starting_img_index> is a number between 0 and 10000, which indicates the starting index from the MNIST 
dataset.
Currently, for the test code, we need to change the weight file at the end of every epoch


The traininig code prints out the index of the image every epoch, just to monitor the progress.
The test code prints out the result of classification based on the count or the correlation metric
at every image presentation. The ouput can be dumped to a log file: ./runtest <starting_img_index> > output.log &

* Configuration
In the beginning of each of the CUDA files are a set of #defines, where one can change the duration of simulation per image, 
the scaling factor, learning rate, etc.

### Who do I talk to? ###

* For any queries: 
Please send an email to - srk68@njit.edu
