Command to find the accuracy of the MNIST data (data is only the subset of MNIST, containing only 0 and 1) :

python project.py -train0 data/training0.mat -train1 data/training1.mat -test0 data/testing0.mat -test1 data/testing1.mat

Additional arguments:
  -print, -p            If you want to print intermediate results
  -plot, -pl            If you want to plot the class distribution of data
                        with pca components
                        
You can check all arguments and the definitions by using:
python project.py --help


Command using all arguments:
python project.py -train0 data/training0.mat -train1 data/training1.mat -test0 data/testing0.mat -test1 data/testing1.mat -print -plot
