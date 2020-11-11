import sys
import argparse
from scipy import io
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train0', '-t0' , dest = "train_file0", required = True, help="Training File 0") 
    parser.add_argument('-train1', '-t1' , dest = "train_file1", required = True, help="Training File 1") 
    parser.add_argument('-test0' , '-te0', dest = "test_file0" , required = True, help="Testing File 0") 
    parser.add_argument('-test1' , '-te1', dest = "test_file1" , required = True, help="Testing File 1")

    parser.add_argument('-print' , '-p'  , dest = "print_output" , action = 'store_true', required = False, help="If you want to print intermediate results")
    parser.add_argument('-plot'  , '-pl' , dest = "plot_pca_comp", action = 'store_true', required = False, help="If you want to print intermediate results")

    return parser.parse_args()


def load_mat_data(options):
    trc0 = io.loadmat(options.train_file0)
    trc1 = io.loadmat(options.train_file1)

    tec0 = io.loadmat(options.test_file0)
    tec1 = io.loadmat(options.test_file1)

    return trc0, trc1, tec0, tec1


def preprocessing(class0, class1):
    data0 = np.reshape(class0['nim0'], (784,-1))
    data0 = np.transpose(data0)
    df0 = pd.DataFrame(data0)
    df0_label = [0]*df0.shape[0]

    data1 = np.reshape(class1['nim1'], (784,-1))
    data1 = np.transpose(data1)
    df1 = pd.DataFrame(data1)
    df1_label = [1]*df1.shape[0]

    data = pd.concat([df0,df1], axis=0)
    label = df0_label + df1_label

    return data, label


def compute_eigen_vectors(training):
    covariance_mtx =  np.array(training.cov())
    eig_values, evects = la.eig(covariance_mtx)
    eig_pairs = [ (np.abs(eig_values[i]), evects[:,i]) for i in range(len(eig_values))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_vectors = np.hstack((eig_pairs[0][1].reshape(784,1), eig_pairs[1][1].reshape(784,1)))

    return covariance_mtx, eig_values, eig_vectors


def plotting(data, labels, fig_name):
    df_compressed = pd.DataFrame(data)

    label_color = []
    colors = {0:'orange', 1:'blue'}         
    for label in labels:
        label_color.append(colors[label])
                
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.75, 0.75])
    axes.scatter(df_compressed[0], df_compressed[1], c=label_color)

    axes.set_title("PCA Analysis")
    axes.set_xlabel("PCA Component 1")
    axes.set_ylabel("PCA Component 2")

    h = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in colors.keys()]
    plt.legend(h, colors.keys())

    fig.savefig(fig_name, dpi = 1000)


def calc_accuracy(data, label, classes, means, covariances, priors):
    samples, feats = data.shape
    
    result = []
    for i in range(samples):
        likelihoods = []
        for c in classes:
            tmp = multivariate_normal.pdf(data[i], mean=means[c], cov=covariances[c])
            likelihoods.append(tmp*priors[c])
    
        marginal = np.sum(likelihoods)
    
        probs = []
        for i, c in enumerate(classes):
            probs.append(likelihoods[i] / marginal)

        result.append(classes[np.argmax(probs)])
    
    return accuracy_score(label, result)


def main():
    options = parse_arguments()

    #Loading the files...
    train_class0, train_class1, test_class0, test_class1 = load_mat_data(options)
    
    #Task - 1: Feature normalization.
    train, label1 = preprocessing(train_class0, train_class1)
    test,  label2 = preprocessing(test_class0, test_class1)

    mean = train.mean()
    sigma = train.std()
    
    training = (train - mean)/sigma
    testing = (test - mean)/sigma

    if options.print_output:
        print("Normalised training data samples:\n", training) 
        print("Normalised testing data samples:\n" , testing)
 
    #Task - 2: PCA using the training samples.
    covariance_matrix, eigen_values, eigen_vectors = compute_eigen_vectors(training)
   
    if options.print_output: 
        print("Covariance Matrix:\n", pd.DataFrame(covariance_matrix))
        for i in range(4):
            print("Variance on PCA {}: {}".format(i+1, eigen_values[i]))
        print("Eigen Vectors:\n", pd.DataFrame(eigen_vectors))


    #Task - 3: Dimension reduction using PCA
    train_components = np.array(training).dot(eigen_vectors)    
    test_components = np.array(testing).dot(eigen_vectors)

    if options.plot_pca_comp:
        plotting(train_components, label1, 'training_PCA_fig')
        plotting(test_components, label2, 'testing_PCA_fig')
 
   
    n_samples, n_feats = train_components.shape
    y = np.array(label1)
    n_classes = np.unique(y).shape[0]
    classes = np.unique(y)
    
    means = {}
    for i in classes:
        idx = np.argwhere(y == i).flatten()
        m = []
        for j in range(n_feats):
            m.append(np.mean( train_components[idx,j] ))
        means[i] = m
    
    
    covariances = {}
    for i in classes:
        idx = np.argwhere(y==i).flatten()
        covariances[i] = np.cov(train_components[idx,:].T)

    if options.print_output:
        print("\nMeans for both classes:\n", pd.DataFrame(means))
        print("\nCo-variance Matrix for class 0:\n", pd.DataFrame(covariances[0]))
        print("\nCo-variance Matrix for class 1:\n", pd.DataFrame(covariances[1]))


    priors = {c: 0.5 for c in classes}

    training_accuracy = calc_accuracy(train_components, label1, classes, means, covariances, priors)
    testing_accuracy  = calc_accuracy(test_components, label2, classes, means, covariances, priors)

    print("\nAccuracy on Training Data: {}%".format(round(training_accuracy*100, 4)))
    print("Accuracy on Testing Data: {}%".format(round(testing_accuracy*100, 4)))


if __name__ == '__main__':
    main()
