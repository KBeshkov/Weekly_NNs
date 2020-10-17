'''Article 1: Introduction to Neural Nets'''
import numpy as np
import matplotlib as pyplot
from sklearn.decomposition import PCA


#Start by defining the network model
#We will use a simple integrate and fire model
#Don't worry too much about the specifics of the model
def IF_model(W,T,I,thresh=0,eq=-60,reset=-70,mu=0.1,epsp=1):
    X = np.zeros([len(W),T])
    X[:,0] = eq
    spikes = np.zeros([len(W),T])
    for t in range(T-1):
        dx = eq-X[:,t]+np.dot(W,epsp*spikes[:,t])+I[:,t]
        X[:,t+1] = X[:,t]+mu*dx
        spikes[np.where(X[:,t+1]>thresh),t+1] = 1
        X[np.where(X[:,t+1]>thresh),t+1] = reset
    return [spikes,X]

#define the paramters of the network
N = 100 #number of neurons
T = 500 #number of time points

#use constant input with some variance
Inputs = 60+10*np.random.randn(N,T)

#sample the weights from a normal distribution with 0 mean and variance 10
#and make the matrix diagonal 0
Weights = 10*(np.random.randn(N,N))
np.fill_diagonal(Weights,0)

#Run the model and store both spikes and activity values
x = IF_model(Weights,T,Inputs)

#calculate some statistics eg. firing rate
fr = 10*np.sum(x[0],1)/T #multipy by 1 over the integration constant mu=0.1

#perform dimensionality reduction with PCA to show low dimensional embeding of activity
pca = PCA(n_components=2)
x_reduced = pca.fit_transform(x[1].T)

#Activity plot
plt.figure(dpi=120)
plt.subplot(2,1,1)
plt.imshow(-x[0],'gray') #raster plot of the spikes
plt.title('Spiking raster plot')
plt.axis('off')
plt.subplot(2,1,2)
plt.imshow(x[1])
plt.title('Activity plot')
plt.xlabel('Time (t)')
plt.yticks([])

#dimensionality reduced activity
plt.figure(dpi=120)
plt.plot(x_reduced[:,0],x_reduced[:,1],'.')
plt.title('Low dimensional embedding of activity')

#Weight matrix plot
plt.figure(dpi=120)
plt.imshow(Weights)
plt.colorbar()
plt.title('The weight matrix')

#Distributions plot
plt.figure(dpi=120)
plt.subplot(1,2,1)
plt.hist(fr,20,density=True)
plt.ylabel('Density')
plt.title('Distribution of firing rates')
plt.subplot(1,2,2)
plt.hist(x[1][x[1]!=-70].flatten(),20,density=True)
plt.title('Distribution of voltage values')
plt.tight_layout()
