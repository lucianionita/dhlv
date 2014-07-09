
def show():
        print"""
        
        Todo list:
    
 +  Make batch learning with decreasing LR more friendly/clear
    Add fancy tanh + lin activation
    Find a better learning algorithm (rprop?)
    Implement RMSprop
    Momentum for training
  ? Maxout
    DropConnect
    Add autoencoders
    Reproducible random seed
    Full test suite for digits and faces
    Add deformation for inputs    
    Add 1-2 other datasets
    Dropout entire conv filter
    Preprocessing (normalization/equalizationa)
    Multiple inputs to layers
    Histogram layer for the filters
    Weight shrinking ( W := W / (sum(W)) )
*   Add input noise
*   Add convnet models
*   Validation based on minibatches
*   Simple YaML-like specification of structure
*   SVD-like Hidden Layer
*   2 Layer MLP
*   Make the MLP not do dropout on validation
*   Make MLP's LR a shared variable
*   Convolutional Layer
*   Multilayer Network with Dropout
*   Training algorithm rewrite
*   Make the voo module that nicely integrates everything
*   DropConnect Hidden Layer
*   Adding a reset/update function to the dropout layer
*   Move everything into voo
*   Move training into voo
*   Move misc functions into voo
*   Move todo into voo too
  D Binarization of layer activations, lateral inhibitions
  D Mean downsampling
  D Max downsampling
  D Stochastic downsampling
  D BoW model for filters 
 x? Maybe add a "Config" variable that works like Param from ProtoMV
 x? Randomize MLP better
 x? Experiment runner class
  ? Dimmensionality reduction layer (isn't that a general hidden layer)
  ? Better training algosirthms
  ? RBF layer
    Dropoiut based on circles of the input data (for occlusion)
  ? patch-level autoencoder
  ? move the training functions and update stuff to the training module/function
 +  Add options for layer
 +  Add option for reset/randomize
    Add option for different params to update/regulate
    Add verbose option for model generation and training
    Data recording from training
  e Boosting inputs
  e Reweighing inputs to simulate different distributions
  e Training with random output, to see if this unsupervised method learns 
                         any good filters or at least is a good pretrainer
    
            
Meanings:
    
*   Done
 +  In progress
  ? Should review
  x Won't do
 x? Should decide if it's worth doing
  e It's an experiment rather than implementation
  D Dependent on some other thing to be implemented/installed
  
        """
