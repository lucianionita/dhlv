
"""

Todo list:
    
  ! Training algorithm rewrite
 +  Make the voo module that nicely integrates everything
 +  Make batch learning with decreasing LR more friendly/clear
  ? Randomize MLP better
    Add autoencoders
  ? Maybe add a "Config" variable that works like Param from ProtoMV
    Reproducible random seed
    Find a better learning algorithm (rprop?)
    Momentum for training
    DropConnect
  ? Maxout
    Full test suite for digits and faces
    Add deformation for inputs    
    Add 1-2 other datasets
    Add input noise
*   Add convnet models
*   Validation based on minibatches
*   Simple YaML-like specification of structure
*   SVD-like Hidden Layer
*   2 Layer MLP
*   Make the MLP not do dropout on validation
*   Make MLP's LR a shared variable
*   Convolutional Layer
*   Multilayer Network with Dropout
    DropConnect Hidden Layer
    Preprocessing (normalization/equalizationa)
    Adding a reset/update function to the dropout layer
    Dropout entire conv filter
    Experiment runner class
    BoW model for filters 
    Multiple inputs to layers
    Histogram layer for the filters
  ? Dimmensionality reduction layer (isn't that a general hidden layer)
    Better training algosirthms
    RBF layer?
    implementing different parameters to optimize and to regularize
    Binarization of layer activations, lateral inhibitions
    Dropoiut based on circles of the input data (for occlusion)
  ? patch-level autoencoder
  ? move the training functions and update stuff to the training module/function
 +  Add options for layer
 +  Add option for reset/randomize
    Add option for different params to update/regulate
    Move everything into voo
    Move training into voo
    Move misc functions into voo
    Move todo into voo too
    Add verbose option for model generation and training
    Data recording from training
  e Boosting inputs
  e Reweighing inputs to simulate different distributions
  e Training with random output, to see if this unsupervised method learns 
                         any good filters or at least is a good pretrainer
    Add the fancy tanh activation
    Weight shrinking ( W := W / (sum(W)) )
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
    .
        
    
    
    
    
    
    
    
    
    
    
    
    
    
"""
