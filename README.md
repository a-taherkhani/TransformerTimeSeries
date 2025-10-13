Article
# A Comparison of Different Transformer Models for Time Series Prediction

# For All use of the data, please 'cite' the following:
Capoglu, Emek Utku, and Aboozar Taherkhani. 2025. "A Comparison of Different Transformer Models for Time Series Prediction" Information 16, no. 10: 878. https://doi.org/10.3390/info16100878

# Abstract
Accurate estimation of the Remaining Useful Life (RUL) of lithium-ion batteries is essential for enhancing the reliability and efficiency of energy storage systems. This study explores custom deep learning models to predict RUL using a dataset from the Hawaii Natural Energy Institute (HNEI). Three approaches are investigated: an Encoder-only Transformer model, its enhancement with SimSiam transfer learning, and a CNN--Encoder hybrid model. These models leverage advanced mechanisms such as multi-head attention, robust feedforward networks, and self-supervised learning to capture complex degradation patterns in the data. Rigorous preprocessing and optimisation ensure optimal performance, reducing key metrics such as mean squared error (MSE) and mean absolute error (MAE). Experimental results demonstrated that Transformer--CNN with Noise Augmentation outperforms other methods, highlighting its potential for battery health monitoring and \mbox{predictive maintenance.

# Not:
After runing the python code and it explores the hyperparameters, there are JSON files in the bayesian_opt directory for each hyperparameter combination searched for each model. Each model also has a corresponding .py file designed to run the best hyperparameter combination for a final training of exactly 200 epochs.
To observe this final training, the working directory must be set to the Master folder. Otherwise, the tuner will create a new directory and start searching for hyperparameters again.













