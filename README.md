Article
** A Comparison of Different Transformer Models for Time Series Prediction

# Citation: Capoglu, E.U.; Taherkhani,
A. A Comparison of Different
Transformer Models for Time Series
Prediction. Information 2025, 1, 0.

# Abstract
Accurate estimation of the Remaining Useful Life (RUL) of lithium-ion batteries is essential for enhancing the reliability and efficiency of energy storage systems. This study explores custom deep learning models to predict RUL using a dataset from the Hawaii Natural Energy Institute (HNEI). Three approaches are investigated: an Encoder-only Transformer model, its enhancement with SimSiam transfer learning, and a CNN--Encoder hybrid model. These models leverage advanced mechanisms such as multi-head attention, robust feedforward networks, and self-supervised learning to capture complex degradation patterns in the data. Rigorous preprocessing and optimisation ensure optimal performance, reducing key metrics such as mean squared error (MSE) and mean absolute error (MAE). Experimental results demonstrated that Transformer--CNN with Noise Augmentation outperforms other methods, highlighting its potential for battery health monitoring and \mbox{predictive maintenance.

After runing the python code and it explores the hyperparameters, there are JSON files in the bayesian_opt directory for each hyperparameter combination searched for each model. Each model also has a corresponding .py file designed to run the best hyperparameter combination for a final training of exactly 200 epochs.
To observe this final training, the working directory must be set to the Master folder. Otherwise, the tuner will create a new directory and start searching for hyperparameters again.










