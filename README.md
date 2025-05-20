This repository contains the implementation of a sequence-to-sequence RNN model with different cell types such as vanilla RNN, LSTM, and GRU. The task is to transliterate Latin sequences to Devanagari. The data is taken from the Dakshina Dataset by Google. 

Firstly, the vanilla RNN/LSTM/GRU is run without an attention mechanism, and then an attention mechanism is implemented to evaluate performance. The attention mechanism performs better even with fewer encoder and decoder layers. 

## Key Features:
- **AdamW Optimizer**: We use the AdamW optimizer with a weight decay of 1e-5 to help prevent overfitting.
- **Gradient Clipping**: After calculating the gradients, they are clipped to a maximum value to mitigate the risk of exploding gradients.
- **Beam Search**: During prediction, we implement beam search instead of greedy search to improve the quality of generated sequences.
- **Accuracy Metrics**: The model calculates two types of accuracy:
  - **Token Accuracy**: This measures the number of correctly predicted tokens divided by the total number of tokens in the dataset.
  - **Sequence Accuracy**: This measures the number of correctly predicted sequences divided by the total number of sequences (or examples).

## Training Configuration:
- The model is trained for **20 epochs**.
- After each epoch, both **token accuracy** and **sequence accuracy** are calculated and logged to monitor performance.

## Results:
For the vanilla RNN (without attention), the following test accuracy was achieved after training for 20 epochs:
- **Token Accuracy**: 71.33%
- **Sequence Accuracy**: 35.41%

Whereas, the best attention-based model achieved the following test accuracy:
- **Token Accuracy**: 77.37%
- **Sequence Accuracy**: 44.29%
