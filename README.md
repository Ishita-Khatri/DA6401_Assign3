# Model Training and Evaluation

This repository contains the implementation of a sequence to sequence RNN model with different cell types. that is trained with the AdamW optimizer, enhanced with gradient clipping and beam search for improved performance. The model uses two types of accuracy (Token Accuracy and Sequence Accuracy) to evaluate its performance during training.

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
After training for 20 epochs, the following accuracies were achieved:

### Validation Accuracy:
- **Token Accuracy**: 70.08%
- **Sequence Accuracy**: 35.08%

### Test Accuracy:
- **Token Accuracy**: 71.33%
- **Sequence Accuracy**: 35.41%

## Usage:
1. **Clone this repository**:
    ```bash
    git clone https://github.com/yourusername/repository-name.git
    cd repository-name
    ```

2. **Install Dependencies**:
    Make sure you have Python and pip installed. Then, install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. **Training the Model**:
    To train the model, simply run:
    ```bash
    python train.py
    ```

4. **Evaluation**:
    After training, you can evaluate the model on the test set using:
    ```bash
    python evaluate.py
    ```

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
