# Resume Screening Next Word Predictor

## Model Summary: Abstract

This project involves creating a resume screening next-word predictor, similar to smart compose features used in emails, LinkedIn, and code completions, which predict the next word based on previous data in a sentence.

## Table of Contents

1. [Introduction](#introduction)
2. [Model Preprocessing](#model-preprocessing)
   - [Dataset Conversion for Supervised Learning](#dataset-conversion-for-supervised-learning)
   - [Token to Number Conversion Using One-Hot Encoding](#token-to-number-conversion-using-one-hot-encoding)
   - [Zero Padding for Uniform Sentence Size](#zero-padding-for-uniform-sentence-size)
3. [Architecture, Strategy, and Evaluation](#architecture-strategy-and-evaluation)
   - [Architecture](#architecture)
   - [Evaluation](#evaluation)
4. [Performance Improvement and Conclusion](#performance-improvement-and-conclusion)
   - [Use More Data](#use-more-data)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Advanced Architectures](#advanced-architectures)
5. [Conclusion](#conclusion)

## Introduction

Creating a resume screening next-word predictor is similar to smart compose features used in emails, LinkedIn, and code completions, which predict the next word based on previous data in a sentence. While there are numerous datasets available on platforms like Kaggle, such as famous quotes, jokes, idioms, pronouns, and stories, training on these datasets can be time-consuming. To optimize this process, I created a custom dataset that contains sentences specifically from resumes. This approach not only reduces the training time but also ensures a considerable level of accuracy suitable for resume screening applications.

## Model Preprocessing

### Dataset Conversion for Supervised Learning

- Convert sentences into a supervised learning format where the input matches the expected output. This is done by tokenizing words using TensorFlow's tokenizer class.

### Token to Number Conversion Using One-Hot Encoding

- Convert tokens into numerical data through one-hot encoding, making it suitable for LSTM model training.

### Zero Padding for Uniform Sentence Size

- Apply zero padding to ensure all sequences are of the same length, matching the maximum sentence size in the dataset.

## Architecture, Strategy, and Evaluation

### Architecture

- **Embedding Layer**: Converts sparse vectors to dense vectors for easier processing.
- **LSTM Layer**: Handles sequential data, capturing dependencies over long sequences.
- **Dense Layer with Softmax Activation**: Converts output into one-hot encoded vectors, providing probabilistic predictions.

### Evaluation

- Predicts the next 5 words for each input word in the dataset. If a word is not present, the model predicts the closest possible word.

## Performance Improvement and Conclusion

### Use More Data

- Increase the dataset size to avoid overfitting and improve generalization.

### Hyperparameter Tuning

- Adjust parameters like epochs, optimizers, and activation functions to enhance performance.

### Advanced Architectures

- Implement advanced architectures such as stacked LSTM, bidirectional LSTM, or transformers like BERT for better results. However, the current model provides satisfactory outcomes.

## Conclusion

Developing a resume screening next-word predictor demonstrates the practical application of machine learning. The approach using embedding layers, LSTM, and dense layers with softmax activation offers a robust framework for word prediction. Future improvements with larger datasets and advanced architectures can further enhance its accuracy and utility, showcasing the potential of tailored models in specific contexts.

---

### How to Use

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the training script with your dataset.
4. Use the model to predict the next words in resume sentences.

### Contributions

Feel free to fork this repository and make contributions. Pull requests are welcome.

### License

This project is licensed under the MIT License.
