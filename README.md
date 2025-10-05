# MovieLens Recommendation System

A collaborative filtering recommendation system built with PyTorch using Matrix Factorization on the MovieLens dataset.

## Overview

This project implements a neural network-based matrix factorization model to predict movie ratings and generate personalized movie recommendations. The model learns latent features for users and items through embedding layers and optimizes them to minimize prediction error.

## Features

- **Matrix Factorization**: Uses user and item embeddings with bias terms
- **Automatic Data Loading**: Downloads and processes MovieLens ml-latest-small dataset
- **Train/Test Split**: Evaluates model performance on held-out data
- **Top-N Recommendations**: Generates personalized movie recommendations for users
- **Training Visualization**: Plots training loss over epochs

## Requirements

```
torch
numpy
pandas
scikit-learn
matplotlib
requests
```

Install dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib requests
```

## Dataset

The project uses the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/ml-latest-small/) dataset, which contains:
- 100,000+ ratings
- 9,000+ movies
- 600+ users

The dataset is automatically downloaded and extracted when you run the script.

## Model Architecture

The `MatrixFactorization` model consists of:
- **User Embeddings**: Latent factor vectors for each user (default: 50 dimensions)
- **Item Embeddings**: Latent factor vectors for each movie (default: 50 dimensions)
- **User Biases**: Bias term for each user
- **Item Biases**: Bias term for each movie

**Prediction Formula**: 
```
rating = dot_product(user_embedding, item_embedding) + user_bias + item_bias
```

## Training Configuration

- **Embedding Dimension**: 50
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.005, weight_decay=1e-5)
- **Epochs**: 50
- **Train/Test Split**: 80/20

## Usage

### Run the Full Pipeline

```bash
python movielens_recommender.py
```

This will:
1. Download and preprocess the MovieLens dataset
2. Train the matrix factorization model
3. Evaluate performance on test set (RMSE)
4. Display training loss plot
5. Generate top-5 recommendations for users 1-5

### Get Recommendations for a Specific User

```python
# Get top 5 recommendations for user ID 10
recommendations = get_top_n_recommendations(user_id=10, n=5)

for movie_id, title, score in recommendations:
    print(f"{title}: {score:.2f}")
```

## Output Example

```
Epoch 10/50, Loss: 0.8234
Epoch 20/50, Loss: 0.7456
Epoch 30/50, Loss: 0.7123
Epoch 40/50, Loss: 0.6987
Epoch 50/50, Loss: 0.6891

Test RMSE: 0.8745

Top 5 recommendations for user ID 1:
Movie: The Shawshank Redemption (ID: 318), Predicted rating: 4.87
Movie: The Godfather (ID: 858), Predicted rating: 4.76
Movie: Pulp Fiction (ID: 296), Predicted rating: 4.65
...
```

## Model Performance

The model typically achieves:
- **Test RMSE**: ~0.85-0.90 (depending on random seed and hyperparameters)
- Lower RMSE indicates better prediction accuracy

## Customization

### Adjust Hyperparameters

```python
# Change embedding dimension
model = MatrixFactorization(num_users, num_items, num_factors=100)

# Modify learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for more epochs
num_epochs = 100
```

### Use Different Dataset

Replace the dataset URL with other MovieLens datasets:
- ml-latest (full dataset, ~27M ratings)
- ml-25m (25M ratings)

## File Structure

```
.
├── movielens_recommender.py    # Main script
└── README.md                    # This file
```

## How It Works

1. **Data Preprocessing**: User and movie IDs are mapped to continuous indices
2. **Model Training**: Embeddings are learned to minimize rating prediction error
3. **Recommendation Generation**: For a given user, the model predicts ratings for all unwatched movies and returns the top-N highest predicted ratings

## Future Improvements

- Add neural network layers for deep matrix factorization
- Implement implicit feedback (click data, view history)
- Add content-based features (genres, tags, metadata)
- Implement batch training for larger datasets
- Add hyperparameter tuning with cross-validation
- Deploy as a web API with Flask/FastAPI

## References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Matrix Factorization Techniques](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This project uses the MovieLens dataset, which is provided for research and educational purposes. Please cite:

```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: 
History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
```
