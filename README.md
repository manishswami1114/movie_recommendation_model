# Movie Recommendation System

This project implements a Movie Recommendation System using a Neural Collaborative Filtering (NCF) model, specifically built with PyTorch. It provides movie recommendations to users based on their past ratings and interactions with movies. This system leverages embeddings for users and movies, making predictions using a feedforward neural network.

 **Table of Contents**

* Overview
* Features
* Requirements
* Installation
* Usage
* Model Architecture
* Endpoints
* License
* Overview

The Movie Recommendation System predicts movie ratings for a given user-item pair. The recommendation system is built using a deep learning model that utilizes user embeddings and movie embeddings along with a feedforward neural network to predict ratings. The model was trained using the MovieLens dataset, and this project provides a FastAPI interface for real-time predictions.

### Core Components:RecommenderNet: 

The core recommendation model implemented in PyTorch.
FastAPI: For serving the model through a web API.
Endpoints:

/health: To check the health of the API.

/predict: To predict ratings for a given user-item pair.
Features

Real-time Recommendations: Predict ratings based on user and movie IDs.

Model Serving: Use FastAPI to serve the recommendation model.
Easy Integration: Simple HTTP API to integrate with other systems.
Model Architecture: Based on the Neural Collaborative Filtering (NCF) approach.

### Requirements

Python 3.7+
PyTorch
FastAPI
Uvicorn
Pandas
Numpy
Scikit-learn
You can install the required packages using the following:

```python
pip install <packages name>
```
Installation

Clone the repository:
```
git clone https://github.com/Manishswami1114/Movie-Recommendation-System.git
```
```
cd Movie-Recommendation-System
```

Download the trained model (movie_rating_model.pth) and place it in the project directory.
Usage

Running the FastAPI Server:
To start the FastAPI server and serve the recommendation model:

uvicorn app:app --reload
This will start the API on ```http://127.0.0.1:8000```. The --reload flag enables hot-reloading during development.

Predicting Ratings:
To get movie recommendations, use the /predict endpoint by making a POST request with the following JSON payload:
```json
{
  "user_id": 1,
  "movie_id": 50
}
```
Health Check:
To check if the model is loaded and the API is working, visit the /health endpoint:

```
http://127.0.0.1:8000/health
```
Model Architecture

RecommenderNet
The model is built using Neural Collaborative Filtering (NCF), which is based on the following components:

User Embedding Layer: Embeds the user IDs into a dense vector.
Movie Embedding Layer: Embeds the movie IDs into a dense vector.
Feedforward Neural Network: Takes the concatenated user and movie embeddings and passes them through a fully connected neural network to predict the rating.
Model Code:
```python
class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.movie_embedding = nn.Embedding(n_movies, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, user_id, movie_id):
        user_embedded = self.user_embedding(user_id)
        movie_embedded = self.movie_embedding(movie_id)
        x = torch.cat([user_embedded, movie_embedded], dim=-1)
        x = F.relu(self.fc1(x))
        rating_pred = self.fc2(x)
        return rating_pred
```
Endpoints
/health (GET)
Description: Returns the health status of the API and whether the model is loaded successfully.
Response:
```json
{
  "status": "healthy"
}
```
/predict (POST)
Description: Predicts the rating for a given user-item pair.
Request Body:
```json
{
  "user_id": 1,
  "movie_id": 50
}
```
Response:
```json
{
  "predicted_rating": 4.23
}
```
