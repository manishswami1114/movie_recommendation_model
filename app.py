from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

# Load the trained model
class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_movies, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.movie_embedding = nn.Embedding(n_movies, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user_id, movie_id):
        user_vector = self.user_embedding(user_id)
        movie_vector = self.movie_embedding(movie_id)
        x = torch.cat([user_vector, movie_vector], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze()

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model (make sure it is on the right device)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# You should load the correct model state, assuming the n_users and n_movies values are known
n_users = 671  # Replace with actual number of unique users
n_movies = 9066  # Replace with actual number of unique movies
model = RecommenderNet(n_users, n_movies, embedding_size=50).to(device)
model.load_state_dict(torch.load('movie_rating_model.pth'), strict=False)

model.eval()  # Set model to evaluation mode

# Define Pydantic model for request body
class RatingRequest(BaseModel):
    user_id: int
    movie_id: int

# Define health check endpoint
@app.get("/health")
def health_check():
    return {"status": "Healthy"}

# Define prediction endpoint
@app.post("/predict")
def predict_rating(request: RatingRequest):
    user_id = request.user_id
    movie_id = request.movie_id

    # Convert inputs to tensors and move to the appropriate device
    user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
    movie_tensor = torch.tensor([movie_id], dtype=torch.long).to(device)

    # Get the prediction
    with torch.no_grad():
        predicted_rating = model(user_tensor, movie_tensor).item()

    return {"predicted_rating": predicted_rating}

