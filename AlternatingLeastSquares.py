import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

# Step 1: Collect user preference data
# Assume we have a user-item interaction dataset called 'interactions'
# where each row contains (user_id, item_id, rating)
interactions = [
    (1, 1, 5),
    (1, 2, 3),
    (2, 1, 4),
    (2, 2, 1),
    (2, 3, 4),
    # ...
]

# Step 2: Preprocess the data

# Step 3: Create a user-item matrix
users, items, ratings = zip(*interactions)
user_item_matrix = coo_matrix((ratings, (users, items)))

# Step 4: Split the data
train_matrix, test_matrix = train_test_split(user_item_matrix, test_size=0.2)

# Step 5: Train the model
model = AlternatingLeastSquares()
model.fit(train_matrix)

# Step 6: Generate recommendations
user_ids = range(user_item_matrix.shape[0])
recommendations = []
for user_id in user_ids:
    recommended_items = model.recommend(user_id, train_matrix)
    recommendations.append(recommended_items)

# Step 7: Evaluate the model
# Assuming we have ground truth ratings for the test set
true_ratings = test_matrix.data
predicted_ratings = model.predict(test_matrix.row, test_matrix.col)
precision = precision_score(true_ratings, predicted_ratings)
recall = recall_score(true_ratings, predicted_ratings)

# Print the recommendations and evaluation metrics
print("Recommendations:")
for user_id, recommended_items in zip(user_ids, recommendations):
    print(f"User {user_id}: {recommended_items}")

print(f"Precision: {precision}")
print(f"Recall: {recall}")
