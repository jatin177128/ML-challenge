from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Extract text and prepare data for training
train_df['text'] = train_texts  # Assign the extracted texts to the DataFrame
vectorizer = TfidfVectorizer(max_features=500)

# Split the data in batches for incremental training
batch_size = 10000
num_batches = len(train_df) // batch_size

# Create the SGDRegressor model
model = SGDRegressor()

# Train incrementally on batches of data
for i in range(num_batches):
    batch_df = train_df.iloc[i*batch_size:(i+1)*batch_size]
    X_batch = vectorizer.fit_transform(batch_df['text']).toarray()
    y_batch = batch_df['entity_value'].str.extract(r'(\d+(\.\d+)?)')[0].astype(float)

    # Partial fit for incremental training
    model.partial_fit(X_batch, y_batch)

# Save model after training
import joblib
joblib.dump(model, 'sgd_model.pkl')
