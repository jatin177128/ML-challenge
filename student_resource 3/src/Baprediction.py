# Load test data and prepare text features
test_image_paths = [os.path.join(test_image_dir, os.path.basename(url)) for url in test_image_links]
test_texts = batch_extract_text(test_image_paths)
test_df['text'] = test_texts
X_test = vectorizer.transform(test_df['text']).toarray()

# Load the trained model
model = joblib.load('sgd_model.pkl')

# Make predictions in batches
test_df['prediction'] = ""

for i in range(num_batches):
    X_test_batch = X_test[i*batch_size:(i+1)*batch_size]
    predictions = model.predict(X_test_batch)
    
    # Format predictions
    test_df.loc[i*batch_size:(i+1)*batch_size-1, 'prediction'] = predictions

# Save predictions
test_df[['index', 'prediction']].to_csv('test_out.csv', index=False)
