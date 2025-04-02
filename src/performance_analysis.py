import random

# Generate 5 random indices
random_indices = random.sample(range(len(X_test)), 5)

# Select the random test values
X_random_test = X_test.iloc[random_indices]
y_random_test = y_test.iloc[random_indices]

# Predict using RF and LR for the random test values
rf_random_predictions = rf_model.predict(X_random_test)
lr_random_predictions = lr_model.predict(X_random_test)


# Prepare data for plotting

# Convert one-hot encoded predictions back to original labels
# Assuming you have the inverse transformation method for y_encoded
# Example using the argmax method:
rf_random_predictions = [list(unique_slice_types)[i] for i in rf_random_predictions.argmax(axis=1)]
lr_random_predictions = [list(unique_slice_types)[i] for i in lr_random_predictions.argmax(axis=1)]

y_random_test_labels = [list(unique_slice_types)[i] for i in y_random_test.values.argmax(axis=1)]
# Create a line graph

plt.figure(figsize=(10, 6))
plt.plot(range(5), rf_random_predictions, marker='o', label='Random Forest')
plt.plot(range(5), lr_random_predictions, marker='x', label='Logistic Regression')
plt.plot(range(5), y_random_test_labels, marker='s', linestyle='--', label = 'Actual')

plt.xlabel('Test Instance')
plt.ylabel('Predicted Slice Type')
plt.title('Random Forest vs. Logistic Regression Predictions')
plt.xticks(range(5))  # Set x-axis ticks for each test instance
plt.legend()
plt.show()
