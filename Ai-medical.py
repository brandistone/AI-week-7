import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

def generate_medical_image_data(num_samples=1000, image_size=16, tumor_intensity_range=(0.6, 0.9)):
    """
    Simulate medical image data with the following characteristics:
    - Healthy tissues have pixel values in a lower range
    - Tumor tissues have pixel values in a higher range with distinct patterns
    
    Args:
        num_samples: Number of images to generate
        image_size: Size of each square image (image_size x image_size)
        tumor_intensity_range: Range for tumor pixel intensity values
        
    Returns:
        X: Flattened image data
        y: Labels (0: healthy, 1: tumor)
    """
    # Initialize arrays
    X = np.zeros((num_samples, image_size * image_size))
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # Generate base image (background tissue)
        image = np.random.uniform(0.1, 0.5, (image_size, image_size))
        
        # Randomly assign label (50% chance of tumor)
        has_tumor = np.random.randint(0, 2)
        y[i] = has_tumor
        
        # If sample has tumor, add tumor-like structure
        if has_tumor:
            # Random position for tumor center
            tumor_x = np.random.randint(3, image_size-3)
            tumor_y = np.random.randint(3, image_size-3)
            tumor_size = np.random.randint(2, 5)
            
            # Create tumor pattern (roughly circular)
            for dx in range(-tumor_size, tumor_size+1):
                for dy in range(-tumor_size, tumor_size+1):
                    # Check if within tumor radius and image boundaries
                    if (dx**2 + dy**2 <= tumor_size**2 and 
                        0 <= tumor_x + dx < image_size and 
                        0 <= tumor_y + dy < image_size):
                        # Assign higher intensity to tumor pixels
                        image[tumor_x + dx, tumor_y + dy] = np.random.uniform(
                            tumor_intensity_range[0], 
                            tumor_intensity_range[1]
                        )
        
        # Flatten the image and store
        X[i] = image.flatten()
    
    return X, y

def train_tumor_detection_model():
    """
    Train an AI model to detect tumors in simulated medical images
    """
    print("Generating simulated medical image data...")
    X, y = generate_medical_image_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")
    
    # Initialize and train the Random Forest model
    print("Training AI model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    
    # Generate confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
    print(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
    
    return model

def analyze_new_image(model, image_size=16):
    """
    Generate a new simulated image and predict whether it contains a tumor
    """
    # Generate a single new test image (50% chance of having tumor)
    has_tumor = np.random.randint(0, 2)
    
    # Create the image with or without tumor
    image = np.random.uniform(0.1, 0.5, (image_size, image_size))
    
    if has_tumor:
        # Add tumor pattern similar to training data
        tumor_x = np.random.randint(3, image_size-3)
        tumor_y = np.random.randint(3, image_size-3)
        tumor_size = np.random.randint(2, 5)
        
        for dx in range(-tumor_size, tumor_size+1):
            for dy in range(-tumor_size, tumor_size+1):
                if (dx**2 + dy**2 <= tumor_size**2 and 
                    0 <= tumor_x + dx < image_size and 
                    0 <= tumor_y + dy < image_size):
                    image[tumor_x + dx, tumor_y + dy] = np.random.uniform(0.6, 0.9)
    
    # Flatten image for prediction
    flattened_image = image.flatten().reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(flattened_image)
    prediction_proba = model.predict_proba(flattened_image)
    
    # Get result
    result = "Tumor detected!" if prediction[0] == 1 else "No tumor detected."
    confidence = max(prediction_proba[0]) * 100
    
    # Print results
    print("\nAnalyzing new medical image...")
    print(f"Actual label: {'Tumor' if has_tumor else 'Healthy'}")
    print(f"AI Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Return true label and prediction for verification
    return has_tumor, prediction[0]

# Main execution
if __name__ == "__main__":
    print("AI Medical Image Analysis Simulation")
    print("-" * 40)
    
    # Train the model
    model = train_tumor_detection_model()
    
    # Analyze multiple new images to demonstrate performance
    print("\nTesting model on new images:")
    correct_predictions = 0
    num_tests = 5
    
    for i in range(num_tests):
        print(f"\nTest Image #{i+1}")
        true_label, predicted_label = analyze_new_image(model)
        if true_label == predicted_label:
            correct_predictions += 1
    
    print(f"\nOverall accuracy on {num_tests} new test cases: {correct_predictions/num_tests:.2f}")