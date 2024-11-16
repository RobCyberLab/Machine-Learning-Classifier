import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Exercise 1: Load and analyze the dataset
def load_and_analyze_data():
    # Load the digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Print dataset information
    print(f"Number of samples in dataset: {len(X)}")
    print(f"Number of features per sample: {X.shape[1]}")
    print("Features represent: 8x8 pixel intensities of handwritten digit images")

    # Display a random sample
    random_idx = np.random.randint(0, len(X))
    plt.figure(figsize=(4, 4))
    plt.imshow(X[random_idx].reshape(8, 8), cmap='gray')
    plt.title(f'Example digit: {y[random_idx]}')
    plt.axis('off')
    plt.show()

    return X, y


# Exercise 2: Apply PCA dimensionality reduction
def apply_pca(X, n_components=8):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Print explained variance ratio
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance with {n_components} components: {explained_variance:.2f}%")

    return X_pca, pca


# Exercise 3: Train and evaluate SVM classifier
def train_evaluate_svm(X, y, test_size=0.4, random_state=42):
    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Test different kernels
    kernels = ['linear', 'rbf', 'poly']
    results = {}

    for kernel in kernels:
        # Train SVM
        svc = SVC(kernel=kernel)
        svc.fit(X_train, y_train)

        # Make predictions
        y_pred = svc.predict(X_valid)

        # Calculate accuracy
        accuracy = accuracy_score(y_valid, y_pred)
        results[kernel] = accuracy
        print(f"Accuracy with {kernel} kernel: {accuracy:.4f}")

    return results


# Exercise 4: Learning curves with different PCA components and training sizes
def plot_learning_curves(X, y):
    # Define parameters to test
    components = [4, 8, 16, 32]
    training_sizes = np.linspace(0.1, 0.7, 36)

    # Initialize dictionaries for errors
    train_errors = {n_comp: [] for n_comp in components}
    valid_errors = {n_comp: [] for n_comp in components}

    # Calculate learning curves for each number of components
    for n_comp in components:
        print(f"Processing {n_comp} components...")

        # Apply PCA
        X_pca, _ = apply_pca(X, n_components=n_comp)

        for train_size in training_sizes:
            # Split data
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_pca, y,
                train_size=train_size,
                random_state=42
            )

            # Train model
            svc = SVC(kernel='linear')
            svc.fit(X_train, y_train)

            # Calculate errors
            train_pred = svc.predict(X_train)
            valid_pred = svc.predict(X_valid)

            train_error = 1 - accuracy_score(y_train, train_pred)
            valid_error = 1 - accuracy_score(y_valid, valid_pred)

            train_errors[n_comp].append(train_error)
            valid_errors[n_comp].append(valid_error)

    # Plot results
    plt.figure(figsize=(12, 6))

    for n_comp in components:
        plt.plot(training_sizes * 100, train_errors[n_comp],
                 '--', label=f'Training (n_components={n_comp})')
        plt.plot(training_sizes * 100, valid_errors[n_comp],
                 '-', label=f'Validation (n_components={n_comp})')

    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Error')
    plt.title('Learning Curves for Different PCA Components')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Load and analyze data
    print("Loading and analyzing data...")
    X, y = load_and_analyze_data()

    # Apply PCA and train SVM
    print("\nApplying PCA and training SVM...")
    X_pca, _ = apply_pca(X)
    results = train_evaluate_svm(X_pca, y)

    # Plot learning curves
    print("\nGenerating learning curves...")
    plot_learning_curves(X, y)


if __name__ == "__main__":
    main()