import numpy as np
from sklearn.model_selection import train_test_split


n_freq = 32
n_time = 96
n_classes = 5


def load_data(
    features_path: str = "data/training.npy",
    labels_path: str = "data/training_labels.npy",
    stratified: bool = True,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    X, y = np.load(features_path), np.load(labels_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=133742069,
        stratify=y if stratified else None,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=694201337,
        stratify=y_train if stratified else None,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_training() -> tuple[np.ndarray, np.ndarray]:
    (X, y), _, _ = load_data()

    return X, y


def load_validation() -> tuple[np.ndarray, np.ndarray]:
    _, (X, y), _ = load_data()

    return X, y


def load_testing() -> tuple[np.ndarray, np.ndarray]:
    print(
        "WARNING: If you are training/experimenting, use the training and validation set instead.\n"
        "Use this function only for the very final evaluation before the competition ends."
    )

    _, _, (X, y) = load_data()

    return X, y


def load_testing_no_labels(feature_path: str = "data/test.npy") -> np.ndarray:
    print(
        "WARNING: If you are training/experimenting, use the training and validation set instead.\n"
        "Use this function only to get the final label predictions of the unknown labels."
    )

    X = np.load(feature_path)

    return X
