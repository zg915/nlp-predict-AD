import pandas as pd

def split_dataset(file_path, label_column, test_fraction=0.2, random_state=42, specific = False):
    """
    Splits the dataset into train and test sets based on label, with specified fractions for each label.

    Parameters:
        file_path (str): Path to the JSON file.
        label_column (str): Name of the column containing labels.
        test_fraction (float): Fraction of each label to use for the test set (default is 0.2).
        random_state (int): Seed for reproducibility (default is 42).

    Returns:
        train_data (DataFrame): The training subset of the data.
        test_data (DataFrame): The test subset of the data.
    """
    # Load the data
    df = pd.read_json(file_path)
    
    # Split based on label
    df_label_0 = df[df[label_column] == 0]
    df_label_1 = df[df[label_column] == 1]

    if specific == True:
        df_label_0_train = df_label_0.sample(n = 200, random_state=random_state)
        df_label_0_test = df_label_0.drop(df_label_0_train.index)

        df_label_1_train = df_label_1.sample(n = 200, random_state=random_state)
        df_label_1_test = df_label_1.drop(df_label_1_train.index)
    
    else:

        # Train-test split for each label
        df_label_0_train = df_label_0.sample(frac=(1 - test_fraction), random_state=random_state)
        df_label_0_test = df_label_0.drop(df_label_0_train.index)

        df_label_1_train = df_label_1.sample(frac=(1 - test_fraction), random_state=random_state)
        df_label_1_test = df_label_1.drop(df_label_1_train.index)

    # Concatenate train and test sets from both labels
    train_data = pd.concat([df_label_0_train, df_label_1_train])
    test_data = pd.concat([df_label_0_test, df_label_1_test])

    return train_data, test_data
