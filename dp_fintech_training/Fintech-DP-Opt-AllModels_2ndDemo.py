import pickle
import pandas as pd
import numpy as np
import warnings

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from diffprivlib.models import DecisionTreeClassifier as DTC_DP
from diffprivlib.models import GaussianNB as GNB_DP
from diffprivlib.models import RandomForestClassifier as RFDP

warnings.filterwarnings("ignore")
DIRECTORY = Path("./data")


def read_in_files():
    map = (
        pd.read_csv(DIRECTORY / "ca_relation_typed.csv")[["person_key", "account_key"]]
        .drop_duplicates("account_key", keep="first")
        .drop_duplicates("person_key", keep="first")
    )

    account = pd.read_csv(DIRECTORY / "account_typed.csv").join(
        map.set_index("account_key"), on="account_key"
    )
    account = account[~account.person_key.isna()]
    deposit = pd.read_csv(DIRECTORY / "deposit_account_typed.csv").join(
        map.set_index("person_key"), on="person_key"
    )
    deposit = deposit[~deposit.account_key.isna()]
    payment = pd.read_csv(DIRECTORY / "payment_typed.csv").join(
        map.set_index("account_key"), on="account_key"
    )
    payment = payment[~payment.person_key.isna()]
    person = pd.read_csv(DIRECTORY / "person_typed.csv").join(
        map.set_index("person_key"), on="person_key"
    )
    person = person[~person.account_key.isna()]

    return account, deposit, payment, person

def clean_files(account, deposit, payment, person):
    account = account.loc[:, [
        "account_key",
        "base_interest_rate",
        "repay_frequency",
        "number_of_total_installments",
        "delay_days",
        "overdue_expenses",
        "total_balance",
        "collateral_amount",
    ]]

    min_delay = account.delay_days.min()
    max_delay = account.delay_days.max()

    account.delay_days = account.delay_days.apply(
        lambda x: ((x - min_delay) / (max_delay - min_delay)) * 210
    )

    deposit = deposit[
        [
            "account_key",
            "accounting_balance",
            "available_balance",
        ]
    ]

    payment = payment[["account_key", "capital_amount", "payinterest", "payexpenses"]]

    person = person[
        [
            "marital_status",
            "gender",
            "account_key",
        ]
    ]

    return account, deposit, payment, person

def join_files(account, deposit, payment, person):
    df = (
        account.join(deposit.set_index("account_key"), on="account_key")
        .join(payment.set_index("account_key"), on="account_key")
        .join(person.set_index("account_key"), on="account_key")
        .drop("account_key", axis=1)
        .reset_index(drop=True)
    )

    df.marital_status.fillna("single", inplace=True)
    df.gender.fillna("M", inplace=True)
    df.fillna(0, inplace=True)

    return df

def feature_engineering_train_test_split(df):
    categorical_columns = ["gender", "marital_status"]
    categories = pd.get_dummies(df[categorical_columns], drop_first=True)
    df[categories.columns] = categories
    df.drop(categorical_columns, axis=1, inplace=True)

    df["target_variable"] = df.delay_days.apply(lambda x: 1 if x > 180 else 0)
    df.drop("delay_days", axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["target_variable"]),
        df.target_variable,
        test_size=0.3,
        stratify=df.target_variable,
    )

    columns_to_scale = [
        "base_interest_rate",
        "repay_frequency",
        "number_of_total_installments",
        "overdue_expenses",
        "total_balance",
        "collateral_amount",
        "accounting_balance",
        "available_balance",
        "capital_amount",
        "payinterest",
        "payexpenses",
    ]

    scaler = StandardScaler().fit(X_train[columns_to_scale])
    X_train[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    # Add noise to the training data
    epsilon = 0.5
    sensitivity = 1.0  # Assuming all features have the same sensitivity
    delta = 1e-5  # Target delta
    n = X_train.shape[0]  # Number of training samples
    scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.laplace(0, scale, X_train.shape)
    X_train_noisy = X_train + noise

    return X_train_noisy, X_train, X_test, y_train, y_test

#   START: GaussianNB MODEL -- START: GaussianNB MODEL -- START: GaussianNB MODEL -- START: GaussianNB MODEL
def fit_model_sklearn_GaussianNB(X_train, y_train):
    # Define the parameter grid
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        'priors': [None, [0.5, 0.5], [0.6, 0.4]]  # Example priors, adjust based on your class distribution
    }
    
    # Initialize the GaussianNB model
    gnb = GaussianNB()
    
    # Initialize GridSearchCV with the model and parameter grid
    grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=5, scoring='accuracy')
    
    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best model with the best found hyperparameters
    best_gnb = grid_search.best_estimator_

    # Print the accuracy for each combination of parameters
    print("Grid search results for GaussianNB:")
    for params, mean_score, scores in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
        print(f"Parameters: {params}, Accuracy: {mean_score:.4f} (+/- {scores:.4f})")
    
    return best_gnb

def OLD_fit_model_no_Noise_Added_GaussianNB(X_train, y_train):
    gnb = GNB_DP().fit(X_train, y_train)
    return gnb

def fit_model_DiffPrivacy_GaussianNB(X_train, y_train):
    # Define the parameter grid
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
        'priors': [None, [0.5, 0.5], [0.6, 0.4]]  # Example priors, adjust based on your class distribution
    }
    
    # Initialize the GaussianNB model with epsilon fixed at 0.5
    gnb_dp = GNB_DP(epsilon=0.5)
    
    # Initialize GridSearchCV with the model and parameter grid
    grid_search = GridSearchCV(estimator=gnb_dp, param_grid=param_grid, cv=5, scoring='accuracy')
    
    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best model with the best found hyperparameters
    best_gnb_dp = grid_search.best_estimator_

    # Print the accuracy for each combination of parameters
    print("Grid search results for GaussianNB with Differential Privacy:")
    for params, mean_score, std_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
        print(f"Parameters: {params}, Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return best_gnb_dp

def OLD_fit_model_DiffPrivacy_GaussianNB(X_train, y_train):
    gnb = GNB_DP(epsilon=0.5).fit(X_train, y_train)
    return gnb
#   END: GaussianNB MODEL -- END: GaussianNB MODEL -- END: GaussianNB MODEL -- END: GaussianNB MODEL

#   START: DecisionTreeClassifier MODEL -- START: DecisionTreeClassifier MODEL -- START: DecisionTreeClassifier MODEL -- START: DecisionTreeClassifier MODEL
def fit_model_sklearn_DecisionTreeClassifier(X_train, y_train):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50],        
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': [None, 'sqrt', 'log2']
    }
    
    dtc = DecisionTreeClassifier()
    grid_search = GridSearchCV(dtc, param_grid, cv=5, scoring='accuracy', return_train_score=True)
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Print all parameter combinations with their corresponding accuracy
    results = pd.DataFrame(grid_search.cv_results_)
    for _, row in results.iterrows():
        print(f"Params: {row['params']} - Mean Test Accuracy: {row['mean_test_score']:.4f} - Mean Train Accuracy: {row['mean_train_score']:.4f}")
    

    # Print best parameters
    print(f"Best parameters found: {grid_search.best_params_}")

    return grid_search.best_estimator_

def OLD_fit_model_no_Noise_Added_DecisionTreeClassifier(X_train, y_train):
    dtc = DTC_DP().fit(X_train, y_train)
    return dtc

def fit_model_DiffPrivacy_DecisionTreeClassifier(X_train, y_train):
    param_grid = {
        'max_depth': [0, 1, 2, 3, 5, 10],
        'criterion': ['gini', 'entropy'],
    }
    
    dtc_dp = DTC_DP(epsilon=0.5)
    grid_search = GridSearchCV(dtc_dp, param_grid, cv=5, scoring='accuracy', return_train_score=True)
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print(f"Best parameters found for DP model: {grid_search.best_params_}")
    
    # Print all parameter combinations with their corresponding accuracy
    results = pd.DataFrame(grid_search.cv_results_)
    for _, row in results.iterrows():
        print(f"Params: {row['params']} - Mean Test Accuracy: {row['mean_test_score']:.4f} - Mean Train Accuracy: {row['mean_train_score']:.4f}")
    
    # Print best parameters
    print(f"Best parameters found for DP model: {grid_search.best_params_}")

    return grid_search.best_estimator_
#   END: DecisionTreeClassifier MODEL -- END: DecisionTreeClassifier MODEL -- END: DecisionTreeClassifier MODEL -- END: DecisionTreeClassifier MODEL

#   START: RandomForestClassifier MODEL -- START: RandomForestClassifier MODEL -- START: RandomForestClassifier MODEL -- START: RandomForestClassifier MODEL
def fit_model_sklearn_RandomForestClassifier(X_train, y_train):
    param_grid = {
        'n_estimators': [10, 50, 100, 150, 200, 250],
        'max_depth': [10, 5, 3, None],
        'min_samples_split': [2, 5, 7, 10],
        'min_samples_leaf': [1, 4, 6, 9],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True]
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print all results
    results = pd.DataFrame(grid_search.cv_results_)
    for index, row in results.iterrows():
        print(f"Params: {row['params']} - Mean Test Accuracy: {row['mean_test_score']:.4f}")

    return grid_search.best_estimator_

def fit_model_DiffPrivacy_RandomForestClassifier(X_train, y_train):
    classes = [0, 1]  # Specify the classes to avoid privacy leakage warning
    param_grid = {
        'n_estimators': [10, 50, 100, 150, 200, 250],
        'max_depth': [10, 5, 3, 1],
        'random_state': [0, 8, 12, 18, 27, 35, 42],
        'bounds': [(0, 1), (-1, 1)],
    }
    rf = RFDP(epsilon=0.05, classes=classes)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print all results
    results = pd.DataFrame(grid_search.cv_results_)
    for index, row in results.iterrows():
        print(f"Params: {row['params']} - Mean Test Accuracy: {row['mean_test_score']:.4f}")

    return grid_search.best_estimator_
#   END: RandomForestClassifier MODEL -- END: RandomForestClassifier MODEL -- END: RandomForestClassifier MODEL -- END: RandomForestClassifier MODEL

#   START: NeuralNetwork MODEL -- START: NeuralNetwork MODEL -- START: NeuralNetwork MODEL -- START: NeuralNetwork MODEL
def fit_model_sklearn_NeuralNetwork(X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(10, 5), (5, 10, 5)],# [(100,), (100, 50), (75, 100, 25), (50, 100, 50), (25, 50, 75, 100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [10000]#[200, 500, 1000]
    }

    mlp = MLPClassifier()
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, n_jobs=-1)#, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")

    # Print the accuracy of each combination of parameters
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, std, param in zip(means, stds, params):
        print(f"{param}: mean accuracy={mean:.4f} (std={std:.4f})")

    return grid_search.best_estimator_
#   END: NeuralNetwork MODEL -- END: NeuralNetwork MODEL -- END: NeuralNetwork MODEL -- END: NeuralNetwork MODEL

def return_accuracy(gnb, X_test, y_test):
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def return_accuracy_NeuralNetwork(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


if __name__ == "__main__":
    accounts, deposits, payments, persons = read_in_files()
    accounts, deposits, payments, persons = clean_files(
        accounts, deposits, payments, persons
    )
    full_df = join_files(accounts, deposits, payments, persons)
    X_train_noisy, X_train, X_test, y_train, y_test = feature_engineering_train_test_split(full_df)

    # START # GaussianNB Models
    model = fit_model_sklearn_GaussianNB(X_train, y_train)
    a = return_accuracy(model, X_test, y_test)
    print(f"The accuracy of the plain GaussianNB model was: {a:.4f}")
    print(f"Best parameters for the plain GaussianNB model: {model.get_params()}")    
    
#    model_dp = fit_model_DiffPrivacy_GaussianNB(X_train, y_train)
#    a_dp = return_accuracy(model_dp, X_test, y_test)
#    print(f"The accuracy of the Differential Privacy GaussianNB model was: {a_dp:.4f}")
#    print(f"Best parameters for the Differential Privacy GaussianNB model: {model_dp.get_params()}")
    
    # Serialize the best models
    with open('best_plain_model_GaussianNB.pkl', 'wb') as f:
        pickle.dump(model, f)
    
 #   with open('best_DiffPrivacy_model_GaussianNB.pkl', 'wb') as f:
 #       pickle.dump(model_dp, f)
    # END # GaussianNB Models

    # START # DecisionTreeClassifier Models
    # Train and evaluate the plain model
    model = fit_model_sklearn_DecisionTreeClassifier(X_train, y_train)
    a = return_accuracy(model, X_test, y_test)
    print(f"The accuracy of the plain model was: {a:.4f}")
    print(f"Best parameters for plain model: {model.get_params()}")
    
    # Serialize the DecisionTreeClassifier plain model
    with open('best_plain_model_DecisionTreeClassifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Train and evaluate the DP model
#    model_dp = fit_model_DiffPrivacy_DecisionTreeClassifier(X_train, y_train)
#    a_dp = return_accuracy(model_dp, X_test, y_test)
#    print(f"The accuracy of the DP model was: {a_dp:.4f}")
#    print(f"Best parameters for DP model: {model_dp.get_params()}")
    
    # Serialize the DecisionTreeClassifier DP model
#    with open('best_DiffPrivacy_model_DecisionTreeClassifier.pkl', 'wb') as f:
#        pickle.dump(model_dp, f)

    # Check model depth and number of nodes for plain model
    print(f"Plain model depth: {model.get_depth()}")
    print(f"Plain model number of nodes: {model.tree_.node_count}")

    # Check model depth and number of nodes for DP model
 #   print(f"DP model depth: {model_dp.get_depth()}")
 #   print(f"DP model number of nodes: {model_dp.tree_.node_count}")
    # END # DecisionTreeClassifier Models
    
    # START # RandomForestClassifier Models
    model = fit_model_sklearn_RandomForestClassifier(X_train, y_train)
    a = return_accuracy(model, X_test, y_test)
    print(f"The accuracy of the best plain model was: {a:.4f} with parameters {model.get_params()}")    
    # Serialize the RandomForestClassifier plain model
    with open('best_plain_model_RandomForestClassifier.pkl', 'wb') as f:
        pickle.dump(model, f)

#    model_dp = fit_model_DiffPrivacy_RandomForestClassifier(X_train, y_train)
#    a_dp = return_accuracy(model_dp, X_test, y_test)
#    print(f"The accuracy of the best DP model was: {a_dp:.4f} with parameters {model_dp.get_params()}")
#    # Serialize the RandomForestClassifier DP model
#    with open('best_DiffPrivacy_model_RandomForestClassifier.pkl', 'wb') as f:
#        pickle.dump(model_dp, f)
    # END # RandomForestClassifier Models

    # START # NeuralNetwork Models
    print("Training model with grid search on clean data...")
    clean_model = fit_model_sklearn_NeuralNetwork(X_train, y_train)
    clean_accuracy = return_accuracy_NeuralNetwork(clean_model, X_test, y_test)
    print(f"The accuracy of the neural network model trained on clean data was: {clean_accuracy:.4f}")
    print(f"Best parameters for model trained on clean data: {clean_model.get_params()}")
    
    # Serialize the clean data model
    with open('best_plain_model_NeuralNetwork.pkl', 'wb') as f:
        pickle.dump(clean_model, f)    

#    print("Training model with grid search on noisy data...")
#    model_noisy = fit_model_sklearn_NeuralNetwork(X_train_noisy, y_train)
#    a_noisy = return_accuracy_NeuralNetwork(model_noisy, X_test, y_test)
#    print(f"The accuracy of the neural network model trained on noisy data was: {a_noisy:.4f}")
#    print(f"Best parameters for model trained on noisy data: {model_noisy.get_params()}")

#    # Serialize the noisy data model
#    with open('best_DiffPrivacy_model_NeuralNetwork.pkl', 'wb') as f:
#        pickle.dump(model_noisy, f)
    # END # NeuralNetwork Models