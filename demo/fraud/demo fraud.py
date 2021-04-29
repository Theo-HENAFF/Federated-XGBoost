import sys
import federatedxgboost as fxgb

from sklearn.metrics import classification_report

# Federated XGBoost automatically runs your training script and passes it the rabit configuration as an argument
# All scripts must start with this line and pass `rabit_config` into xgb.Federated()
rabit_config = sys.argv[1]

################################
# Instantiate Federated XGBoost
################################
fed = fxgb.Federated(rabit_config)

# Get number of federating parties
print("Number of parties in federation: ", fed.get_num_parties())

################################
# Load training data
################################
# Ensure that each party's data is in the same location with the same name

print("\nAbsolute path to data folder (ex: /home/Theo.Henaff/FXGB/example/data/) : ")
path_data = input()

print("\nName of the training data file (ex: train.csv) : ")
train_file = input()

print("\nName of the validation data file (ex: val1.csv) : ")
val_file = input()

print("\nName of the testing data file (ex: test.csv) : ")
test_file = input()

dtrain = fed.load_data(path_data+train_file)
dval = fed.load_data(path_data+val_file)

# Calculate the unbalance between True and False labels
weights = (dtrain.get_label() == 0).sum() / (1.0 * (dtrain.get_label() == 1).sum())
print("\nWeight : {}".format(weights))

################################
# Train a model
################################
params = {
        "max_depth": 5,
        "scale_pos_weight": weights,
        }
num_rounds = 100

print("\nTraining")

bst = fxgb.train(params, dtrain, num_rounds, evals=[(dtrain, "dtrain"), (dval, "dval")])


################################
# Get predictions
################################
# Loading test data
dtest = fed.load_data(path_data+test_file)

print("Predicting")
ypred = bst.predict(dtest)
ypred = [0.0 if pred < 0.5 else 1.0 for pred in ypred]

print("\nThe first twenty REAL predictions are: ", ypred[:20])

print(classification_report(dtest.get_label(), ypred, target_names=["NotFraud", "Fraud"]))


# Save the model
bst.save_model("XGBoost_model.model")

# Shutdown
fed.shutdown()
exit()
