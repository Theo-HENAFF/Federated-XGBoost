# Federated XGBoost Example

## Introduction

**Federated XGBoost** is a gradient boosting library for the federated setting, based off the popular [XGBoost](https://github.com/dmlc/xgboost) project.
In addition to offering the same efficiency, flexibility, and portability that vanilla XGBoost provides,
Federated XGBoost enables multiple parties to jointly compute a model while keeping their data on site, avoiding the need for a central data storage. 

This project is currently under development as part of the broader **mc<sup>2</sup>** effort (i.e., **M**ultiparty **C**ollaboration and **C**oopetition) by the UC Berkeley [RISE Lab](https://rise.cs.berkeley.edu/).

Please feel free to reach out to us if you would like to use Federated XGBoost for your applications. We also welcome contributions to our work!

## Installation

1. Clone this repository and its submodules.

```
git clone --recursive https://github.com/mc2-project/federated-xgboost.git
```

2. Setting up Federated XGBoost.

Optionnal : Create a conda env
```
conda create --name federated-xgboost python=3.8
conda activate federated-xgboost
```

Install Federated XGBoost dependencies.
```
sudo apt-get install cmake libmbedtls-dev
pip3 install pandas numpy grpcio grpcio-tools
```

3. Build Federated XGBoost.
```
cd federated-xgboost
mkdir build
cd build
cmake ..
make
```

4. Install the Python package.
```
cd python-package
sudo python3 setup.py install
```

## Fraud detection example


### Disclaimer
This code has been tested on the April 29th 2021, last commit was on August 5th 2020, new commit might lead to failure.
Feel free to keep it updated if it is not working properly.

### Fraud detection example
This example uses the tutorial located in `demo/fraud`. In this tutorial, each of the two parties in the federation
starts an RPC server on port 50051 to listen for the aggregator. The aggregator sends invitations to all parties to join
the computation. Once all parties have accepted the invitation, training commences -- the training script `demo fraud.py` is run.

0. Install additional dependencies
 ```
 pip install scikit-learn
 ```

1. The aggregator must modify `hosts.config` to contain the IP addresses of all parties in the federation.
   Each line in `hosts.config` follows the following format:

```
<ip_addr>:<port>
```

For the purposes of this demo, `<port>` should be `50051` and `<ip_addr>` should be ipv4 format.

2. This demo uses data from the
   [Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/ntnu-testimon/paysim1).
   The data have been prepared according to analysis
   [Predicting Fraud in Financial Payment Services](https://www.kaggle.com/arjunjoshua/predicting-fraud-in-financial-payment-services)
   made by [Arjun Joshua](https://www.kaggle.com/arjunjoshua)
   The script `0 - Data prep.py` transforms the base dataset to the example ready data.
   
The `fraud/data/` directory contains multiples files including training data: `train_1.csv`, `train_2.csv`,
   `val1.csv`, `val2.csv` and test data: `test.csv`.
   

3. Start the RPC server at each party. 

```
python3 serve.py
```

4. At the aggregator, send invitations to all parties.

```
dmlc-core/tracker/dmlc-submit --cluster rpc --num-workers 2 --host-file '/path/to/federated-xgboost/demo/fraud/hosts.config'  --worker-memory 4g '/path/to/federated-xgboost/demo/fraud/demo.py'
```

Each party should receive an invitation through their console:

```
Request from aggregator [ipv4:172.31.27.60:50432] to start federated training session:
Please enter 'Y' to confirm or 'N' to reject.
Join session? [Y/N]:
```

5. Once all parties submit `Y`, training begins.
   
6. Follow guided instructions and see the results !
