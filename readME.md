## Installation

Download and run the application:
```sh
-docker build -t encryptdev/dp_fintech_training:0.1 -f ./dp-fintech/Docker/Dockerfile.dp.fintech.training . 
```
```sh
-docker run --rm -p 9443:9443 --name manager -e CMD='python3.9,/dp-fintech/dp_fintech_training/dummy_fintech.py' -e API_URL='http://10.0.0.6:9000/computation_output' encryptdev/dp_fintech_training:0.1
```

## Requests

Upload csv files:
```sh
curl --location 'http://127.0.0.1:9443/upload' --form 'file=@"/path_to_file/account_typed.csv"' --form 'file=@"/path_to_file/person_typed.csv"' --form 'file=@"/path_to_file/payment_typed.csv"' --form 'file=@"/path_to_file/deposit_account_typed.csv"' --form 'file=@"/path_to_file/ca_relation_typed.csv"'
```
```sh
Insert ID:
curl --location 'http://127.0.0.1:9443/insert_ID' --header 'Content-Type: application/json' --data '{"ID":1}'
```

## Expected Output
```python
========Output of the script========

The accuracy of the plain model was: 0.9426
The accuracy of the DP model was: 0.8655

========Computation complete!========
```