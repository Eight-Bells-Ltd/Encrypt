#!/bin/bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/sgx-aesm-service/aesm 
/opt/intel/sgx-aesm-service/aesm/aesm_service &

pid=$!

trap "kill ${pid}" TERM INT
sleep 2
exec "$@"

