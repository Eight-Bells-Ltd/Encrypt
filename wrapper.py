import os, sys, requests, subprocess, functools

# to flush the print in gramine (sometime it gets stucked)
print = functools.partial(print, flush=True)

# environment variable set in the 'docker run' command
cmd = os.getenv('CMD')                                          # Command that starts the computations
api_url = os.getenv('API_URL')                                  # For api URL to upload the computation result
computation_script_path = os.getenv('COMPUTATION_SCRIPT_PATH')  # For the script that start the computation 
# env variables created in the Dockerfile
result_file = os.getenv('RESULT_FILE')                          # File where we save the compuation result
computation_ID_file = os.getenv('COMPUTATION_ID_FILE')          # File where we save the compuation ID
src_dir = os.getenv('SRC_DIR')                                  # Path to the folder that contains the computation script and resources
ppt = os.getenv('PPT')                                          # For the privacy-preserving technology to be used

# The server is the script that enables the data owner to upload their data
def start_server():
    script_path = 'server.py'
    # Execute the script
    print ('========Executing the server to upload the data!========\n')
    if ppt == 'tee':
        result = subprocess.run(['gramine-sgx', './server', script_path], stdout=subprocess.PIPE)
    else:
        result = subprocess.run(['/usr/bin/python3.9', script_path], stdout=subprocess.PIPE)
    # Extract stdout
    stdout = result.stdout.decode('utf-8')
    print("========Output of the script=========\n")
    print(stdout)
    print ('========The server stopped!==========\n')


# Start the computation
def start_computation():
    # Split the string 'cmd' into substrings using comma as separator
    # we do this to determine the command that needs to be launched in subprocess.run.
    commands = cmd.split(',')
    # Remove any whitespace
    commands = [command.strip() for command in commands]

    # Execute the script
    print ('========Executing the computation!========\n')
    print(commands)
    result = subprocess.run(commands, stdout=subprocess.PIPE)
    # Extract stdout
    stdout = result.stdout.decode('utf-8')
    print("========Output of the script========\n")
    print(stdout)
    print ('========Computation complete!========\n')

# function that sends the result of the computation to the api ingress service
def send_to_api():
    with open(computation_ID_file, "r") as f:
        computation_ID = f.read()
    with open(result_file, "r") as file:
        content = file.read()

    data = {'computation_ID': computation_ID, 'result': content}
    print(data)
    response = requests.post(api_url, json=data)

    if response.status_code == 200:
        print('POST request was successful!')
        print('Response content:')
        print(response.text)
    else:
        print(f'POST request failed with status code: {response.status_code}')
    sys.exit()    
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == '__main__':
    print('========Starting the manager========\n')
    start_server()
    start_computation()
    send_to_api()


