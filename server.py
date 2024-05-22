from flask import Flask, request, jsonify, abort
import os, signal, functools
from pathlib import Path
from flask_cors import CORS

# to flush the print in gramine (sometime it gets stucked)
print = functools.partial(print, flush=True)

app = Flask(__name__)
CORS(app)

# env variables created in the Dockerfile
src_dir = os.getenv('SRC_DIR')                          # Path to the folder that contains the computation script and resources
data_dir = os.getenv ('DATA_DIR')                       # Folder for data that will be upload 
result_file = os.getenv('RESULT_FILE')                  # File where we save the compuation result
computation_ID_file = os.getenv('COMPUTATION_ID_FILE')  # File where we save the compuation ID

# path to the DATA_FOLDER
DATA_FOLDER = Path(f"{data_dir}")
app.config['DATA_FOLDER'] = DATA_FOLDER


# Check if there is a '.' in the file name (indicating an extension),
# also check if the extension is among those specified in ALLOWED_EXTENSIONS
ALLOWED_EXTENSIONS = {'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get('/')
def test():
    return jsonify({'message': 'TEST'}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_files = request.files.getlist('file')  # Get a list of uploaded files
    if not uploaded_files:
        return jsonify({'message': 'No files uploaded'}), 400

# check if the files are allowed 
    for file in uploaded_files:
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'message': 'Invalid file(s) upladed'}), 400
        file.save(os.path.join(app.config['DATA_FOLDER'], file.filename))

    return jsonify({'message': 'File received'}), 200


@app.route('/insert_ID', methods=['POST'])
def receive_ID():
    req_data = request.get_json()
    computation_ID = req_data["ID"]
    with open(computation_ID_file, "w") as f:
        f.write(str(computation_ID))
    # kill the process
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({'message': 'ID received'}), 200


if __name__ == '__main__':
    #app.run(ssl_context=('cert.pem', 'key.pem'), port=9443, host='0.0.0.0') 
    app.run(port=9443, host='0.0.0.0')