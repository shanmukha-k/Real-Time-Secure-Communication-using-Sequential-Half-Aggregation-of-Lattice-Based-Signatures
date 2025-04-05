import hashlib
import math
import os
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://sarathmdb:sarathmdb@finalyearproject.hl6cr.mongodb.net/?retryWrites=true&w=majority&appName=finalyearproject")
db = client['signature_database']  # Database name
collection_keys = db['keys']  # Collection for storing keys
collection_messages = db['messages']  # Collection for storing messages
collection_signatures = db['signatures']  # Collection for storing signatures
collection_Li = db['Li']  # New collection for storing L_i separately

# Define parameters
q = 8380417  # Example prime modulus
ell = 160
k = 160
eta = 2
B = 8777343
chunk_size = ell + k

# Norm threshold values
min_threshold = 0.7e8
max_threshold = 1.05e8

def sample_S_l_k_eta(ell, k, eta):
    return np.random.randint(-eta, eta + 1, size=(ell + k,))

def is_invertible(x, q):
    return np.gcd(int(x), q) == 1

def has_invertible_coeff(t, q):
    return any(is_invertible(coeff, q) for coeff in t)

def matrix_vector_mod_q(A, s, q):
    return (np.dot(A, s) % q).astype(int)

def key_generation(A, ell, k, eta, q):
    t = np.zeros(A.shape[1], dtype=int)
    while not has_invertible_coeff(t, q):
        s = sample_S_l_k_eta(ell, k, eta)
        t = matrix_vector_mod_q(A, s, q)
    return s, t

def save_to_db(collection, data, filter_query=None):
    try:
        if filter_query:
            result = collection.update_one(filter_query, {"$set": data}, upsert=True)
            return result.upserted_id is not None or result.modified_count > 0  # Works for update_one()
        else:
            result = collection.insert_one(data)
            return result.inserted_id is not None  # Works for insert_one()
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return False

    
def load_from_db(collection, required_keys, query=None):
    result = collection.find_one(query or {}, sort=[("_id", -1)])

    if not result:
        return None

    return {key: result[key] for key in required_keys if key in result}

def hash_function(u, L, z_prev):
    data = np.concatenate([u, L.flatten(), z_prev])
    return int(hashlib.sha256(data).hexdigest(), 16) % (2**32)  # Example hash function

def sample_y(ell, k):
    return np.random.randint(-2, 3, size=(ell + k,))

def rejection_sampling(z_i):
    norm_z_i = np.linalg.norm(z_i)
    print(f"Norm of z_i: {norm_z_i}")

    if min_threshold <= norm_z_i <= max_threshold:
        print(f"z_i accepted: {norm_z_i} is within the threshold range.")
        return True

    else:
        print(f"z_i rejected: {norm_z_i} is outside the threshold range ({min_threshold}, {max_threshold}).")
        return False

def rejection_sampling_ver(z_i, challenge_term):
    norm_z_i = np.linalg.norm(z_i)
    print(f"Norm of z_i: {norm_z_i}")

    # Check if norm_z_i lies within the specified threshold
    if min_threshold <= norm_z_i <= max_threshold:
        print(f"z_i accepted: {norm_z_i} is within the threshold range.")
        return True  # Accept the z_i if within the threshold range
    else:
        print(f"z_i rejected: {norm_z_i} is outside the threshold range ({min_threshold}, {max_threshold}).")
        return False  # Reject the z_i if outside the threshold range

def extract_from_end(L_i, count):
    if isinstance(L_i, dict):  
        L_i = np.array(L_i.get('L_i', []))  # Convert to numpy array safely

    if not isinstance(L_i, np.ndarray) or L_i.size == 0:
        return []  # Return empty if L_i is invalid

    rows, cols = L_i.shape
    count = min(count, rows)  # Prevent index errors

    extracted_pairs = []
    for row_index in range(rows - count, rows):
        t_i = L_i[row_index, :-1]  # Extract all columns except the last
        m_i = L_i[row_index, -1]   # Extract the last column
        extracted_pairs.append((t_i, m_i))
    
    return extracted_pairs

def load_previous_signature():
    previous_signature = collection_signatures.find_one({}, sort=[("_id", -1)])  # Get the latest signature

    if not previous_signature:
        print("You are the first signer!!")
        print("Initialized the values with σ0 = (0, 0)")
        return np.zeros(ell + k, dtype=int), np.zeros(ell + k, dtype=int)

    try:
        u_i = np.array(previous_signature["u_i"], dtype=int)
        z_i = np.array(previous_signature["z_i"], dtype=int)
        print("Loaded the previous signature successfully")
        return u_i, z_i
    except KeyError:
        raise ValueError("Previous signature data is incomplete.")

def load_previous_signature_ver():
    """Load the latest signature from MongoDB."""
    previous_signature = collection_signatures.find_one({}, sort=[("_id", -1)])  # Get the latest signature

    if not previous_signature:
        raise ValueError("Signature not found in MongoDB.")

    try:
        u_i = np.array(previous_signature["u_i"], dtype=int)
        z_i = np.array(previous_signature["z_i"], dtype=int)
        print("Loaded the previous signature successfully")
        return u_i, z_i
    except KeyError:
        raise ValueError("Previous signature data is incomplete.")

def is_invertible_mod_q(t_i, q):
    for element in t_i:
        if math.gcd(element, q) != 1:
            return False
    return True

def check_norm(z_dict, B):
    if isinstance(z_dict, np.ndarray):  # If z_dict is a numpy array, use it directly
        z_values = [z_dict]
    elif isinstance(z_dict, dict):  # If it's a dictionary, get its values
        z_values = z_dict.values()
    else:
        return jsonify({'error': 'Invalid z_dict format'}), 400  # Handle unexpected cases

    for zi_values in z_values:
        norm_zi = np.linalg.norm(zi_values)
        if norm_zi > B:
            return jsonify({'error': f'Some value exceeds Norm:'}), 400

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_keys')
def generate_keys():
    A = np.random.randint(0, q, size=(ell + k, ell + k))
    sk, pk = key_generation(A, ell, k, eta, q)

    key_data = {"_id": "keys", "A": A.tolist(), "secret_key": sk.tolist(), "public_key": pk.tolist()}
    if not save_to_db(collection_keys, key_data, {"_id": "keys"}):
        return jsonify({'error': 'Failed to save keys'}), 500

    return jsonify({'secret_key': sk.tolist(), 'public_key': pk.tolist()})

# message = data['message']

@app.route('/save_message', methods=['POST'])
def save_message():
    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    filter_query = {"_id": "message"}
    if not save_to_db(collection_messages, {"_id": "message", "message": message}, filter_query):
        return jsonify({"success": False, "error": "Failed to save or update message."})

    return jsonify({"success": True})

@app.route('/generate_signature')
def generate_signature():
    # Load keys, A, and message from MongoDB
    key_data = load_from_db(collection_keys, ['A', 'secret_key', 'public_key'], {"_id": "keys"})
    message_data = load_from_db(collection_messages, ['message'], {"_id": "message"})
    
    if not key_data or not message_data:
        print("/////sk, A, or m are missing///////")
        return jsonify({'error': 'Required data is missing.'}), 400

    sk = np.array(key_data['secret_key'])
    A = np.array(key_data['A'])
    m = np.array([int(message_data['message'])])
    print("Loaded sk, A, and m successfully")

    # Load previous signature
    u_i_prev, z_prev = load_previous_signature()

    # Load `L_i` separately
    Li_data = load_from_db(collection_Li, ['L_i'], {"_id": "L_i"})
    L_i = np.zeros((ell + k, ell + k), dtype=int) if not Li_data else np.array(Li_data['L_i'])

    print("Loaded L_i successfully" if Li_data else "Initialized L_i")

    # Signing process
    while True:
        y_i = sample_y(ell, k)
        u_i = np.dot(A, y_i) % q
        u_i_current = (u_i_prev + u_i) % q
        c_i = hash_function(u_i_current, L_i, z_prev)
        z_i = (c_i * sk + y_i) % q
        z_i_valid = rejection_sampling(z_i)

        if z_i_valid is not None:
            t_i = np.dot(A, sk) % q
            L_i[:-1, :] = L_i[1:, :]
            L_i[-1, :] = t_i
            L_i[-1, -1] = m[0]
            signature = np.concatenate([u_i_current, z_prev, z_i])

            # Update L_i in the database instead of inserting a new document
            Li_data = {"L_i": L_i.tolist()}
            if not save_to_db(collection_Li, {"_id": "L_i", "L_i": L_i.tolist()}, {"_id": "L_i"}):  # Use a filter to update L_i
                return jsonify({'error': 'Failed to save or update L_i in the database.'}), 500

            # Save signature data separately
            signature_data = {
                "u_i": u_i_current.tolist(),
                "z_i": z_i.tolist(),
                "signature": signature.tolist()
            }
            if not save_to_db(collection_signatures, signature_data, {"_id": "signatures"}):
                return jsonify({'error': 'Failed to save signature to the database.'}), 500

            print("Saved L_i and signature to MongoDB successfully.")
            break

    return jsonify({
        'signature': signature.tolist()
    })

@app.route('/verify_signature', methods=['POST', 'OPTIONS'])
def verify_signature():
    if request.method == 'OPTIONS':
        return '', 200  # Respond to preflight requests

    # Log the incoming data for debugging purposes
    data = request.json
    print("Incoming request data:", data)

    # Load necessary data from MongoDB
    try:
        A = load_from_db(collection_keys, ['A'], {"_id": "keys"})
        m = load_from_db(collection_messages, ['message'], {"_id": "message"})
        print("Loaded A and m successfully")
    except ValueError as e:
        return jsonify({'error': f'Failed to load data from MongoDB: {str(e)}'}), 400

    # Load previous values (u_i_prev and z_prev)
    try:
        u_i, z_dict = load_previous_signature_ver()
        print(u_i)
        print(z_dict)
        L_i_dict = load_from_db(collection_Li, ['L_i'], {"_id": "L_i"})
        L_i = L_i_dict.get('L_i', [])  # Extract only the list part

        print("Loaded previous signature and L_i successfully")
    except ValueError as e:
        return jsonify({'error': f'Failed to load previous values: {str(e)}'}), 400
        
    # Example: Extract the pair from row 2 (third row, 0-based index)
    count = 1
    extracted_pairs = extract_from_end(L_i, count)

    for i, (t_i, m_i) in enumerate(extracted_pairs):
        print(f"Extracted t_i from row {L_i.shape[0] - 1 - i}: {t_i}")
        print(f"Extracted m_i from row {L_i.shape[0] - 1 - i}: {m_i}")
    
    z0 = np.ones(ell + k, dtype=int)

    for i, (t_i, m_i) in enumerate(extracted_pairs):
        if not is_invertible_mod_q(t_i, q):
            print(f"t_i from pair {i} is not invertible.")
            return jsonify({'verification': 'Failed', 'reason': 'Non-invertible element found'}), 400
        else:
            print(f"t_i from pair {i} is invertible.")

    check_norm({"z_i": z_dict}, B)  # Wrap the numpy array in a dictionary

    # Verification successful
    return jsonify({'verification': 'Success'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, ssl_context=("/etc/ssl/flaskapp/flaskapp.crt", "/etc/ssl/flaskapp/flaskapp.key"))
