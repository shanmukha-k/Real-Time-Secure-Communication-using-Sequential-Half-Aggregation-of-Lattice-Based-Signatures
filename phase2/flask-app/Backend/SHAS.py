import hashlib
import math
import os
from flask import Flask, jsonify, render_template, request # type: ignore
from flask_cors import CORS
import numpy as np # type: ignore

app = Flask(__name__)
CORS(app)

# Define parameters
q = 8380417  # Example prime modulus
ell = 160
k = 160
eta = 2
B=8777343
chunk_size = ell + k

# Norm threshold values
min_threshold = 1e8
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

def load_vector_from_file(filename):
    with open(filename, 'r') as file:
        return np.array(list(map(int, file.read().split())), dtype=int)

def load_matrix_from_file(filename):
    with open(filename, 'r') as file:
        return np.array([list(map(int, line.split())) for line in file], dtype=int)

def save_vector_to_file(vector, filename):
    with open(filename, 'w') as file:
        file.write(' '.join(map(str, vector)))

def save_matrix_to_file(matrix, filename):
    with open(filename, 'w') as file:
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')

def hash_function(u, L, z_prev):
    data = np.concatenate([u, L.flatten(), z_prev])
    return int(hashlib.sha256(data).hexdigest(), 16) % (2**32)  # Example hash function

def sample_y(ell, k):
    return np.random.randint(-2, 3, size=(ell + k,))

def rejection_sampling(z_i, challenge_term):
    norm_z_i = np.linalg.norm(z_i)
    if min_threshold <= norm_z_i <= max_threshold:
        return z_i  # Accept the z_i if within the threshold range
    else:
        return None  # Reject the z_i if outside the threshold range
    
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
    extracted_pairs = []
    
    # Iterate backwards from the last row
    for row_index in range(L_i.shape[0] - 1, L_i.shape[0] - count - 1, -1):
        if row_index < 0:
            break
        t_i = L_i[row_index, :-1]  # Extract t_i from the row, excluding the last column
        m_i = L_i[row_index, -1]   # Extract m_i from the last column of the row
        extracted_pairs.append((t_i, m_i))
    
    return extracted_pairs
def load_previous_signature_ver(filename):
    if not os.path.exists(filename):
        return jsonify({'error': f'Signature Doesnt exist:'}), 400

    with open(filename, 'r') as file:
        data = list(map(int, file.read().split()))
        print("Loaded the previous signature successfully")
    
    if len(data) < 2 * (ell + k):
        raise ValueError("Signature file does not contain enough data.")
    
    u_i = np.array(data[:ell + k], dtype=int)
    remaining_values = data[chunk_size:]
    z_chunks = [remaining_values[i:i + chunk_size] for i in range(0, len(remaining_values), chunk_size)]
    z_dict = {}
    for idx, chunk in enumerate(z_chunks[::-1]):  # Reverse order
        z_dict[f'z{idx+1}'] = chunk
    
    return u_i, z_dict

def is_invertible_mod_q(t_i, q):
    for element in t_i:
        if math.gcd(element, q) != 1:
            return False
    return True

def check_norm(z_dict, B):
    for zi_key, zi_values in z_dict.items():
        # Calculate the Euclidean norm ∥zi∥2
        norm_zi = np.linalg.norm(zi_values)

        # If ∥zi∥2 > B, return 0 for that zi
        if norm_zi > B:
            return jsonify({'error': f'Some value exceeds Norm:'}), 400

def load_previous_signature(filename):
    if not os.path.exists(filename):
        print("You are the first signer!!")
        print("Initialized the values with σ0 = (0, 0)")
        # Return initial values (0,0) for first signature
        return np.zeros(ell + k, dtype=int), np.zeros(ell + k, dtype=int)

    with open(filename, 'r') as file:
        data = list(map(int, file.read().split()))
        print("Loaded the previous signature successfully")
    
    if len(data) < 2 * (ell + k):
        raise ValueError("Signature file does not contain enough data.")
    
    u_i = np.array(data[:ell + k], dtype=int)
    z_i = np.array(data[ell + k:], dtype=int)
    
    return u_i, z_i

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_keys')
def generate_keys():
    A = np.random.randint(0, q, size=(ell + k, ell + k))
    sk, pk = key_generation(A, ell, k, eta, q)

    # Save the keys and matrix to files
    save_matrix_to_file(A, "../Data/A.txt")
    save_vector_to_file(sk, "../Data/secret_key.txt")
    save_vector_to_file(pk, "../Data/public_key.txt")
    print("Saved A, sk, pk to the respective files")

    return jsonify({
        'secret_key': sk.tolist(),
        'public_key': pk.tolist()
    })

@app.route('/save_message', methods=['POST'])
def save_message():
    data = request.get_json()
    message = data['message']
    try:
        with open('../Data/message.txt', 'w') as f:
            f.write(message)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/generate_signature')
def generate_signature():
    if not os.path.exists("../Data/secret_key.txt") or not os.path.exists("../Data/A.txt") or not os.path.exists("../Data/message.txt"):
        print("/////sk, A or m files are missing///////")
        return jsonify({'error': 'Required files are missing.'}), 400

    sk = load_vector_from_file("../Data/secret_key.txt")
    A = load_matrix_from_file("../Data/A.txt")
    m = load_vector_from_file("../Data/message.txt")
    print("Loaded sk, A, and m successfully")

    # Load previous signature
    try:
        u_i_prev, z_prev = load_previous_signature("../Data/signature.txt")
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Initialize L_i
    L_i = np.zeros((ell + k, ell + k), dtype=int) if not os.path.exists("../Data/L_i.txt") else load_matrix_from_file("../Data/L_i.txt")
    print("Loaded L_i successfully" if os.path.exists("../Data/L_i.txt") else "Initialized L_i")

    # Signing process
    while True:
        # Sample y_i
        y_i = sample_y(ell, k)

        # Compute Commitment u_i
        u_i = np.dot(A, y_i) % q

        # Update u_i
        u_i_current = (u_i_prev + u_i) % q

        # Compute challenge c_i
        c_i = hash_function(u_i_current, L_i, z_prev)

        # Compute z_i
        z_i = (c_i * sk + y_i) % q

        # Use rejection sampling to ensure z_i is valid
        z_i_valid = rejection_sampling(z_i, c_i * sk)

        if z_i_valid is not None:  # Check if z_i was accepted
            # Update L_i with the new t_i and m_i
            t_i = np.dot(A, sk) % q
            L_i[:-1, :] = L_i[1:, :]  # Shift rows of L_i up
            L_i[-1, :] = t_i  # Add new t_i as the last row
            L_i[-1, -1] = m[0]  # Update the last element of the last row with m[0]

            # Construct signature
            signature = np.concatenate([u_i_current,z_prev,z_i_valid])

            # Save updated values
            save_vector_to_file(u_i_current, "../Data/u_i.txt")
            save_vector_to_file(z_i_valid, "../Data/z_i.txt")
            save_matrix_to_file(L_i, "../Data/L_i.txt")
            save_vector_to_file(signature, "../Data/signature.txt")
            print("Saved u_i, z_i, L_i, and signature to the respective files")
            break  # Exit loop after successful signature generation

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
      # Load the necessary files for verification
    if not os.path.exists("../Data/A.txt") or not os.path.exists("../Data/message.txt"):
        return jsonify({'error': 'Required files are missing.'}), 400

    A = load_matrix_from_file("../Data/A.txt")
    m = load_vector_from_file("../Data/message.txt")
    print("Loaded A and m successfully")
    N = len(m) 
    print(N)

        # Load previous values (u_i_prev and z_prev)
    try:
        u_i,z_dict = load_previous_signature_ver("../Data/signature.txt")
        print(u_i)
        print(z_dict)
        L_i = load_matrix_from_file("../Data/L_i.txt")
        print("Loaded previous signature and L_i successfully")
    except Exception as e:
        return jsonify({'error': f'Failed to load previous values: {str(e)}'}), 400
        
    # Example: Extract the pair from row 2 (third row, 0-based index)
    count = 1
    extracted_pairs = extract_from_end(L_i, count)

    for i, (t_i, m_i) in enumerate(extracted_pairs):
        print(f"Extracted t_i from row {L_i.shape[0] - 1 - i}: {t_i}")
        print(f"Extracted m_i from row {L_i.shape[0] - 1 - i}: {m_i}")
    
    z0=np.ones(ell + k, dtype=int)

    for i, (t_i, m_i) in enumerate(extracted_pairs):
        if not is_invertible_mod_q(t_i, q):
            print(f"t_i from pair {i} is not invertible.")
            return jsonify({'verification': 'Failed', 'reason': 'Non-invertable element found'}), 400
        else:
            print(f"t_i from pair {i} is invertible.")

    check_norm(z_dict,B)

    # Verification successful
    return jsonify({'verification': 'Success'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, ssl_context=("/etc/ssl/flaskapp/flaskapp.crt", "/etc/ssl/flaskapp/flaskapp.key"))
