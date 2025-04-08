# Secure Communication

A secure communication channel using sequential half-aggregation of lattice-based signatures. This project ensures the integrity, authenticity, and confidentiality of messages exchanged between users.


![Project Demo](https://github.com/user-attachments/assets/20947fe2-b120-4514-9eea-1022a532a65f)
![](https://github.com/user-attachments/assets/551b90fb-b158-46c5-82a5-3e0fd0da8c0e)


## Features

- **Key Generation**: Generate RSA public and private keys.
- **Message Signing**: Sign messages with private keys.
- **Signature Aggregation**: Combine multiple signatures into a single compact signature.
- **Signature Verification**: Verify signatures using public keys.
- **Message Encryption**: Encrypt messages to ensure confidentiality.
- **Message Decryption**: Decrypt received messages.

## Technologies Used

- **Backend**: AWS EC2 Instance
- **Frontend**: HTML, CSS, JavaScript
- **Cryptography**: crypto module

## Getting Started

### Prerequisites

- Create a Ec2 Instance with ssh, http and https port

### Installation

1. Clone the repository to the Ec2 instance:
    ```sh
    git clone https://github.com/shanmukha-k/Real-Time-Secure-Communication-using-Sequential-Half-Aggregation-of-Lattice-Based-Signatures.git
    cd Real-Time-Secure-Communication-using-Sequential-Half-Aggregation-of-Lattice-Based-Signatures
    ```

2. Install dependencies:
    ```sh
    install required modules
    ```

### Running the Application

1. Start the server:
    ```sh
    sudo python3 SHAS.py
    ```

2. Open `index.html` in your web browser using public ip of your Ec2 instance.

### Usage

1. **Generate Key**: Click the "Generate Key" button to create your RSA key pair.
2. **Sign Message**: Enter a message and click "Sign Message" to sign it with your private key.
3. **Aggregate Signatures**: Combine multiple signatures into one.
4. **Verify Signature**: Verify the validity of a received signature.
5. **Encrypt Message**: Encrypt your message before sending it.
6. **Decrypt Message**: Decrypt a received message to read its contents.

### Project Structure


### Contributing

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a pull request.

### License

This project is licensed under a custom license. See `LICENSE` for more information.


### Contact

- **Shanmukha Sai Kotamsetti** - [shanmukhasai020504@gmail.com](mailto:shanmukhasai020504@gmail.com)
- **Sarath Chandra Edubelli** - [sarathchandraedubelli@gmail.com](mailto:sarathchandraedubelli@gmail.com)
- **Project Link**: [https://github.com/shanmukha-k/Secure Communication](https://github.com/shanmukha-k/Real-Time-Secure-Communication-using-Sequential-Half-Aggregation-of-Lattice-Based-Signatures)

#### Note: The project is just started with its principles and prerequisite knowledge.

