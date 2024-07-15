# Secure Communication

A secure communication channel using sequential half-aggregation of lattice-based signatures. This project ensures the integrity, authenticity, and confidentiality of messages exchanged between users.

## Features

- **Key Generation**: Generate RSA public and private keys.
- **Message Signing**: Sign messages with private keys.
- **Signature Aggregation**: Combine multiple signatures into a single compact signature.
- **Signature Verification**: Verify signatures using public keys.
- **Message Encryption**: Encrypt messages to ensure confidentiality.
- **Message Decryption**: Decrypt received messages.

## Technologies Used

- **Backend**: Node.js, Express.js
- **Frontend**: HTML, CSS, JavaScript
- **WebSockets**: Socket.io
- **Cryptography**: Node.js crypto module

## Getting Started

### Prerequisites

- Node.js and npm installed on your machine.

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/Secure-Communication.git
    cd Secure-Communication
    ```

2. Install dependencies:
    ```sh
    npm install
    ```

### Running the Application

1. Start the server:
    ```sh
    node server.js
    ```

2. Open `index.html` in your web browser.

### Usage

1. **Generate Key**: Click the "Generate Key" button to create your RSA key pair.
2. **Sign Message**: Enter a message and click "Sign Message" to sign it with your private key.
3. **Aggregate Signatures**: Combine multiple signatures into one.
4. **Verify Signature**: Verify the validity of a received signature.
5. **Encrypt Message**: Encrypt your message before sending it.
6. **Decrypt Message**: Decrypt a received message to read its contents.

### Project Structure

