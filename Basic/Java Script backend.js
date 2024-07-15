const express = require('express');
const bodyParser = require('body-parser');
const crypto = require('crypto');
const { generateKeyPairSync, sign, verify } = require('crypto');
const app = express();

app.use(bodyParser.json());

// Generate key pair
app.post('/register', (req, res) => {
  const { publicKey, privateKey } = generateKeyPairSync('rsa', {
    modulusLength: 2048,
    publicKeyEncoding: { type: 'pkcs1', format: 'pem' },
    privateKeyEncoding: { type: 'pkcs1', format: 'pem' },
  });

  // Save publicKey to database
  // ...

  res.json({ publicKey });
});

// Authenticate users
app.post('/authenticate', (req, res) => {
  const { message, signature, publicKey } = req.body;

  const isVerified = verify(
    'sha256',
    Buffer.from(message),
    {
      key: publicKey,
      padding: crypto.constants.RSA_PKCS1_PSS_PADDING,
    },
    Buffer.from(signature, 'base64')
  );

  res.json({ isVerified });
});

const server = app.listen(3000, () => {
  console.log('Server is running on port 3000');
});

// WebSocket setup
const io = require('socket.io')(server);

io.on('connection', (socket) => {
  socket.on('message', (msg) => {
    // Broadcast the message to all clients
    io.emit('message', msg);
  });
});
