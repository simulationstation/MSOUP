#!/bin/bash
# Script to copy Globus config to remote server

REMOTE_HOST="172.239.51.121"
REMOTE_USER="primary"

echo "Copying SSH key to remote server (enter password when prompted)..."
ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}

echo ""
echo "Now copying Globus configuration..."

# Copy .globus directory
echo "Copying ~/.globus..."
scp -r ~/.globus ${REMOTE_USER}@${REMOTE_HOST}:~/

# Copy .globusonline directory
echo "Copying ~/.globusonline..."
scp -r ~/.globusonline ${REMOTE_USER}@${REMOTE_HOST}:~/

# Copy Globus Connect Personal tarball (easier to set up fresh on new machine)
echo "Copying Globus Connect Personal tarball..."
scp ~/MSOUP/globusconnectpersonal-latest.tgz ${REMOTE_USER}@${REMOTE_HOST}:~/

echo ""
echo "Done! On the remote server, run:"
echo "  cd ~"
echo "  tar xzf globusconnectpersonal-latest.tgz"
echo "  cd globusconnectpersonal-*"
echo "  ./globusconnectpersonal -setup"
