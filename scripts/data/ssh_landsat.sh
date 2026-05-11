#!/bin/bash
# Usage: bash scripts/data/ssh_landsat.sh <instance-id> [key-path]
set -euo pipefail

INSTANCE_ID="${1:?Usage: $0 <instance-id> [key-path]}"
KEY="${2:-$HOME/.ssh/landsat-debug-2.pem}"
REGION="us-west-2"

echo "Waiting for instance $INSTANCE_ID to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Instance: $INSTANCE_ID"
echo "IP:       $IP"
echo "Key:      $KEY"
echo ""
echo "Connecting... (user-data log: tail -f /var/log/landsat-worker.log)"
echo ""

ssh -o StrictHostKeyChecking=no -i "$KEY" "ec2-user@$IP"
