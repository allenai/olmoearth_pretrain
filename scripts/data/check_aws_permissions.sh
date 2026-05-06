#!/bin/bash
# Quick check of AWS permissions needed for the Landsat EC2 launcher.
# Usage: bash scripts/data/check_aws_permissions.sh [bucket-name]
set -uo pipefail

BUCKET="${1:-olmoearth-landsat-staging}"
REGION="us-west-2"
PASS=0
FAIL=0

check() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "  ✓ $label"
        PASS=$((PASS+1))
    else
        echo "  ✗ $label"
        FAIL=$((FAIL+1))
    fi
}

echo "=== AWS Permission Check for Landsat EC2 Launcher ==="
echo "Bucket: $BUCKET | Region: $REGION"
echo ""

# Identity
echo "[Identity]"
IDENTITY=$(aws sts get-caller-identity --output text 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "  ✓ Authenticated as: $IDENTITY"
    PASS=$((PASS+1))
else
    echo "  ✗ No AWS credentials found. Export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"
    echo ""
    echo "RESULT: Cannot proceed without credentials."
    exit 1
fi
echo ""

# S3 staging bucket
echo "[S3 Staging Bucket: $BUCKET]"
check "s3:ListBucket" aws s3 ls "s3://$BUCKET/" --region "$REGION"
check "s3:PutObject"  bash -c "echo test | aws s3 cp - s3://$BUCKET/_perm_test --region $REGION"
check "s3:GetObject"  aws s3 cp "s3://$BUCKET/_perm_test" /dev/null --region "$REGION"
check "s3:DeleteObject" aws s3 rm "s3://$BUCKET/_perm_test" --region "$REGION"
echo ""

# S3 requester-pays (usgs-landsat)
echo "[S3 Requester-Pays: usgs-landsat]"
check "s3:ListBucket (requester-pays)" aws s3 ls s3://usgs-landsat/collection02/level-1/standard/oli-tirs/2024/ --request-payer requester --region "$REGION"
echo ""

# EC2
echo "[EC2]"
check "ec2:DescribeInstances" aws ec2 describe-instances --max-results 5 --region "$REGION"
AMI=$(aws ssm get-parameter --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64" --region "$REGION" --query "Parameter.Value" --output text 2>/dev/null)
if [ -n "$AMI" ]; then
    echo "  ✓ SSM AMI lookup: $AMI"
    PASS=$((PASS+1))
    check "ec2:DescribeImages (AMI $AMI)" aws ec2 describe-images --image-ids "$AMI" --region "$REGION"
else
    echo "  ✗ SSM AMI lookup failed (need ssm:GetParameter or pass --ami manually)"
    FAIL=$((FAIL+1))
fi
echo ""

# Summary
TOTAL=$((PASS+FAIL))
echo "=== Summary: $PASS/$TOTAL passed ==="
if [ $FAIL -eq 0 ]; then
    echo "All checks passed. Ready to launch."
else
    echo "$FAIL check(s) failed. Fix permissions before launching."
fi
