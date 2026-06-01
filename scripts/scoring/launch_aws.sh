#!/usr/bin/env bash
# Launch a Stage-8 GPU box: create a one-off SSH-from-my-IP SG, then run-instances with
# AZ-hop on InsufficientInstanceCapacity. Prints INSTANCE_ID / SG_ID / PUBLIC_IP for the caller.
# Usage: bash launch_aws.sh <instance_type> <sg_name>
# e.g.   bash launch_aws.sh g6.xlarge     iccd-eeww-small
#        bash launch_aws.sh g6.12xlarge   iccd-eeww-24b
set -euo pipefail
ITYPE="$1"; SGNAME="$2"
AMI="ami-012ba162b9cd2729c"        # DL PyTorch 2.7 / Ubuntu 22.04 (L4/L40S)
KEY="anthropic-fellows-key"
REGION="us-east-1"
VPC="vpc-003b5ab4402aba736"

MYIP="$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')"
echo "my ip: $MYIP"

SG_ID="$(aws ec2 create-security-group --group-name "$SGNAME" \
  --description "ICCD Stage-8 ephemeral SSH" --vpc-id "$VPC" --region "$REGION" \
  --query GroupId --output text)"
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 \
  --cidr "${MYIP}/32" --region "$REGION" >/dev/null
echo "SG_ID=$SG_ID  (ssh from ${MYIP}/32)"

IID=""
for AZ in a b c f d e; do
  echo "trying us-east-1${AZ} ..."
  SUBNET="$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=availability-zone,Values=${REGION}${AZ}" "Name=vpc-id,Values=${VPC}" \
    --query 'Subnets[0].SubnetId' --output text 2>/dev/null || echo None)"
  [ "$SUBNET" = "None" ] && { echo "  no subnet in ${AZ}"; continue; }
  if IID="$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" \
      --instance-type "$ITYPE" --key-name "$KEY" --security-group-ids "$SG_ID" \
      --subnet-id "$SUBNET" \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${SGNAME}}]" \
      --query 'Instances[0].InstanceId' --output text 2>/dev/null)"; then
    echo "launched $IID in us-east-1${AZ}"; break
  else
    echo "  capacity/err in ${AZ}, hopping"; IID=""
  fi
done
[ -z "$IID" ] && { echo "FAILED: no capacity in any AZ"; aws ec2 delete-security-group --group-id "$SG_ID" --region "$REGION"; exit 1; }

echo "waiting for instance-status-ok ..."
aws ec2 wait instance-status-ok --instance-ids "$IID" --region "$REGION"
PUBIP="$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "READY  INSTANCE_ID=$IID  SG_ID=$SG_ID  PUBLIC_IP=$PUBIP"
