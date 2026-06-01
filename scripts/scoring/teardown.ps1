# Terminate a Stage-8 GPU instance and delete its security group.
# Usage: ./teardown.ps1 -InstanceId i-xxxx -SgId sg-xxxx
param(
  [Parameter(Mandatory = $true)][string]$InstanceId,
  [Parameter(Mandatory = $true)][string]$SgId
)
aws ec2 terminate-instances --instance-ids $InstanceId
aws ec2 wait instance-terminated --instance-ids $InstanceId
aws ec2 delete-security-group --group-id $SgId
Write-Output "torn down: $InstanceId, $SgId"
