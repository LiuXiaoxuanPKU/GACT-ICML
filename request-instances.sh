SPOT_PRICE=1.18
INSTANCE_COUNT=4
LAUNCH_SPECIFICATION=file://g4dn.12xlarge.json

SPOT_INSTANCE_REQUEST_IDS=$(aws ec2 request-spot-instances --spot-price $SPOT_PRICE --instance-count $INSTANCE_COUNT --type "one-time" --launch-specification $LAUNCH_SPECIFICATION --query SpotInstanceRequests[*].SpotInstanceRequestId)
SPOT_INSTANCE_REQUEST_IDS=$(echo $SPOT_INSTANCE_REQUEST_IDS | tr -d []'"'' ' | tr ',' ' ')

INSTANCE_IDS=()
rm -f public_ip_addresses.tmp
for SPOT_INSTANCE_REQUEST_ID in ${SPOT_INSTANCE_REQUEST_IDS[*]}; do
  echo $SPOT_INSTANCE_REQUEST_ID

  INSTANCE_ID=null
  while [[ $INSTANCE_ID = null ]]; do
    INSTANCE_ID=$(aws ec2 describe-spot-instance-requests --filters Name=spot-instance-request-id,Values=$SPOT_INSTANCE_REQUEST_ID --query SpotInstanceRequests[0].InstanceId)
  done
  INSTANCE_ID=$(echo $INSTANCE_ID | tr -d '"')
  INSTANCE_IDS+=($INSTANCE_ID)
  echo $INSTANCE_ID

  PUBLIC_IP_ADDRESS=$(aws ec2 describe-instances --filters Name=instance-id,Values=$INSTANCE_ID --query Reservations[0].Instances[0].PublicIpAddress)
  PUBLIC_IP_ADDRESS=$(echo $PUBLIC_IP_ADDRESS | tr -d '"')
  echo $PUBLIC_IP_ADDRESS >> public_ip_addresses.tmp
done
cp public_ip_addresses.tmp IPs-$(date | tr :' ' -)

VOLUME_ID=vol-05f43b1ec6f46b873
DEVICE=/dev/sdf
aws ec2 attach-volume --region us-east-2 --volume-id $VOLUME_ID --instance-id ${INSTANCE_IDS[0]} --device $DEVICE
