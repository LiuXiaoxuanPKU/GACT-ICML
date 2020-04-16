PEM=~/.ssh/yu.pem
SSH="ssh -i $PEM -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"

IPS=()
for IP in $(cat public_ip_addresses.tmp); do
  IPS+=($IP)
done

MASTER=${IPS[0]}
$SSH ubuntu@$MASTER "sudo mount /dev/nvme2n1 /store"
for IP in ${IPS[*]:1}; do
  $SSH ubuntu@$IP "sshfs ubuntu@$MASTER:/store /store -o Compression=no -o IdentityFile=~/.ssh/yu.pem -o ssh_command='$SSH'"
done

RANK=0
NPROC_PER_NODE=4
ARCH=resnet18
ARGS="-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True"
for IP in ${IPS[*]}; do
# $SSH ubuntu@$IP "cd RN50v1.5; bash -i dist-train ${#IPS[*]} $RANK $MASTER $NPROC_PER_NODE $ARCH \"$ARGS\""
  $SSH ubuntu@$IP "tmux new-session -d 'cd RN50v1.5; bash -ix dist-train ${#IPS[*]} $RANK $MASTER $NPROC_PER_NODE $ARCH \"$ARGS\"'"
  RANK=$((RANK + 1))
done
