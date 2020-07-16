PEM=~/.ssh/yu.pem
SSH="ssh -i $PEM -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"
SCP="scp -i $PEM -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"
IPFILE=$1

IPS=()
for IP in $(cat $IPFILE); do
  IPS+=($IP)
done

MASTER=${IPS[0]}
$SSH ubuntu@$MASTER "sudo mount /dev/xvdf /store"
#for IP in ${IPS[*]:1}; do
#  $SSH ubuntu@$IP "sshfs ubuntu@$MASTER:/store /store -o Compression=no -o IdentityFile=~/.ssh/yu.pem -o ssh_command='$SSH'"
#done

RANK=0
NPROC_PER_NODE=8
ARCH=resnet50
ID=resnet56_a32w32g32
# ARGS="-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True"
ARGS="-c quantize --qa=False --qw=False --qg=False"
for IP in ${IPS[*]}; do
  $SCP dist-train ubuntu@$IP:RN50v1.5/
  $SSH ubuntu@$IP "tmux new-session -d 'cd RN50v1.5; bash -ix dist-train 1 0 127.0.0.1 $NPROC_PER_NODE $ARCH \"$ARGS\" $ID'"
  RANK=$((RANK + 1))
done