TASK_NAME="sst2"
CONFIG="config/default.yaml"

for lr in 1e-5; do
python run_glue.py \
  --model_name_or_path bert-large-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 128 \
  --learning_rate $lr \
  --num_train_epochs 1 \
  --seed 42 \
  --actnn \
  --opt_level L1 \
  --output_dir log/$TASK_NAME/L2.1/ \
  --pad_to_max_length \
  --get_speed
done