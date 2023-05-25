#!/bin/sh
# source /shared/virtualenvs/cnn/bin/activate


optim="adamw sgd adadelta"  # Optimizer to use
batch_sizes="16 32"  # Batch sizes to iterate over
learning_rates="0.001 0.0001"  # Learning rates to iterate over
epochs=200  # Number of epochs
save_model_path="results"  # Path to save the trained models
seed=42  # Random seed
log_interval=20  # Log interval during training

# Iterate over batch sizes
for optim in $optim
do
    for batch_size in $batch_sizes
    do
        # Iterate over learning rates
        for learning_rate in $learning_rates
        do
            echo "Training with batch size: $batch_size, learning rate: $learning_rate, optimizer: $optim"

            # Launch training command
            python3 scripts/train.py --batch-size=$batch_size --optim=$optim \
                --epochs=$epochs --lr=$learning_rate --save-model-path=$save_model_path \
                --seed=$seed --log-interval=$log_interval

            echo "Training completed with batch size: $batch_size, learning rate: $learning_rate, optimizer: $optim"
        done
    done
done

