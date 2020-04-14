CUDA_VISIBLE_DEVICES=0 python train.py \
    --data_dir "video_prediction/datasets/google_push" \
    --exp_name "push_1" \
    --dataset_type "google_push" \
    --batch_size 16 \
    --model_type "pixel_advection" \
    --max_epoch 100