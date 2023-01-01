python train.py \
--refine \
--weights /pretrained_model_path \
--ft3d_dataset_dir /dataset_path \
--kitti_dataset_dir /dataset_path \
--exp_path pt_refine \
--batch_size 16 \
--gpus 0,1,2,3 \
--num_epochs 10 \
--max_points 8192 \
--iters 32 \
--root ./ \