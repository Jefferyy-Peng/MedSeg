deepspeed --master_port=24999 train_ds.py \
  --version="./model/llava/ckpt" \
  --dataset_dir='/data/leo/drive1/Datasets/vis' \
  --vision_pretrained="/model/segment_anything/ckpt/sam_vit_h_4b8939.pthM" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="lisa-7b"