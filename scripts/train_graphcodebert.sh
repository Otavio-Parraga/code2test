CUDA_VISIBLE_DEVICES=2,3 python main.py -ptm microsoft/graphcodebert-base --batch_size 8 --gpus 2
CUDA_VISIBLE_DEVICES=3 python evaluation.py -ptm 'microsoft/graphcodebert-base' \
-ckpt '/home/parraga/projects/_masters/code2test/checkpoints/microsoft-graphcodebert-base/best_model.ckpt'