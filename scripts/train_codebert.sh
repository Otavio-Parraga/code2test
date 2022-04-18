CUDA_VISIBLE_DEVICES=0,1 python main.py -ptm microsoft/codebert-base --batch_size 8 --gpus 2
CUDA_VISIBLE_DEVICES=1 python evaluation.py -ptm 'microsoft/codebert-base' \
-ckpt '/home/parraga/projects/_masters/code2test/checkpoints/microsoft-codebert-base/best_model.ckpt'