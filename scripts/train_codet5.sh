CUDA_VISIBLE_DEVICES=1,2 python main.py -ptm Salesforce/codet5-base --batch_size 8 --gpus 2
CUDA_VISIBLE_DEVICES=2 python evaluation.py -ptm 'Salesforce/codet5-base' \
-ckpt '/home/parraga/projects/_masters/code2test/checkpoints/Salesforce-codet5-base/best_model.ckpt'