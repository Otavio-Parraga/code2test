# CUDA_VISIBLE_DEVICES=1,2 python main.py -ptm Salesforce/codet5-base --batch_size 8 --gpus 2 --prefix --output_dir ./checkpoints/prefix 
# CUDA_VISIBLE_DEVICES=2 python evaluation.py -ptm 'Salesforce/codet5-base' \
# -ckpt '/home/parraga/projects/_masters/code2test/checkpoints/prefix/Salesforce-codet5-base/best_model.ckpt'

# CUDA_VISIBLE_DEVICES=1,2 python main.py -ptm microsoft/codebert-base --batch_size 8 --gpus 2 --prefix --output_dir ./checkpoints/prefix 
# CUDA_VISIBLE_DEVICES=2 python evaluation.py -ptm 'microsoft/codebert-base' \
# -ckpt '/home/parraga/projects/_masters/code2test/checkpoints/prefix/microsoft-codebert-base/best_model.ckpt'

CUDA_VISIBLE_DEVICES=2,3 python main.py -ptm microsoft/graphcodebert-base --batch_size 8 --gpus 2 --prefix --output_dir ./checkpoints/prefix 
CUDA_VISIBLE_DEVICES=3 python evaluation.py -ptm 'microsoft/graphcodebert-base' \
-ckpt '/home/parraga/projects/_masters/code2test/checkpoints/prefix/microsoft-graphcodebert-base/best_model.ckpt'