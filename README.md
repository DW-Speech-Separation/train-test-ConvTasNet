# train-test-ConvTasNet

### Experiment name
<Train/Test> <Dataset(CallFriend/All)> <tipo de loss> <tipo de embedding>


# Train

1. python setup.py
2. cd src/models/
3. !python train_model.py --experiment_name "Entrenamiento modelo original con dataset CallFriend con loss original SI-SDR, sin embeddign" --tags "Modelo original SIN embedding" --save_best_model "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_CallFriend_Dataset_Sum_Loss/" --default_root_dir "/content/drive/Shareddrives/TG-Separación-Fuentes/code/Checkpoints-separation-models/ConvTasnet/best_model_100_epochs_embedding_Wav2Vect_CallFriend_Dataset_Sum_Loss/" --epochs 2 --num_workers 2



