dvc pull -r s3_store
ls data
apt update
apt install curl -y
apt install unzip -y
apt install vim -y
curl https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip
unzip awscliv2.zip
./aws/install --update
nvidia-smi
python3 src/train.py experiment=bird_classifier_ex  trainer=gpu +trainer.log_every_n_steps=1
#python3 src/train.py experiment=bird_classifier_ex +trainer.log_every_n_steps=1
aws s3 ls pytorch-model-emlov4-predictions/bird_classification --recursive
python3 src/infer.py data=birddata inference=bird_infer_aws
aws s3 ls pytorch-model-emlov4-predictions/bird_classification --recursive
