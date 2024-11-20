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
echo "Start Training!!!!"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
python3 src/train.py experiment=bird_classifier_ex  trainer=cpu +trainer.log_every_n_steps=1
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
#python3 src/train.py experiment=bird_classifier_ex +trainer.log_every_n_steps=1
aws s3 ls pytorch-model-emlov4-predictions/bird_classification --recursive
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "RUNNING aws s3 ls pytorch-model-emlov4-predictions/bird_classification --recursive"
python3 src/infer.py data=birddata inference=bird_infer_aws
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "START INFERNCING"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "RUNNING aws s3 ls pytorch-model-emlov4-predictions/bird_classification --recursive"
aws s3 ls pytorch-model-emlov4-predictions/bird_classification --recursive
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

#docker run --gpus all --shm-size=8g  --env AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} --env AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -it --entrypoint /bin/bash 575108919357.dkr.ecr.us-east-1.amazonaws.com/cr/food-classifier:latest
# python3 src/eval.py data=food100data evaluation=food_100_eval
# python3 src/infer.py  data=food100data inference=food_100_infer --cfg job
# docker run --gpus all --shm-size=4g  --env AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} --env AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -it --entrypoint /bin/bash 575108919357.dkr.ecr.us-east-1.amazonaws.com/cr/food-classifier:latest