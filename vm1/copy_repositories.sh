# Move to $HOME
cd $HOME
# Clone just LFS pointers for Med-R1 models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/yuxianglai117/Med-R1
# Pull only the Med-R1/VQA_CT model.
# This should take about 6 minutes with the basic Colab runtime.
cd Med-R1/ && git lfs pull --include="VQA_CT"
# Pull the OmniMedVQA dataset.
wget https://huggingface.co/datasets/foreverbeliever/OmniMedVQA/resolve/main/OmniMedVQA.zip
unzip OmniMedVQA
rm OmniMedVQA.zip
# Install packages + get JSON files.
mkdir gh
cd gh && git clone https://github.com/Yuxiang-Lai117/Med-R1.git
