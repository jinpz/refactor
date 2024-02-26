conda create -n refactor python=3.7
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install scipy
pip install --no-index torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.6.0+cu101.html
pip install torch_geometric==1.7.0
pip install graphviz
pip install pandas
pip install https://github.com/Lightning-AI/pytorch-lightning/archive/refs/tags/0.10.0.zip
pip install GitPython
