# How to setup environment for running scGPT on Sherlock

// configure conda environment first; if you have conda installed, skip this block

<code> mkdir -p ~/miniconda3 </code>

<code> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh </code>

<code> bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 </code>

<code> rm -rf ~/miniconda3/miniconda.sh </code>

<code> ~/miniconda3/bin/conda init bash </code>

<code> ~/miniconda3/bin/conda init zsh </code>

// re-login, move (e.g. scp) the attached scgpt.yml to your folder

<code> conda env create --name scgpt --file=scgpt.yml </code>

<code> conda activate scgpt </code>

<code> pip3 install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html </code>

<code> pip3 install flash-attn==1.0.4 </code>  //make sure you got GPU allocated when run this command, e.g. hit a developer session <code> srun -p dev --time=2:00:00 --mem=16GB --gpus=1 --pty bash </code>

// Sherlock for light workload (fine-tune) job

<code> srun -p gpu --time=2-00:00:00 --mem=32GB --gpus=1 -C GPU_SKU:RTX_2080Ti --pty bash </code>  //on my end, the server node is sh03-12n16

<code> jupyter notebook --port 9099 </code>

// On a separate terminal, ssh to build the tunnel

<code> ssh -t -t login.sherlock.stanford.edu -L 8888:localhost:8000 ssh sh03-12n16 -L 8000:localhost:9099 </code>

// On your web browser, open the Jupyter by following address

<code> http://localhost:8888/tree </code>
