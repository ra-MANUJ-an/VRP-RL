#!/bin/bash
#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################
#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=8           # Number of CPUs to request for the job
#SBATCH --mem=120GB                 # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUs? If not, delete this line
#SBATCH --time=01-00:00:00          # How long to run the job for?
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?
#SBATCH --requeue                   # Remove if you do not want the scheduler to requeue your job
#SBATCH --constraint=h200
#SBATCH --partition=researchshort           
#SBATCH --account=caozhiguangresearch   
#SBATCH --qos=research-1-qos       
#SBATCH --mail-user=manujm@smu.edu.sg 
#SBATCH --job-name=sbatch-llmfifth     

# Purge the environment and load modules
module purge
module load Python/3.12.8
module load GCC/13.3.0
module load CUDA

# Create virtual environment directory if it doesn't exist
python -m venv ~/llmfifth

# Activate virtual environment
source ~/llmfifth/bin/activate

# Add library path (to prevent shared library errors)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import sys; print(sys.prefix)")/lib

# Install required packages
pip install --upgrade pip
pip install papermill nbformat jupyter ipykernel

# Create kernel for Jupyter
python -m ipykernel install --user --name llmfifth --display-name "Python (llmfifth)"

# --------------------------
# Step 1: Run Setup Notebook
# This notebook can handle package installation, state resets, or any other initialization.
# --------------------------
srun --gres=gpu:1 papermill \
    --kernel llmfifth \
    --autosave-cell-every=50 \
    --progress-bar \
    --log-output \
    /common/home/users/m/manujm/neuralcombinatorialoptimization/llm_finetuning/setup_notebook.ipynb \
    /common/home/users/m/manujm/neuralcombinatorialoptimization/llm_finetuning/setup_notebook_papermill_output.ipynb

# --------------------------
# Step 2: Run Main Analysis Notebook
# This notebook contains your core computations and will run in a fresh kernel session.
# --------------------------
srun --gres=gpu:1 papermill \
    --kernel llmfifth \
    --autosave-cell-every=50 \
    --progress-bar \
    --log-output \
    /common/home/users/m/manujm/neuralcombinatorialoptimization/llm_finetuning/grpo_verl_finetuning.ipynb \
    /common/home/users/m/manujm/neuralcombinatorialoptimization/llm_finetuning/grpo_verl_finetuning_papermill_output.ipynb
