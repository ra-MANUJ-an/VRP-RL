### Exaplanation
Since the whole code base is built on top of verl codebase used internally in my company, I cannot upload the whole code base for you to use. However instead, I will provide the steps on reproducing the code based on the open-source verl code base.

### Step1: Replace/Add files
We have 5 core files in total. Please replace/add the following files in the corresponding locations in the verl code base:

1. Replace `verl/verl/utils/reward_score/__init__.py` with `__init__.py`
2. Replace `verl/verl/workers/reward_manager/__init__.py` with `__init__(1).py` # remember to rename the file to `__init__.py` without '(1)' suffix
3. Put `hgs.py` in `verl/verl/utils/reward_score/hgs.py` 
4. Put `hgs(1).py` in `verl/verl/workers/reward_manager/hgs.py` # remember to rename the file to `hgs.py` without '(1)' suffix
5. Replace `verl/verl/trainer/main_ppo.py` with `main_ppo.py`

Here I think the code base is good to go

### Step2: Setup the environment
Please try to run the code with vllm-v1: https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html
I use the internal docker image that I cannot share with you. However, I have provided a `requirements.txt` for your reference.

### Step3: Prepare the dataset
Now th new codebase will not need 'tsp100' dataset anymore. It will use the training and testing parquet datasets you give me. Just properly set the training and testing data path in the shell script will make the magic happen.

ONCE you setup the code and the environment, please maintain a docker file. Please upload your code in a new branch `vllm-v1-dev-manuj`, and this will be the stable version of the codebase we are using now and in the future.