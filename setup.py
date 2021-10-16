# Make this project pip installable with 'pip install -e'


# 1. Install dependencies (ConvTasNet and Embedding)
# 2. Download Pretrained-model 
# 3. Dowload data (optional)

import subprocess

def install():
    subprocess.call(['pip', 'install','-r', 'requirements.txt'])


install()