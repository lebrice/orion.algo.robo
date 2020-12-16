python -m pip install --upgrade pip
pip install tox tox-gh-actions
sudo add-apt-repository ppa:teward/swig3.0 -y
sudo apt-get install -y libeigen3-dev swig3.0
pip install PyYAML Jinja2
pip list
pip install -r requirements.txt
pip install -e .
