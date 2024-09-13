
#The code below installs 3.11 (assuming you now have 3.10) and restarts environment, so you can run your cells.
import sys #for version checker
import os #for restart routine

def upgrade_python():
    if '3.11' in sys.version:
        print('You already have 3.11, nothing to install, proceeding.')
    elif '3.10' in sys.version:
        print("Python version is: ", sys.version)
        print("WARNING: This script will install Python 3.11 and restart the "
        "environment. You WILL recieve a message saying the kernel 'crashed' do "
        "not be alarmed, this is expected. Just click 'Reconnect' and you should "
        "and run this cell again.")

        print("Printing content of /usr/local/lib/python* to see available versions")
        !ls /usr/local/lib/python*

        !sudo apt-get update -y > /dev/null
        !sudo apt-get install python3.11 python3.11-dev python3.11-distutils libpython3.11-dev > /dev/null
        !sudo apt-get install python3.11-venv binfmt-support  > /dev/null #recommended in install logs of the command above

        #change alternatives
        !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 > /dev/null
        !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 > /dev/null

        # install pip
        !curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11  > /dev/null

        #install colab's dependencies
        !python3 -m pip install ipython traitlets jupyter psutil matplotlib setuptools ipython_genutils ipykernel jupyter_console notebook prompt_toolkit httplib2 astor  > /dev/null

        #minor cleanup
        !sudo apt autoremove  > /dev/null

        #link to the old google package
        !ln -s /usr/local/lib/python3.10/dist-packages/google /usr/local/lib/python3.11/dist-packages/google > /dev/null

        #this is just to verify if 3.11 folder was indeed created
        print("Printing content of /usr/local/lib/python3.11/")
        !ls /usr/local/lib/python3.11/

        !sed -i "s/from IPython.utils import traitlets as _traitlets/import traitlets as _traitlets/" /usr/local/lib/python3.11/dist-packages/google/colab/*.py
        !sed -i "s/from IPython.utils import traitlets/import traitlets/" /usr/local/lib/python3.11/dist-packages/google/colab/*.py

        #restart environment
        os.kill(os.getpid(), 9)
    else:
        # you shouldn't be here...
        print("Your out of the box Python is not 3.10, do you even colab bro?")

def install_java():
    !apt-get install -y openjdk-8-jdk-headless -qq > /dev/null      #install openjdk
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"     #set environment variable
    !java -version       #check java version

def setup_env_for_cldk():
    print("Upgrading Python to 3.11")
    upgrade_python() # delete this line when colab is updated to 3.11
    print("Installing Java")
    install_java()
