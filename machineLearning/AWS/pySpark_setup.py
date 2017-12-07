"""
--------------------------------------------------------------------------------
-- Insatllation on Command Line for pyspark, hadoop, mapreduce etc. for EC2
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- setting up Anaconda and python:
--------------------------------------------------------------------------------

ubuntu@ip-172-31-21-42:~$ wget http://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh
ubuntu@ip-172-31-21-42:~$ bash Anaconda3-4.1.1-Linux-x86_64.sh

--------------------------------------------------------------------------------
-- Use Anaconda3 python:
--------------------------------------------------------------------------------

ubuntu@ip-172-31-21-42:~$ source .bashrc
ubuntu@ip-172-31-21-42:~$ which python
/home/ubuntu/anaconda3/bin/python

--------------------------------------------------------------------------------
-- Configure Jupyter Notebook
--------------------------------------------------------------------------------

ubuntu@ip-172-31-21-42:~$ jupyter notebook --generate-config
Writing default config to: /home/ubuntu/.jupyter/jupyter_notebook_config.py

ubuntu@ip-172-31-21-42:~$ mkdir certs
ubuntu@ip-172-31-21-42:~$ cd certs/
ubuntu@ip-172-31-21-42:~/certs$ sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem
Generating a 1024 bit RSA private key
.............++++++
......++++++
writing new private key to 'mycert.pem'
-----
You are about to be asked to enter information that will be incorporated
into your certificate request.
What you are about to enter is what is called a Distinguished Name or a DN.
There are quite a few fields but you can leave some blank
For some fields there will be a default value,
If you enter '.', the field will be left blank.
-----

Country Name (2 letter code) [AU]: US...

--------------------------------------------------------------------------------
-- Setting up congfig file
--------------------------------------------------------------------------------

ubuntu@ip-172-31-21-42:~/certs$ cd ~/.jupyter/
ubuntu@ip-172-31-21-42:~/.jupyter$ vi jupyter_notebook_config.py

- this opens up visual editor

--------------------------------------------------------------------------------
-- setup jupyter notebook
--------------------------------------------------------------------------------

- type "i" to insert

c = get_config()
c.NotebookApp.certfile = u'/home/ubuntu/certs/mycert.pem'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888

-press escape to stop insert process
-type ":wq!"
-w: write
-q: quit
-then hit enter, this will exit vi

--------------------------------------------------------------------------------
-- open jupyter notebook on ec2:
--------------------------------------------------------------------------------

-in browser:
https://ec2-52-14-84-72.us-east-2.compute.amazonaws.com:8888

--------------------------------------------------------------------------------
-- Install Java in order to install Scala in order to install Spark
--------------------------------------------------------------------------------

ubuntu@ip-172-31-21-42:~/.jupyter$ sudo apt-get update

1) install java

ubuntu@ip-172-31-21-42:~$ sudo apt-get install default-jre

-check it worked

ubuntu@ip-172-31-21-42:~$ java -version
openjdk version "1.8.0_151"
OpenJDK Runtime Environment (build 1.8.0_151-8u151-b12-0ubuntu0.16.04.2-b12)
OpenJDK 64-Bit Server VM (build 25.151-b12, mixed mode)

2) install Scala

ubuntu@ip-172-31-21-42:~$ sudo apt-get install scala

-check it worked

ubuntu@ip-172-31-21-42:~$ scala -version
Scala code runner version 2.11.6 -- Copyright 2002-2013, LAMP/EPFL

3) install Spark with hadoop

-install library py4j to connect python to java, to do this we must connect pip to Anaconda

ubuntu@ip-172-31-21-42:~$ export PATH=$PATH:$HOME/anaconda3/bin
ubuntu@ip-172-31-21-42:~$ conda install pip
ubuntu@ip-172-31-21-42:~$ pip install py4j

- notice the directory change

ubuntu@ip-172-31-21-42:~$ wget http://apache.mirrors.tds.net/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz
ubuntu@ip-172-31-21-42:~$ sudo tar -zxvf spark-2.2.0-bin-hadoop2.7.tgz

--------------------------------------------------------------------------------
-- Set home of Spark in Ubuntu
--------------------------------------------------------------------------------

ubuntu@ip-172-31-21-42:~$ export SPARK_HOME='/home/ubuntu/spark-2.2.0-bin-hadoop2.7'
ubuntu@ip-172-31-21-42:~$ export PATH=$SPARK_HOME:$PATH
ubuntu@ip-172-31-21-42:~$ export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
"""
