## virtualenv on spike

- mkdir {foldername}
- cd {foldername}
- virtualenv -p python3 .
- source bin/activate

## install jupyter notebook in virtualenv
- pip3 install jupyter notebook

## hoste notebook server with forwarded gui
- jupyter notebook --no-browser --port={1234} --ip=127.0.0.1

## open ssh tunnel
- ssh -L {port}:localhost:{port} {username}@{spike ip}
- open browser & go to localhost:{port}



## test env
- test environment is installed on spike /home/alisa/projects/graphalgorithmtest/
- activate with source /home/alisa/projects/graphalgorithmtest/bin/activate
- jupyter server is installed
- /home/alisa/projects/graphalgorithmtest/jupyter/
- start notebook server with same command as above

- in /jupyter/ there is one file testing loading & calling some functions but no pipelines or proper networks yet