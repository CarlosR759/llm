# Llm made in pytorch 

## instructions
First you need to have pytorch, for that you need to got to this link and download the version that you would like to use

```
https://pytorch.org/get-started/locally/
```

then create your virt env in python like 

```
python -m venv <your folder name which is going to be created>
```

or if you just downloaded your repo with git clone, in your cloned folder use:

```
python -m venv .
```

after that install the requirments like this in your virt env : 

```
source /bin/activate && pip install -r requirements.txt
```

### To install and use ipykernel:

To use the code without Jupyter notebook UI, but also being comfy in running the code in separate cells you can use ipkernel and use repl or something similar. To install ipkernel you need to use this commmand to setup your jupypter kernel to be able for use in IDEs that have repl functionality:

```
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```


