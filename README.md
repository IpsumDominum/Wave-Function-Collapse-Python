# COMPSCI 715 Group Project

### Setup python / enviornment
The code should be python2 and 3 compatible.
But recommended to use python3, since python2 is no longer maintained.

```bash
python -m pip install -r requirements.txt

```
or 
```bash
python3 -m pip install -r requirements.xt
```

If it doesn't work, for instance, if you don't have pip. Download pip https://pip.pypa.io/en/stable/installation/.
Use python from https://www.python.org/.
If using windows, remember to click "Add to system path" when you first install python. Otherwise follow
this to add python to your system path.
(https://geek-university.com/python/add-python-to-the-windows-path/)

### How to run

```bash
python test.py --img_name.png
```
Example:
```bash
python test.py --Flowers.png
```

### TODO

1.Tile Based Satisfy constraint 1. (Done)
2.Tile Based Satisfy constraint 2. (Done? Choosing each expansion with regards to a probability based on the frequency of the pattern's occurance)
3.Overlap based
4.Overlap Based Rotation and Symmetry.
5.Tile Based Rotation and Symmetry.

### Results

Flower.png             |  Skyline.png           |  Platformer.png
:-------------------------:|:-------------------------:|:-------------------------:
![Flower](./gifs/Flowers.png.gif) |![Skyline](./gifs/Skyline.png.gif) |![Platformer](./gifs/Platformer.png.gif)



