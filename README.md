# Wave Function Collapse Algorithm (Another Python Implementation)

<!--
#### Tile Based (Adjacency extracted from input image relationships)

Flower.png             |  Skyline.png           |    Spirals.png           |  
:-------------------------:|:-------------------------:|:-------------------------:|
![Flower](./gifs/Flowers_speedup.gif) |![Skyline](./gifs/Skyline_speedup.gif) |![Spirals](./gifs/Spirals_speedup.gif)

#### Overlap Based (Needs fixing...)
Flower.png             |  Skyline.png           |    Cat.png           |  Red Maze.png           |  
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![Flower](./gifs/Flowers.png.gif) |![Skyline](./gifs/Skyline.png.gif) |![Spirals](./gifs/Cat.png.gif)|![Red Maze](./gifs/RedMaze.png.gif)
-->
### Visualization 
Flower.png             | Red Maze.png| 3Bricks.png | Platformer.png
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![Flower](./wfc_outputs/Flowers.png.gif) | ![Red Maze](./gifs/RedMaze.png.gif)| ![3Bricks](./wfc_outputs/3Bricks.png.gif)|![Platformer](./wfc_outputs/Platformer.png.gif)

### Original Project
The original project which introduced the wave fu)nction collapse algorithm is by Maxim Gumin. [Original Github](https://github.com/mxgmn/WaveFunctionCollapse)
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
python run.py --spec file location
```
Example:
```bash
python run.py flowersAbsoluteGC.yaml
```

### TODO

1.Tile Based (Done (Need Revising))

2.Manual adjacency creator (TODO)

3.Overlap Based. (Done)

4.Overlap Based Rotation and Symmetry. (Done)


5.Overlap Backtrack. (Done)

6.Overlap global constraints -> Ground (Done)

7.Overlap more global constraints (Done)
