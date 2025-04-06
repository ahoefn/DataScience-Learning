A collection of files I'll write while learning the basics of data science, mostly following [this review](https://arxiv.org/abs/1803.08823).

### Compiling

First run

    pip install .
    
while in the DataScience-Learning directiory. Then, the chapters can be ran using 

    python src/main.py [arguments]

where [arguments] denotes which chapter should be run. Valid inputs here are '-ch2' and '-ch4-grad'. Note that multiple chapters can be run at the same time. For example, one can run

    python src/main.py -ch4-grad

to run src/ch4_gradient_descent/ch4_gradient_descent.py

### Dependencies

[numpy](https://numpy.org/)

[matplotlib](https://matplotlib.org/)
