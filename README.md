Trading Using Q-Learning
==================

In this project, I will present an adaptive learning model to trade a single stock under the reinforcement learning framework. This area of machine learning consists in training an agent by reward and punishment without needing to specify the expected action. The agent learns from its experience and develops a strategy that maximizes its profits. This is my capstone project for the [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009), from Udacity. You can check my report <a href="https://www.dropbox.com/s/tnwkztqmozys1h4/trading-q-learning.pdf?dl=0" target="_blank">here</a> and the notebook with the tests of the codes used in this project <a href="https://nbviewer.jupyter.org/github/ucaiado/QLearning_Trading/blob/master/learning_trade.ipynb" target="_blank">here</a>. The TEX file was produced with help of [Overleaf](https://www.overleaf.com/read/mmzwqfbrkdvf).


### Install
This project requires **Python 2.7** and the following Python libraries installed:

- [Matplotlib](http://matplotlib.org/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [Seaborn](https://web.stanford.edu/~mwaskom/software/seaborn/)
- [Bintrees](https://pypi.python.org/pypi/bintrees/2.0.2)


### Run
In a terminal or command window, navigate to the top-level project directory `QLearning_Trading/` (that contains this README) and run one of the following commands:

```python qtrader/agent.py```  
```python -m qtrader.agent```


### Reference
1. T.M. Mitchell.  *Machine  Learning*.   McGraw-Hill International Editions, 1997. [*link*](http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/mlbook.html)
2. M. Mohri, A. Rostamizadeh, A. Talwalkar. *Foundations of Machine Learning*. 2012. [*link*](https://mitpress.mit.edu/books/foundations-machine-learning)
3. N.T. Chan, C.R. Shelton.  *An Electronic Market-Maker*. 2001 [*link*](ftp://publications.ai.mit.edu/ai-publications/2001/AIM-2001-005.pdf)
4. N.T. Chan.  *Artificial Markets and Intelligent Agents*. 2001 [*link*](http://cbcl.mit.edu/cbcl/publications/theses/thesis-chan.pdf)
5. R. Cont, k. Arseniy, and S. Sasha. *The price impact of order book events*. Journal of financial econometrics 12.1, 2014 [*link*](https://pdfs.semanticscholar.org/d064/5eb3d744f9e962ff09b8a5e9156f2147e983.pdf)


### License
The contents of this repository are covered under the [Apache 2.0 License](LICENSE.md).
