{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from numpy import genfromtxt\n",
    "from sklearn import linear_model\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#加载数据\n",
    "data = genfromtxt(r'longley.csv',delimiter=',')   #只读取数据,无法读取字符\n",
    "#print(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#切分数据\n",
    "x_data = data[1:,2:]\n",
    "y_data = data[1:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.40875510204081633\n",
      "(16, 50)\n"
     ]
    }
   ],
   "source": [
    "#创建模型\n",
    "#生成50个值\n",
    "alpha_to_test = np.linspace(0.001,1,50)   #岭回归系数\n",
    "#创建模型，保存误差值\n",
    "model = linear_model.RidgeCV(alphas=alpha_to_test,store_cv_values=True)\n",
    "model.fit(x_data,y_data)\n",
    "\n",
    "#岭系数\n",
    "print(model.alpha_)\n",
    "#loss值\n",
    "print(model.cv_values_.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x21f53d06d90>]"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#画图\n",
    "#岭系数与loss的关系\n",
    "plt.plot(alpha_to_test,model.cv_values_.mean(axis=0))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
