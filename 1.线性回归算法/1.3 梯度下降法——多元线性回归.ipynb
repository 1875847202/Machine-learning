{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from numpy import genfromtxt\n",
    "import matplotlib.pyplot as plt\n",
    "from mpl_toolkits.mplot3d import Axes3D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[100.    4.    9.3]\n",
      " [ 50.    3.    4.8]\n",
      " [100.    4.    8.9]\n",
      " [100.    2.    6.5]\n",
      " [ 50.    2.    4.2]\n",
      " [ 80.    2.    6.2]\n",
      " [ 75.    3.    7.4]\n",
      " [ 65.    4.    6. ]\n",
      " [ 90.    3.    7.6]\n",
      " [ 90.    2.    6.1]]\n"
     ]
    }
   ],
   "source": [
    "#加载数据\n",
    "data = genfromtxt(r\"Delivery.csv\",delimiter=',')\n",
    "print(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[100.   4.]\n",
      " [ 50.   3.]\n",
      " [100.   4.]\n",
      " [100.   2.]\n",
      " [ 50.   2.]\n",
      " [ 80.   2.]\n",
      " [ 75.   3.]\n",
      " [ 65.   4.]\n",
      " [ 90.   3.]\n",
      " [ 90.   2.]]\n",
      "[9.3 4.8 8.9 6.5 4.2 6.2 7.4 6.  7.6 6.1]\n"
     ]
    }
   ],
   "source": [
    "#切分数据\n",
    "x_data = data[:,:-1]\n",
    "y_data = data[:,-1]\n",
    "print(x_data)\n",
    "print(y_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "#学习率\n",
    "lr = 0.0001\n",
    "#参数\n",
    "theta0 = 0\n",
    "theta1 = 0\n",
    "theta2 = 0\n",
    "\n",
    "#最大迭代次数\n",
    "epochs = 1000\n",
    "\n",
    "#最小二乘法\n",
    "\n",
    "def compute_error(theta0,theta1,theta2,x_data,y_data):\n",
    "    totalerror = 0\n",
    "    for i in range(len(0,x_data)):\n",
    "        totalerror += (y_data[i] - (theta1*x_data[i,0] - theta2*x_data[i,1] + theta0))**2\n",
    "    return totalerror/float(len(x_data))\n",
    "\n",
    "#梯度下降法\n",
    "def gradient_descent_method(x_data,y_data,theta0,theta1,theta2,lr,epochs):\n",
    "    #计算总的数据量\n",
    "    m = float(len(x_data))\n",
    "    \n",
    "    for i in range(epochs):\n",
    "        theta0_grad = 0\n",
    "        theta1_grad = 0\n",
    "        theta2_grad = 0\n",
    "        #计算求和，然后计算平均\n",
    "        for j in range(0,len(x_data)):\n",
    "            theta0_grad += (1/m)*(y_data[j] - (theta1*x_data[j,0] + theta2*x_data[j,1] + theta0))\n",
    "            theta1_grad += (1/m)*(y_data[j] - (theta1*x_data[j,0] + theta2*x_data[j,1] + theta0))*x_data[j,0]\n",
    "            theta2_grad += (1/m)*(y_data[j] - (theta1*x_data[j,0] + theta2*x_data[j,1] + theta0))*x_data[j,1]\n",
    "        #更新参数\n",
    "        \n",
    "        theta0  = theta0 - lr*theta0_grad\n",
    "        theta1  = theta1 - lr*theta1_grad\n",
    "        theta2  = theta2 - lr*theta2_grad\n",
    "    return theta0,theta1,theta2  \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "after 1000 iterations theta0 = -1.298633992544752e+221 theta1 = -1.094846092908564e+223 theta2 = -3.806952720058741e+221\n"
     ]
    }
   ],
   "source": [
    "theta0,theta1,theta2 = gradient_descent_method(x_data,y_data,theta0,theta1,theta2,lr,epochs)\n",
    "print(\"after {0} iterations theta0 = {1} theta1 = {2} theta2 = {3}\".format(epochs,theta0,theta1,theta2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
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
    "ax = plt.figure().add_subplot(111,projection = '3d')\n",
    "ax.scatter(x_data[:,0],x_data[:,1],y_data, c = 'r',marker = 'o',s = 100)\n",
    "x0 = x_data[:,0]\n",
    "x1 = x_data[:,1]\n",
    "\n",
    "x0,x1 = np.meshgrid(x0,x1)\n",
    "z = theta0 + theta1*x0 + theta2*x1\n",
    "\n",
    "ax.plot_surface(x0,x1,z)\n",
    "\n",
    "plt.show()"
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
