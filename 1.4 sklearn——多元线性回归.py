{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from numpy import genfromtxt\n",
    "from sklearn import linear_model\n",
    "import matplotlib.pyplot as plt\n",
    "from mpl_toolkits.mplot3d import Axes3D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
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
   "execution_count": 11,
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
    "#数据切分\n",
    "x_data = data[:,:-1]\n",
    "y_data = data[:,-1]\n",
    "print(x_data)\n",
    "print(y_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#创建模型\n",
    "model = linear_model.LinearRegression()\n",
    "model.fit(x_data,y_data)   #二元线性回归传入数据"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "coefficient [0.0611346  0.92342537]\n",
      "intercept -0.8687014667817081\n",
      "predict: [9.06072908]\n"
     ]
    }
   ],
   "source": [
    "#系数\n",
    "print('coefficient',model.coef_)   #对应两个系数\n",
    "#截距\n",
    "print('intercept',model.intercept_)  #偏置\n",
    "\n",
    "#测试\n",
    "x_test = [[102,4]]\n",
    "predict = model.predict(x_test)\n",
    "print('predict:',predict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<mpl_toolkits.mplot3d.art3d.Poly3DCollection at 0x1d19c40c4f0>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPMAAADyCAYAAACYqvOaAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy86wFpkAAAACXBIWXMAAAsTAAALEwEAmpwYAABhZElEQVR4nO2dd3hc5Zn2f2eqNOrdsmTZli3bcu+YTiCw+QCDwbQQSgohySbBuwm7ISHLpkFIliWQhJBdliSEdGxKQklII3Tb4CLZluUiyep1RtJo+sx5vz/kcxiNpuvM2BJzX5cv0MxpM3Pu8zzvU+5HEkKQQQYZTH/oTvUFZJBBBtogQ+YMMpghyJA5gwxmCDJkziCDGYIMmTPIYIYgQ+YMMpghMMR4P5O3yiCD1EPS4iAZy5xBBjMEGTJnkMEMQYbMGWQwQ5AhcwYZzBBkyJxBBjMEGTJnkMEMQYbMGWQwQ5AhcwYZzBBkyJxBBjMEGTJnkMEMQYbMGWQwQ5AhcwYZzBBkyJxBBjMEGTJnkMEMQYbMGWQwQ5Ah8ymAEAKv14vf7ycjdZyBVoglTpCBxpBlGa/Xi9vtVl/T6/UYjUYMBgN6vR5J0qRXPYP3GaQYliFjNjSCEAK/34/f70eSJHw+n/q6EAJZllUSezwe8vLyMJlMGXK/P6DJD5yxzGmA4lYHE1aBJElIkoROp1O3PX78OPPmzcNisQAZy51BfMiQOcXw+/10dnYSCASoqqpCkiTVGocjpUJuvV6PXq9XrbbL5VK3NxgM6r8MuTNQkCFzihDsVsuyrLrXiSKc5Q4EAvj9fnUbg8GgWm6dTpch9/sUGTKnALIs4/P5VLdascbxItr2yvEUhJJbkqQJljtD7vcPMmTWEAqxlOCWYk0jkTOSq50IwpHb7/er16BYdYPBgMlkypB7BiNDZo0ghMDn8xEIBCYRLJTMsax1opY8dN9Qcnd2dgJQWVmZsdwzGBkyawAld6xY2nAR61NVHBJ8PUpAzefzTbDcyppbr9dnyD2NkSHzFBCaO1bc6lCEI7PT6cRkMmEwTP4JUkl+JVKuIBy5lWCawWAI+3DK4PREhsxJIjR3HO2GDyanLMs0NzczMjJCIBBAr9dTVFREUVER+fn5E6LWWiLS9YUjt9frxePxAOPrfqPRqFruDLlPX2TInASUIFcktzoUCpkdDgeNjY3MmjWL2tpaAHw+H8PDw/T19XHkyBFMJhM+nw+Hw0FeXl7aiRON3MHBtGC3PIPTA5lyzgQQ6lbHS7ShoSHa2trweDwsW7aMgoKCCWvsYLjdbpqamtDpdHg8HrKyslTLnZOTkxS5Ozs70el0zJ49O+F9gxHsXRw6dIglS5ao7niG3FNCppwznQiXO44HgUCAtrY2HA4HZ555JkajMer2WVlZWCwWZs2aRX5+Pi6XC5vNph4jJydHJXd2dnZaLXdwIM3tdqsW3Ov14vV6ATKW+xQiQ+YYUHLHx44dY+7cuQkReWxsjMbGRoqLi8nKyopJZAWKWy5JEhaLBYvFQlVVleqq22w2jh07htvtJjc3VyV3VlbWVD5qwgguPYX3mkZCyR1cV54hd+qQIXMUBOeOe3p6mD9/ftz7dnV1ceLECVasWEEgEKCrq2vK1yNJErm5ueTm5jJnzhyEENjtdmw2G4cPH8br9ZKfn6+S22QyTfmcsa4n9O/QHLcQAo/HMymgliG39siQOQJCc8fxwu/3c+jQIQA2btyIwWBgZGREs3LO0O3y8/PJz89n7ty5yLLM6OgoNpuNrq4uAoEABQUFyLJMTk5O3OfXCrHILYSY4JIrqbAMkkOGzCEILskMzh0Hu76RYLfbaWxsZO7cuVRVVamvp6toRKfTUVhYSGFhIfPnzycQCDAyMkJ7ezs2m42+vj4KCwspKiqisLBwQtQ6HQhHblmWVaGG7u5uqqur1fx7piMsMWTIHIRouWNJkpBlOSwBlJLJzs5OVq5cSW5u7oT3w5E5VeWcwdDr9RQXF+N0OtHpdJSXlzM8PIzVaqW1tVUlf1FREQUFBWl3eUO/497eXmbPnp1RYUkSGTKfRKySTJ1OF5Zgfr+fAwcOYDQa2bhxY1iyn8pyzmAYDAZKS0spLS0FxqPQw8PD9Pf3c+zYMQwGA0VFRRQXF5Obm3tK1rOhOe7QXu4MuSPjfU/mREoyZVme8NrIyAgHDx5k/vz5VFZWRjyHli2QWsJkMlFeXk55eTkwLlekrLftdrsmOe6pIFwvd4bckfG+JnMiuWOdTqeSWQjBiRMn6O3tZdWqVTGDS6eDZY7nJjebzcyaNYtZs2YhhMDtdmOz2Thx4gRjY2MTctyn4vPEQ+73swrL+5LMoX3HiZRker1eDhw4QHZ2Nhs3bozLFT3VljmZY0mSRHZ2NtnZ2cyePRshBE6nE5vNxvHjx3E6nRw6dOiU5biVawynwqIELxWYTCbMZvOM7wh735E52ZJMnU7H8PAwLS0tLFy4kIqKirjPeTpY5qlCkiRycnLIycmhurqaXbt2MWfOHGw2G83NzXg8nrTmuCNdYyi529vbMRqNlJeXT2j3nIm93O8rMidbkqlUXrW2trJ27Vqys7MTOm8omYUQ9Pb2otPpKCoqmtQGOR3IL0kSeXl55OXlUVNTgyzL2O12rFbrhBy3kgaLt/pN62tUctkGg2HGSyy9L8gcKXccD7xeL42NjQghWL58ecJEhonk9Pl8HDhwQC2SOHHiBJIkqVHk/Pz8aUHmUOh0OgoKCigoKFBz3EoBS3t7O0KIU5LjlmV5Qq1AaJ47VGJpOpN7xpNZKcncs2cPq1evTujHsVqtNDU1sWjRIvr6+pK+BoWco6OjHDhwgPnz51NaWqp6CD6fTy3qOHLkiGrVLBbLKYkia4HgPm0YT+GNjIxgs9nSmuMOJnMowpF7OquwzGgyB+eOnU5nQm718ePHsVqtrFu3jqysLAYGBialphKB2+3m4MGDalGJcsMA6ppOSREdO3YMv98/IYpcXFysdkpNRxgMBkpKSigpKQFQH2DBOW6Px8PIyAh5eXmakTsamUORiArL6UjuGUnmcLnjeL90t9tNY2MjhYWFrF+/PqbCZiwotdp+v5+zzz47LvfSaDSSl5dHRUXFhE6pI0eOqIGm4uJiCgsLT0mgSQuEPsBcLhf79u2ju7sbu92O2WxWLXtubm7SpEmEzKGIJtTQ3d1NRUUFFovltJFYmnFkTkTOJxSDg4M0NzezZMkS1YIoCM4zxwulBbK6uhqn0znhxoh2vOAHR2inVHAzRWdnJ7Isn9J6a61gMpkwmUzU19cDqH3c7e3tjI2NYbFYVHJbLJa4f9epkDkUweS2Wq1UVFRMUGFRLPep6uWeUWQODWaE/uCRGiVkWebYsWOMjIywfv16zGbzpG0Stczd3d20tbWxYsUKcnNz6ejomHQtySC0mUJZi1qtVlpaWtS1anFxsabuaqoRSrpIOe6WlhacTueEPu5oSw8tyRwMRb8tuJcb3hNq+NznPsdXv/pVlixZovm5I2FGkDme3LFiCUMtl8vloqGhgdLSUtavXx/xiR+vZZZlmaamJnw+n9oCmShxE3lwhK5FvV4vNptNdVezsrLUVkktRPdThWjXFprjFkIwNjY2YemRl5enkjv4YZwqMofeS8HlpTBuudMd35j2ZI43dxyOzP39/Rw9epT6+nqKi4ujnicegjmdThoaGqisrKSmpka9lnQSyGQyUVFRoa63XS4XLS0tDAwM0Nvbq970xcXFYT2QeJCKtJkQIqFAVbgct81mU+MTSo7b7/dPicxSXx/6nTvBZoOiIgIbNyJmzVKvIxIcDsek7rlUY9qSOdIomEgItqyyLHPkyBEcDgcbNmyIK4gUyzL39fVx7Ngxli1bRmFhYdyfIdwNoVWeWZLGZYfy8/MpLS2loqKCsbExNeXm8/kmBNPiLexIhYUPN+42XgTnuOfNmzchx22329m/f79qtQsKCsJqlU+Cy4XpRz9C/49/gBCg04EsY3zsMQJnn4105plRd1eWAunEtCRztFEwkaCQUbGeFRUVLF68OO4bKBLBknkwnCoEWzRFmURZb7e3twNQWFioFq+kM5im5QMiOMdttVpZuXKlarlbW1vVIh1Fq3zS5/T7Md13H/q9exGzZ48TWYEso3/jDRYcPgxnngkRfm+fz5f2e2HakTlW33Ek6HQ6+vr66OrqSsh6Bu8fapndbjf79++nrKwsoQdDLKSrAkwpJw0u7LDZbAwMDKi5XyW/HazhnSrLnIq1rRACk8k0Kcc9PDw84XMq30NeXh7G3bvR79mDqK6G0M+p0yFmzya3qQn9W28ROP98za85WUwbMsfbdxwOitulBKWSqRMOJbOSxopnvT1dYDAYKCsro6ysDBjvb7ZarXR2dmK327FYLKrV1hrpDM4ZjcZJn3N4eJienp7x1ORjj5Gn12Pw+zEYjZNErYUkEbBYMDz9dFgyn6pA47Qg81Ryxw6Hg4aGBkwmE3V1dUkX/CviBEIIjh07xvDwcMQ01lQRTgjhVMBsNlNZWUllZeWk9JDdbqepqUkNpk3VpUwkAKY1zGazGjQEMN17L+78fBwOBz6/H4Nej8lsxmwyoVcaNnJz0bW2jq+nw9yPp4LQpz2ZA4EAra2tFBcXJ1QsAO/lepcvX05nZ+eUXFelhvrdd9+loKAgahprqjgd00fB6aGKigoaGxupqqrCarVy8OBB/H7/hOKVuIJMQZhKAExr6IxGLFlZWAwGhNeLPDyM327HpdPhNpvRGwzIgQCBCA9cv99/Sop3TlsyB7vVo6OjCc1dCgQCNDU1EQgE1FxvMhVcwXA6nXR0dLBixQrVPUsWHo+HxsZGvF6vatlCmw1O564pxYoqMr9KBFkJprW1tU3qBItldU+nHLi8Zg363buRhoeRentBCEyABRBZWbhranC6XAzU1dG8e/ekHLcyeUQLSJK0Dfgk4yNsHhNCPBRp29OSzKG5Y51ORyAQiGtfRe62pqaGqqqqCcn8ZMgshKCtrY3e3l4qKyunTGQlF1pXV4fFYmFkZETNd5vNZoqLi9Uo/emKcMRTlECV+EFoJ5hSa11cXBy2EyxVAbBk4L/4YoxPPgmyjLBYJrrRPh9ZTU1QUIDltttYv2aNmu47dOgQg4ODPPnkkwQCAYaGhiaVBScCSZKWM07kjYAX+KMkSS8IIY6G2/60InMkOZ94iCiEoKuri/b2dlasWEFeXt6E9xN5ICjw+Xw0NjaSnZ1NXV0dIyMjiX2gkOs7ceIEPT09rF27Vp32GByIcblcWK1W+vr6cLvdOBwOlSCnW8or1sMmXCNFqJ5YcCdYKixz0iWzra0IiwXJbgdZhmCXWadDyDI6rxd/be0kD8Xj8dDb28uPf/xjtm7dypo1a/je976X7EeoB94WQjgBJEn6B3AV8N1wG582ZI6WO9br9VGJ6Pf7OXjwIDqdTnWrQ5Gomz0yMsKBAwdYsGABs2bNYmhoKOmbw+/343a7sdvtbNiwIeLnyc7OpqqqCoPBgMvlori4GKvVyoEDBwgEAqorl0xDhZZESeZ7CK21Du0EU4TvfT6fZqokSVl7nw/D888TWLsWqa8PXUsLeDxqRFvo9fgWLsQLGF9/ncDmzRN2N5vNLFu2jHXr1vHTn/50qsulA8C9kiSVAC7gUuCdSBufFmSOlTuOZpmVhv958+ZFHVkaL5mFEHR0dNDV1cWaNWuwWCxA8hFmJZqu1+tZvnx53Psp9dTBa1JFwL6lpUXNASsa19HIqvX6e6pWNFwn2IkTJxgeHqaxsVGzTrBkyCz19oLLBYWF3FdWx2PlF+DTm/mo9TDfMlkRhYX4vF7E0BD6PXsmkRlQPQ/lsyYLIUSTJEnfAf4MjAH7AX+k7U8pmeOV8wnnIgeTLtwUiXDHiEVGxcLr9fpJgvaRRPCjQSnxXL58OQcPHox7v3BFI3q9fkLhg5IDVloEFbdVmTiZSmjtEut0OrKzs9Hr9dTU1OD3+yc8uJT1uFLUES9B4yXzA7/7G4//tREfeiSdAany/6EzZSPljmdPpOEevlkpIRj/7kVQeWc4aFnKKYR4HHgcQJKk+4DOSNueMjInkjvW6/UTlDkUHS2z2RxxikQodDrdhGOEQgmcRbLwiVhmIQRHjx5V3WplvaslCUJzwA6HA6vVyuHDh/H5fBQUFKg111ojFZH2YOKFm7wR2gmmBNOipStDyTyJtHojOnM2ksmCVDKPcFGJgHOUrzmOQlCdjAD0bjfy4sVhz6txNLtcCNEvSVINcDUQsSj8lJA50ZJMZbg3wPDwMAcPHlTXsvEimmVWxq9Gs/DxWmav16sW9q9duzYp8iZazhnsttbU1ExKE3m9XvLy8sjKytKsxzkVwapIxwzXCRbc2xzaCfbDp1/lkZf34RO6k6TdFZO0Ya8p4Mc80svH870TX/f50AlB4KKLwu6nccfUjpNrZh/wWSGELdKGaSVzsiWZiputpIiC17LxIhyZw+WjIyEey6w8aBYtWjQphaUQNB0pp9A0UUtLC36/X7Vs2dnZqtua6PcIqckJx1sBpnSCWSwWHn1xN7/82z5knxvh84AcIHv+OnQ5BUjFc+MmbST4hntpyTkBTExNGXp6GLv6aiwRtNOVLIQWEEKcG++2aSNzsprVMP5D9/T0UF5eHvcUiVCEklkJTFVVVTFnzpyY1xPNMivr9+7u7ogPmkSsrdaNFnq9Xq3cUsoyrVYrx44dw+12q22QRUVFcUWSU9VoEelh+tXHnp1AWuFzI3ucCK9zwnaWJeeizy3U5Hr8o4McNTej6+hCAOh0SLIMBgNDW7YgrrmGSI9Bh8PBnDlzNLmORJByMiczCiYYysSEnJwcVR8qGQSTube3l+PHj7N8+XIKCgri2j+SZQ4EAmpaTEk7Rdr/dKjqCi7LDNYUs1qtqrRRcO9vugo5hBB89zd/55ldzcheN8LvQXjDkzYcTLPqMJXN1eRaZK+L9bOzkL78MO7jx9Hv3w8eD2LWLAIbNmDt7aU0ihd3KoQJIMVkTnYUjLJvS0sLg4OD1NfX09vbO6VrUVz1pqYmXC5Xwt1T4Syz0+lk//79VFdXx3wSh5I52neRCuJHU2BRNMVgogRucFVacLBpqpb5pnuf4PUDreOE9bnHSet1IjyxSRv2M1gKyK5dl/T1BEMIgX+kj189sG3874UL8S9cOGGbQFdX1KDrjCOzEuTatWsXGzduTOjHV2qX8/Pz2bBhAy6XK+HqrVD4fD76+vqYP38+S5YsSfhmDLXMys0er3VP1M3WEok8GMJVbikpIiXYlJ2dHVdkf5y0bSBk8HuQPU4CrlGEx5HsR5kMnZ6cJeci6bW5lf3Dvbz+zevxer0Rq+4UMb9IOBUqI5ACMofmjhOtMx4aGuLw4cMTgkjJ1lUrGBgYoLm5mby8PObPn5/UMRTLrLRAjoyMJKQskqi1PR1ccnivKq2qqgohBHa7na6uLmw2G++88w6FhYXc88u/88bhLvB7QQ4gvC7tSRvp+mrXo88p1ORYAecIX928Qh1YEKkTLBAIRF1+BBeNpBOakjkZOZ/gfZU+YWWKhIJk6qrhPQnd0dFRVqxYQVtbW8LHUCBJEn6/X22BXLduXUKfL5TMQ0NDWK1WSkpKJnWEnS7r61B89Ns/55WGFgj4TgaiXMgeR9Lu8VRhLK3BNGth7A3jgAj4yfaNcOulZwNE7QRTdLIjweFwTOoNSAc0I7MQAo/Hk7CcD4zL7zQ0NFBcXBy2TzhWbXY4eDweGhoaKCoqYt26dXg8nilZd7vdztjYGKtXr1Zd0ESgEFSJBQwNDVFeXq6qeOTk5FBSUnJaqJbcet8TQaR1n4wenzrShoNkziG7bpNmSxLfcC9N//P5Ca9F6gTr7u6moaEhYieYshyZKiRJ+lfgNsbrVBqBjwkh3JG214zMCoHDfbnRAiYDAwMcOXIk7BQJBYk2SSiuevAxk7XuAB0dHXR2dmKxWJIiMrwnbnD48GGys7NZu3Ytfr9/QuPB0NAQhw4dwuPxoNPpsFqtFBQUpKzR/YuP/I7trx8E/+lL2vCQyFlyDjqDNp1k/tEBDn7/tpjbKfGEEydOsH79etxu96ROsEOHDuH1eqfsZkuSVAXcASwVQrgkSfodcAPws0j7aOpmhyNdJPF5WZbVksdY8jvxPn2DrV44Vz1RyxwIBDh06BBCCDZu3MjOnTsT2j/0WA0NDdTW1jJ79uwJ1xJcwTV37lyGhobo7OxkcHCQ48ePYzQaVaudqNoKwP+9uItf72pB6IzoDCYkYxaSKRvPif34BtqS/kynClnzVmHIn1pfuQLZ62LTnJyE5J8U4xSuE+xPf/oTnZ2dbNiwgU2bNvHII49M5WFsALIlSfIxro3QHWvjlEJxkYM/kDJFoqysLOG1ZyQoc5Rzc3MnDHxTkCiZlbRTvEUl0dDf38/w8DDLly+PqwTVYDCQlZXFokWLgPFlyNDQEC0tLbhcLvLy8igpKZlU5PGd37zMY3/ci6wzoNObkExmdOYcdGYL5tkTc/SeniPTksiGwlmYq5dpciwhBP7Rfn7+wB0J7RfuXlAeyF/4whd49tln2blzpzqHO8lr65Ik6QGgnfH2x5eFEC9H2yflZDYYDPj9fjXqq3QSLV26VJV4nSqUMsq6urqIbnAiQSXF9U9GkjcYQoyPhrXZbJSWlibtemVlZakRZVmW+faTz/OTv/8eEYa0ptmxZxv57UO4jkdsiz1tIRnNWBafpdk62T/cy7H/TYzI8UCSJMxmM+vWJZ/7liSpCLgSmA8MA09JknSTEOIXkfbRlMzhvmTFMsuyTHNzMy6XSzOxeEW9I5567XhugOCI+lSv0e/309DQQE5ODuvWrePgwYOTHiaRrkl58Pzwmb/z0LO7kPUKabPQmS0nLW1yA8lknwdH06vjud9pBsuis9GZEq8lD4eAc4QHbz47qX1jGQWNMhEfBFqFEAMAkiQ9DZwFpIfM4aDX63E4HBw8eJBZs2YlVbChIDiQ5vf7aWxsVNsgp1p2qLjpeXl5U1beHBsbo6Ghgfnz51NZWQlE9wz2HDnBdff+BllvRGcwIhmDLG2VdlMEhRA4m99IS/5Xa5ir6jEWRxafSAQi4CcnYGfzWSsS3jeWiqjX69VKKaUd2CRJkoVxN/sioqiMQBrI7Ha7OXLkCKtWrYq7DjocgssIFXWRYLJMBYpEUDQ3PV709fVx/PjxSTpk0ch83X//QVPSRoKn4wB+W9QYymkJfW4JWfNWa3a88TTUZ5PaN1wwNxha9TILIXZKkrQd2MO4ushe4H+j7ZMyNzsQCHD48GFcLhd1dXVTIjKMW3i/309/fz8dHR1xqYvEg87OTjo6Oli9evWUfoTgyrD169dPctEjkbn2Y/9NVqU2hQ/R4LP14D7RkPLzaA69EcuSc5B02qTn4k1DRUKsUk4t67KFEP8J/Ge826fEMo+Njaki6Yry4lSh0+k4ePAgBoMhbnWRaFCaLmRZnvLxfD4fDQ0N5OXlRYzOhyPzoo99B3NF6okse5w4m19nvPZgesGycCP6bG2qqWSvkw8sKJjSFJJ4yHwqSjkBNO9v6+rqoqGhgWXLllFTU4PBYJhyk8TY2BjDw8MUFBSwYsWKKRPZ5XKxe/du8vPzEz5eKCHHxsbYvXs3VVVVLFq0KGZQS8E19/wfuuI5SHptlCgjXq8s4zj82njz/jSDqWIBpvLkaulDIYSMf6SfOy5ZSk9PDx5Pct9HrLrsU0lmTS1zf38/Q0NDE1Q7DAYDXq83xp6R0dPTQ2trK0VFRVMWoIeppZ1C2/96e3tpaWkJq9MdaV+A9vZ29vT5MBRok5qLBnfrHgKjAyk/j9bQZeeTvWCDZsfzD/dx9H/vmCBYH9xIUVRUFNdDPZ4186nomAKNyVxeXk5RUdEkzWu/P6I6aETIsqwOBN+4cSPNzc1TsvCKgGBra2vSA9+UwhNJkiYI9sUTvQwm8zn/8TvNGumjwTtwAk/34ZSfR3NIOk3bGgOOYX582/lI0sT51Ip8sTK3ObgWO9I4pFhu9qnqmIIUBMC0aJJQBqJXVlZSU1Oj6oUl2yihrGkBVq1alfSaSZIkvF4vTU1N5OfnJyTYp5B5zo33YqqsS+r8iSDgHMF59K2UnycVyK5dhz5XG69FBHzkyXYuWjdZpSZUvtjr9YYdYVtcXEx2djaQ3gBYokhLBVgiZFaa/kPd4GQeCvCeSP6CBQvo7IwoORwXZFlmz5491NXVqeM/44UkSWz6wo8xlS9IuaifCPjHC0MCiXtEpxrGkjmYKhdpdjzfcB/v/M/n4trWZDIxa9YsZs2aFXbqRkFBAZIkRfXEZgyZI1WAxeNmK40XY2NjYauvkiGzMntq1apV5OTk0N3dnbSr3tvby9jYGGvWrElqGNh9v/0HIr9Ks06fSBBC4Dy2E9mZ/FysUwXJbNG0rXEqaajg5hdFK21kZIQTJ06ogojhplw6HI4p1ypIkrQY+G3QS7XAPdEmQEIaGy2iQelnLi0tjei6JuJmK+ttv9/Phg0b1GBcMq66EIIjR47gcDjUIWeJwul08tdWJ8bC+HW+k4W35yi+/taUn0d7SOQsPgedUZvh9bLHyZpSppSGCoZOp6OoqIixsTEqKiooLS3FZrPR29vLkSNHyMrKYmRkhN7eXubNmzelcwkhmoHVAJIk6YEu4JmY1zils8aBWG724OAg7777LgsXLqS2tjbiUzley6yknXJzc1m5cuUE+dZEyez1enn33XfR6/WsWbMGvV6fVM58yacfSQuR/fZBXC3Tr4ECIGvuCgwFU7NoCoSQkUf7ufOyVZocLxjKmlnpbV6yZAkbNmxg4cKF9PX18corr/DlL3+Z2267jePHj2txyouA40KIE7E2TFujRSiCO4riiS6HjqgJh8HBQZqbmyN2ZCVC5tHRURobGyeUeCZj2atv/BbmNAS8xhsoXpuWDRSGggrMc+IfqhcL/uFejj62jd27d2t2TAXhAmCKMP8111zDzp07ufbaa7FYLOTn50c4SkK4Afh1PBum3M0Op/Dh9XppaGigoKAg7qaGaEQKFiWI9mCIl4zd3d20tbVNKvFMVJur/tZ7MZXNRZJS6wBN5wYKyWDGsvhszb6jgGOYn3/2kilJREU9fhxFI8p9PVVIkmQCrgC+HM/2mpM5lj60zWbj0KFDYUe4REMkC68MRLdYLGFFCYIRi8yyLHPkyBFVVzt0wkIiY2Hv+sGvcVvK0RtTO5ERwNPROC0bKAAsi85EZ9amrVEEfBRJDs5cviAm6ZJFmlNT/w/YI4Toi2fjtI2nEULQ1tZGf38/a9euTTiQFI7MyuTG2trauBQ8opFRGfhWXFzM4sWL466vDkUgEGDXrl38Zt+AZi170TBtGygA0+wlGEuqNTueb7iXt0+moZIatB4HYlWAjY2NaanM+WHidLEhTWQWQrB3716ys7PZsGGDJrOiFFc4ke6pSGRWWiBjeQuxLLPL5WL//v1c96NXMVfUxnVNU4HsceA8/HrKz5MK6HOKyJ6/RrPj+Uf6ef3r16gkjkW6ZBGPAL5GypwW4GLgU/Huk3I3e2RkBIfDQW1t7ZR6j4MVSw4fPozX6405uTEU4cio5KLjaYGMZpmV5cMNP/xbWiq8hBzA0fQawj/9GijQGbAsOVeztkbZ4+CiBfmMjIzQ3t6O2WwmLy8PWZY1H3IXD5mTmawZCiGEE0iooCFlllmZjNjV1UVOTs6UE+k6nQ6v18vu3bupqKigvr4+4R8pmMyKjJHH45mQi453/2Aon/NffvEmxrK5mt2k0eBq3UPAPpjy86QCloUb0Vs0ifSOp6HsAzz6hW3qa06nk56eHhwOh9odpwggJvLwD4dYa/FUeQTxICVk9vv9HDx4EL1ez8aNG9m7d2/MJ1osjI6OYrPZWLt2bdJC8QoZPR4P+/fvp7S0NCEZo9DhcYqX4PP5aBmRGRD5GEyJF5UkCu9AG97u5pSfJxUwls/HpOESxG/r5ej/bpvwmsVioaysDJ/Px6JFi9SpFCdOnECSJIqLi8NOEokH0dbip3oKieZkHhsbY9++fcydO5eqqirgvZLOZATyhBC0trbS399PXl7elCY+6PV6RkZGeOedd1i8eDGlpaUJ7R88PE4JmJWUlFBfX8/mhx7CpGEwJxLGGyjeTvl5UgFdVh6WBRs1O17AYePnn7sk7HsK6ZTKLaXuILSZIicnRyV3PNVisaa1JDrNRUtoTubh4eFJ/b3JNkkoon1ZWVmsW7eOPXv2TOnabDYbAwMDbNy4Mal1jbJmttvtNDQ0qAGzOTfdj3nWgildWzwQAd+0baBA0o3L/xi0EWMQfi8leidnLg//vUeyoOGaKUL7m4uLiyksLEzYk1TGD50qaE7mOXPmTCJuMmojStpJEe1TpksmA8UdHhsbo7KyMukAhU6nw2az0dbWxqpVq8jNzaXmw9/UbHhZNAghcB6dng0UAFnz12DIS7xBJRJ8w328+b+Ru6HiSU0FN1PU1NSo/c3KwAGj0ai2QCqzpKKR1ev1alYLngxSEs0ORaKWOVzaKVnXRVkfl5WVUVZWhtVqTeo4QggGBgZwu92cccYZGI1G/unOH2IomaNZE300eKfpBAoAQ3FV0jrfwQh4XQQcwwS8bg7/6DNRt00mzxza3+x2u7FarbS2tuJ0OsnPz8fn8+Hz+cK2QZ5KySBIU545kTbI5uZm3G53wmmncFAmXSgD5KxWa1Jlfoq7D1BdXY3RaGTP4TaaR/UY8lP/4403ULyb8vOkApIpG0vdmXE/jMcJO4Kkk9AZs8bnYumNII3/rSucxT0XzolpAbUoGsnKymL27NnqbDAlkKYIXYS2QGqlMiJJUiHwf8ByxlUYPy6EiKk0kTYyx9MGuX//frUTZapBhM7OTjo7OydUmyXTKKHMnJo7d1zmR9Ez23L/DkylNVO6xngwnRsoACyLz0ZnmljSGgh48Y9a0UkgGbPQmRTC6tAZs9AXhc8ICCEo8A5yy8VbYp5X6wownU5Hbm4uFouFNWvWqONdlRZIj8fDX/7yF60E8B8G/iiEuOZkfXZc68LTws22Wq00NTVRX18/5fnESi9zIBBgw4YNE4IYyY6GXb58OQUFBfT09CDLMnNu/Dbm2WlaJ0/TBgoA05wVCAF+h23cyhpM44TVGTEXJd4SKnscvP1gfGIDqSjnDM4hKy2Q5eXlCCHo7u7G6XTy7rvvsmbNGrZt28ZHP/rRhM8hSVI+cB7wUQAhhBeISxEzLZbZYDCElTYNrtcOHcEaCdEqehTrXlFRwdy5cydtl0ijRHt7O729vRO6sHQ6HRfe/QtMadC6BvC0n/4NFPqi2RhyS0EC4fMgexzIbjs6UzbZc1do1g0lhMzLX7o07u1lWZ7yMi0UkQpGJEmiqqqKyy67jKysLO6//36Gh4eTPU0tMAD8VJKkVcC7wDYhRMwn+ilzs/1+PwcOHMBsNsddrx0qdRsMpZxyqkPbZVnm0KFDAJOu644fPYtUXK1ZeiUafLZu3O2nRwOFvqACfV4Zkl4PPi8Bzxiyy47sthOwdRMIeeBIBhOWZRdqSGTBxiIvNeWFce8z1SKlZI6prJmzsrLiavyJAAOwFvj8yRE1DwN3Af8Rz46aIh43WxmsNm/ePGbPjr+zSDlOMMGUstHu7u6Y3VixyOzxeNi3bx+zZs1SVUEVtLe383a3D2PB1JYB8WC8geKNlJ8nGLrcUgyFFeNrV/+4hQ24xpDdowRG+giMxNWFB0B23SZ0WdoFBiXHEP/2kTPYtWvXhNLMaOvTVLjZaWp/7AQ6hRA7T/69nXEyx0Ta3GyFzIkIx4dCr9dPIKNiRYUQk9bH4RCNzErnVCTLni6t65Q2UOSWYCqYhaTTIfxehNf5HmnHBvGOTb3W21S5SNPAoOz3cfj7nwDGH9yjo6MMDQ3R0dGhlmaG07k+FWR2Op1TJrMQoleSpA5Jkhaf1AK7CDgUz75pc7N9Ph9NTU243e64heNDEaxaoqyPKysrmTNnzpTUSrq7uzlx4kTEGc/pkv4BcLVMsYHCnDP+0NEZwO9BeF3j61jXGGJsCO/YkHYXGwKdpZDs+Ws1O54Qgv/8f+/VcUuSREFBgTqEMLQ0Uyn3LS4uTlkALNoxx8bGEvI0o+DzwC9PRrJbgI/Fs1NayBwIBBgcHGT+/PlTSjspbnay0e9QMivKm06nM2LnVN0t38JUNi/l0j8A3v42vD1xNFAYsjCW16IzmSDgR/Y4kd12ZJcd4XHg7YzrQa4tdHpy6rWbQiGEoNg/xIc/sCXiNqGlmXa7HavVyoEDB3A4HGq1Vn5+vib10vFYZo3Gue4DEtYdSvma2WazcfDgQbKzs6mtnVq3jCRJdHd3Mzw8HHf0OxjBXU/KlIuCggJWr14d9sf+0o+fwZ87C71G8q/REHAMT2yg0Jsxls9HZ86GgB/hdRJwjweehNeJr/sUEDYKshdsQG+Z2tjeYMgeB288+Im4t5ckifz8fPLz85k3b54qJdXd3c3hw4fJycmhpKSE4uLipEsu4wmAnSoBfEiRZVa6i06cOEFfXx9r1qxRK6iShVI3a7FYklYrUaAE4KLJDTmdTn6zuwNj0dSHuUeC3+0m4BgAGWSnDVP5/PEIscuO8Djx9UyPOVHGsrmYKrRrNBFC5rW7r5jyccrKyqipqVEbKoaGhjh06BCBQECt3iooKIj7XgoEAlGXh1qsmaeClPUzNzY2YjKZ2LBhw4TWwWSgyPFkZ2dTU1MzJSL7fD7279/PypUrJwbgfD4YGwOdDvLzWfzpH2LWYJyo3+8mMDI0Pi/LlI3OlD2e2pJ06M1m9OZqnM2v4xuIKYt8WkKXlYtl4Rmatf0JIfhApY7ykqlJ7wSvb4MbKubOnYvf78dms6mjkLKzs9U2yGjeXiAQiPr+jKvNFkKwZ88eqqqqNAkGKFVYS5cuTbq2Wrmu1tZWvF4vZ5555nu91XY70q5d6HbtQvh8SEJQeViHOYF5R36/H/9wHzodQYQdr3bS680YSqoi7uvpbp62REaSTrY1ajdyR3IM8aM74nevIyFasMpgMKiNN0IIddyMIjRRWFhISUkJhYWFE44RzzhXDcX8EkZK1szxamFHgxBCddOV9fHw8HBSbZCBQIADBw5gNBqxWCzvEdlqRf+znyHGxhDl5WAyseKvJzCV1SKF3Ah+vx//aD+SkNGZstErhNXp0ev1GEojEzYS/KMD07aBAiBr3moMeYkJPERDcBpqyseKM5otSRI5OTnk5OSo7bvDw8MMDg5y7NgxzGaz2kn1vlwzh+aDE0UgEFBlh4LXx8mIHCguenV1NdXV1bz55pvjbwiB7qmnED4fVI8rhPzoQBeD+lwk++B4wb85GwxmJIWwGkrnyj43jsPTt4HCUFSJuWqpZscTQvDNzdql/5JNTYW2QSpW+8iRI4yMjOD1egkEAmHFCzRU5mwD7EAA8Ash4opsp003G6LXVStQupSqq6uZM2fOhPd0Ol1Cg9uVEs+w42o6OpB6ehA17xU4fDcwn6zy1EeuhZBxHn4D4XGm/FypgGTMwrLoLE3XycW+Aa49d4smx1OgxfVZLBYsFgvV1dU0NjZSXFyM1WpVxQuUCLnFYsHlciU1WDACPiCESKjgIPXJ05OIx1oPDQ2xd+9e6uvrJxE53mMo6OjooLm5mbVr104gshKMk44eRQTllWt2ejWbQBgL7vZG/MM9aTlXKjDe1qidcKHsHuPpf5t69DrVEEJQXFxMXV0dGzZsYMmSJeh0OlpaWti8eTNOp5Pnn3+esbGxU3J9KSFzom2QSnDq+PHjrF+/fsKQ9XiPoUAp8bRarWzYsGHSk1ItHHG74SSZF786gD439TXXAD5rN572qaXpTiXM1cs0TdcJWeYHW+bQ0dHB7t27OXbsGDabLWWzoqaC0DVzVlYWVVVVrFixgu3bt2M0Gnnrrbe4+eabp3oqAbwsSdK7kiTdHu9OaXOzIyl0BgenpjorSlHMLC0tjairrR6jtBTJ6+Vj7/TiLVmYFkVF2e3A2Tw9J1AA6PNKyZqr3ZhUIQQX1eioqahg9uzZ5OTkTEoZKevXU6mtpSBaAMxsNmMwGLj//vu1ONXZQohuSZLKgT9LknRYCPFqrJ3SSuZQq6qsj+fMmUN1dWyZ2miWOVQxMxIUMoslS9jz29/z96zwc6W0hpADOA6/ivDH1Wd++kFvHE9Daane4Rjkh/98G01NTUiShF6vp7S0lNLSUjVlFFroUVJSosr0pBvRYj5aqnIKIbpP/rdfkqRngI3AqSFzuA8cqtCpzFJWVDziQSQyK51YimJmNKiWuaCAq/xL0JnSc1O4Wt4lYE9dk0OqYanbhD5Lu7SL7Pdy6KGPA+FJEpwyqqmpUQs9ent7aW5uVsszS0pKJnl7qZS7jSaMoUXwS5KkHEAnhLCf/P9LgG/Es2/a3WxlfRxrlnI4hGuUOHbsGKOjo3F3YinHmHf7IxjypzYyJ154+1vx9hxJy7lSAdOshZq2fwohuG/zIgKBAIFAQL0vEin0UMozDxw4gCzLagVXfn6+5vOl4oGG1V8VwDMnr98A/EoI8cd4dkwrmZU1rdlsZt26dUlJoSqWWSkZtVgsrF27NqERM2fd+Rj6vMSLPJLBpAaKaQadpYDs2qkPDlcghKACK9desBVZlunu7lZjKQq5JUlSp1GEQ2h5piKu19XVpTZV+P3+iJK4qYBWypxCiBYgqcBE2tzsQCDA0aNHqaurS7rMUyFzsGJmosd65I/v4MipSs862X9yAoWcnHj/KYdOT86SczTVBZfdY7z2/U8C4wqqQ0NDrF27Vk07BgIBdeCB8v96vV4leDiEiuvZbDaam5tVSVzFHc/NzU36d4/lup/qumxIk2UeGBigp6eHysrKKdVr63Q6PB4Pe/fuTWitraC9e5A/92Qj6dJAZCFwHn0b2TWa8nOlCtm169DnFMXeME4IWWb3169Ql1pjY2OsWrVKJWmwNVbGsSpjfIG4rbbFYiEnJ4eVK1eqAgbt7e2MjY2pskPFxcUJCf7Jshz1QaCRZNCUkFIyCyFoaWnBarVSW1ub9HgZ5VidnZ24XC7OO++8pFIVF3z7eXTm9Dw9vd3N+AanaQMFYCypwTRLu/JKIQTnlXk4cuQIXq8Xk8nE8uXLI5IyuIQXSMhqB6+9QwUMFNmh9vZ2dDqdarWV8TOREE+TxYwksyRJ6po2OzubdevWMTAwgN1uT+p4sixz8OBB9ambDJHn3/Z9dAVJKyYmBP/oAK7W6dtAIZlzyK7Trq0RQO8c5LF/u52mpiaEEOTn56vi8UVFRZSWllJUVBSV3NGstt/vV7eJFEgLlh2qra3F6/UyNDREW1ubOn4m0hzneMT8ZqSb7Xa72bVrlzr0DZIbHgeTFTPfeivmlI5JOPuOH6IrmJWewhCv++QEilM7qzd5SOQsOUfT0lbZ56Hp+5+ksbGR3Nxc5s+fjyRJapeSMp3zyJEjWCwWNdcc6aEdyWor/3W73cBEtzwcTCYTlZWVVFZWIsuyarXb2towGAyq1bZYLKd9xxSkiMxms5lVq1ZNeFLFO28qGLEUM+PBI0//nV5jeXoCXkLG2fw6wjs9GygAsuauwpAfuegmUQgh+N41y9VZ1jU1E5U7QwtFHA4Hg4ODNDY2qimnsrKyqDpewVbb5XLR2trKggULVAuuEDoasXU6HYWFhRQWFrJgwQJ1aFxLSwsulwuLxYLP54tI6hntZoe6HMlMgoymmBkPvF4vD77Wm5YpjQDuE434h3vTcq5UwFAwC/McbdsaKyUblWYvFRWxg5/BKad58+bh8/mwWq10dHSo6pulpaWUlJSETTkpiq1LliyhsLBQJbMsy+o/v9+vVptFS42GDo3r6uqip6eHPXv2YDKZVKutFIo4nc4pj1Y6+R3ogXeALiHE5Ynsm7bUVLxkFkLQ3NyMy+WKqJgZD4QQLP6XJ9Fb8pPaP1H4rF14OqZvA4VkMGNZfJamKqTCZee/bl5JdXU1FRUVCe9vNBqpqKigoqJCDV4NDg6qwSvFoufk5KhErq+vV7Mc4dxxxRVX/inbxbLaFouFkpISFixYgMvlYmhoSF3zd3V10draOmXBypPYBjQBCd+4KTNZoYOpDQZDTDc7HsVMpYUx2lNVlmUW3PYwuqL0FIbI7jGczemdQKE1LIvPQmdOzgMKByHLPHrlPGprayktnboaSXDwasGCBXg8HgYHBzl+/DhjY2P4fD5qa2ujurqKO24wGCZYbYXU0VJfwe51dna2KnahrNGbm5t55ZVX+M1vfsNDDz3E/PmJ68dJklQNXAbcC3wh0f1PaaNFMBTFzAULFkR9iivFBZHI7PV6ufCLj6ArSlNhiBzA0TSNGygAc1U9xmLtHnxCCM4uGmX9+g9GbGedKsxmM1VVVRQWFtLQ0MDChQtxOBzs3r0bs9msWu1I9dLBVttoNMZMfUVaK+v1es477zyeeeYZ7r//frV4JUk8BPw7kJRcSdrIHK19UWl5m6SYGeE4gUAgrPs9NjbG9558nh5TeogMJxsoxqxpOVcqoM8tJmveak2PqbP38fBXbyI/P7VLnLGxMRobG1mxYsUEi+x0OhkcHKSpqQmfz0dxcTGlpaVRZXVjpb68Xm/UtJeSmqqrSy43L0nS5UC/EOJdSZIuSOYYaXOzw5EruOliw4YNk7pfwiGShVcioD9tltEZtJ3+FwnTvYECvQHLknORdNp9X7LPw74Hbk15ztVut3PgwAFWrlw56VwWi4Wamhq128pqtdLT06PWbStWO9L9FrrWHh0dpbe3l6VLl0Zca2ugzHk2cIUkSZcCWUC+JEm/EELcFO8B0qoBFgxFlMBkMiXUdBFKZkXFs7+/n4/99iiGXO3KD6NhujdQAFgWnoE+WztpWCEED12zLOVEHh0d5dChQ6xatSpmpsNgMEyo2x4bG2NwcJD9+/cD43XbpaWlkwbPKXA4HBw8eJCVK1eSm5sbsWDFarVOKTUlhPgy8GWAk5b5zkSIDKeIzKGKmYkg2F0PngJ50/++jr4gsWMli2nfQAGYKmoxaSDyr0AIQY1+mMvOWqnZMcNheHiYw4cPs2rVqoT7hyVJIi8vj7y8PObPn69WgJ04cYKxsTEKCgooLS1V67YdDgcNDQ0T3PhwEfI333yT48ePnxLBhGCk1M0OB2XoW1jFzDigWOZgiaCv/PJVAgXVaSoMETiPvjWtGyh02flkL9ig6TGFa5S//EAbzetIULqhVq9enfCcsXAIrQAbGRlhcHCQ1tZWdDodTqeTpUuXRrW4e/bs4d///d95++23oyrcJAIhxCvAK4nul1bL7Pf7aW5uTmromwKdTsfY2BiHDx+mrq6Oo30j7LJlpS3gNd5A0Z6Wc6UEkm5c/kevXZ+vkAPsue8azY4XDlarlaNHj2pG5FDodDqKioooKirC6XSyd+9eqqqq6Ozs5Pjx42Hrx/ft28fnP/95nn76aebOTf3s7lhIC5llWebw4cMEAgHWr18/pR/D7XbT29vL2rVrycvL44xv/SltErnTvYECIHv+WgwaKpEKITi/xMGxY8coKyujtLRUc7INDQ1x7NgxVq9enXJhP6fTSUNDw4TMSmj9+N69e+nv7+cPf/gDzz33HAsWaDc0bypIOZmD3eFE+4+DoQS6RkZGmD9/Pnl5ecz71KOajkeJhunfQAGG4mpMsxdrekyjc4DHvvtpnE4nAwMDHDx4kEAgQElJCWVlZREDS/FiYGCA1tZW1qxZE1e2YypwuVw0NDSwdOnSCZHp0Prx4eFhfvWrX1FQUMCtt97KL3/5S62qv6YEKYaCQtJ3rvI0C1bMbGhoUImYCJRAF0BeXh56vZ5/+sbvcOela50s4zjwt2lddy2ZLOStvUzjbig3Rx+8cdLrPp+PoaEhBgYG1MBSWVkZxcXFUTuPQtHf38+JEydYvXp1yuV/lKBscDloOBw5coRbbrmFX/7yl6xYsYKRkREsFstUr0+TmzhllnlgYICmpqYJipnxlHSGwuv1sm/fPsrLy5k7dy5dXV3c/eSf00ZkAPeJhmlN5PG2xrM1IbLy8Beynydv2xR2G6PRqAoCKIGlgYEBjh8/jtlsjssd7+vro729PS1EDlfXHQ6tra3ccsstPPHEE6xYsQJgSt6m1kgZmfPz8ycpZibaOaWUeNbV1amRwtauAV4fsKRF+geUBooDaTlXqmCuWY6hIHajw3temkDIAfD7wO/BjJ+KHMHN5yzho5svSOjcwYElIC53vKenh66uLtasWZN0o028cLvd7Nu3jyVLlkQlZnt7OzfeeCOPP/44a9asSek1JYuUfVNZWVn4fL4JryVC5oGBAbXEMzg1cOsvGzWdcxQNgRnQQKHPLyerZtyKqGQVAiH7EX4vOr+HXEOAFdVFfPXDF1BXPZH0fr+fhoYGysvLE64JCAeLxcLcuXNVVU1Fwsdut1NQUIBer8dut7NmzZqEXPJkEEzkaDXkXV1d3HDDDTz66KNs2KBtSk9LpGzNLMvyJDK3tbVhMpmi9rUqga6BgQFWrVo1Iegx77YfYChMj/SPkAOM7f/T9Kq71hvRmXNO/rMgGc2Y80sozTVz4cq53HXNOQnFK3w+H/v27aO6ulpVjEkVZFnm2LFj9PX1YTQa43bHk4UiDLl48eKo9Q69vb1cc801fO973+P888/X/DpO4vReM4dDLLWR4EBXcImnEIL1n3sYfYF285FjwXX8ndOOyJIxSyWrZM4enyGt11OYk83156/gSzdeAmNjGL7/fQw/+QmS34+QJCS/H79nM/6LliHiJLMi11RbWzuxGGJwEMP27Ujt7YiiIgKXX46or5/yZ+vs7MThcHDWWWeh1+tTFh2H+Inc39/Ptddey3/913+lksiaIWWWWQiB1zuxLbC7uxuPxzO511MIfM3NHN29m8LKSirPOw/ppEUWQvDNn7/EEw1j6SsM6W/B2fxmWs6lQpKQTBbVqupMFiSjCaPBQFVpHh89fwlr55WripZ2u52BgQEGBwcxGo3j0x7MZgquvRbdsWMgSaC4qbI8PlQ+Px/P888jFi2KeilKZHfRokXvqWf4/Ri/+EUMTz4JOh2SyzU+EtdoRF65Es+vfgWzkvOa2traGBkZYcWKFWFLIhV3fHBwUHXHk4mOw3hAde/evdTV1UVVBhkcHGTr1q1885vf5EMf+lDCnylBaHJjp5XM/f39jI6OsnDhwvcu4M03CfzkJziOHiUnLw+T0Qi5uchbtxK4/HIGrFbOvP9vmnb2REPAMYx930va113r9CctqgWdMQvJmIVkMKLT6Sk3wa9WzmbR+WuQ164dJ+JJCCHUwWpLliwJ+0BzuVwMDAxQ8KUvUfq3vyEZDOhO9uFOgNeLmDMH986dE84RDIfDQWNj48TIrhCYbr4Z/Z/+hOScrG8mDAbErFm4334bEizRbW1txW63R5XdDUZwdNxqtSbkjitEXrhwYVRNOZvNxtVXX81Xv/pVNm/enNDnSRLTj8xK7nHJkiUA6H7/e/w//CEjBgOF1dUYlfWxywV9ffg/8AHmtxZrOrAs6jX7vdj3vYTsSlwSWDKYkRSrarYgGbMACYNBx8p5Ffz67psTbgxQOstyc3Opra2N7pnYbGSvXIlg/EeThZgkZicJAZKE51e/Qj777EmHUNoKQ/uDdW+9hfmKK8ISWYEwm/F//vP4vv71uD6boqnucrlYunRp0k0Kijs+ODgY1R2Pl8gjIyNs3bqVO++8k6uvvjqpa0oCp/eaOZYOmOjowP3IIzhycymeNQt9sIh5VhZUVzO/UUJXkh4tYiEEziNvRySyZMp+L7hksoDRDDodWXqJ685Zxrdu36Lp9SgSShUVFfGNu33rLYRejyQEEqBjnNSKNI5KbI8H/vAHCCFzcDdSaFuh4eGHxx+wUSB5PBgeewzff/yHOsQ+EpSBf16vl2XLlk1p+RQrOl5WVkZubi6NjY0sWLAgKpHtdjvXXXcd27ZtSyeRNUPaA2BKH2jvz35GkSxTWlmp/pjKzYcQnG0vRzcrfZ1QvsEOZL8XY3nteOrLYERCR6HFyHc+dinzSowEAgHq6+tT3urm8XjUWVpxC+G5XJNKTSXGH6o6pV3v5PtDbW207tmjTlV0OBxq7XM4V1X37rvjVj0WvF6kvj5EVWQJIiEER44cQZZlli5dqunvG65Ypa+vj4aGBnJzc3G73bjd7rCf0eFwcMMNN3D77bdz/fXXa3ZN6URKyRxO1M/n8/HOO++w4vBhcufMCUvk/7FLdM9artkP/V5+VUb4feN6XT4XZrysn1fC926/lJycHPr7+xkYGMDv91NaWkp5eTlZWVknRRRyVPH2VEJZs04IPsUBMWfOuK920pUOB50kgclE2TnnkFVfz8DAAHv37sXlclFdXY3P58NsNk/+jIk8vKJsqyivAhHX/1pBp9ORm5ur1irk5OREjI673W4+/OEPc9NNN3HzzTen7JpSjbRaZpfLhdVqZc2aNeRlZak3nWBczRHG1zbfLr1o/MaLAxOrlmQI+JB9biSfm3yjzNZNi7j7pn+K61hz5sxhzpw5+Hw+BgcHOXr0KDabjcLCQk00kWNhZGSEQ4cOsXz58sTr19evRxQXI/X1QaTyR1kGnQ7/NdeQnZ2NwWDAaDSyatUqRkZGaG1txeFwUFRURFlZmdruJ591FtKOHUixCn4sFkQEMTslkGcwGKirq0v5Q1HJkc+fP19NrYVzx++44w7GxsY477zzuOGGGzQ5t9IdWFVVxfPPPz/hPSEE27Zt48UXX8RisfCzn/2MtWvXanLetJFZEe1Txo/ICxag27cPkZWlEhmgzrQJfZBo/QSyBgIQ8CJ73ej9birzDPzLlk1cda625XVGo5H8/HxOnDih1uB2dXXR1NREQUEB5eXlFBcXa+puDw0Nqf26iQbKANDp8H3965j++Z8hEHgvLaXgpNfjv+kmmDWL9vZ2hoaG1Eori8WiNukHt/vl5uZS9ZGPUPWHP0RdN4usLHyf/ezk8zL+Gx46dAiz2cyCBQtSTmS/38++ffuYO3duWMEAxR1XhP42bdqE0Wjkxhtv5Lnnnpvy+R9++GHq6+sZHZ0sYPHSSy9x9OhRjh49ys6dO/nMZz7Dzp07p3xOSIObLcsybW1tDA4Osn79et59d7wfWFx2Gbz99jhBJQkkiR+OGiFPwj/Sh8njZImzj/s/tJI5119NQ0MDtbW1U5ExjRvDw8M0NTVNsJBlZWXIsszw8LBaapqbm0t5eTklJSVTqiHu6emho6ODtWvXTqnNL3DFFXiHhzF99avg949b4pNLHclgwL91K95vfpPWlpZJ41QVBE9GFEKM57MtFrjkEir/9Cf0J+c4BUOYzYj58/F/7nOT3lOG/uXk5KSlTdDv97N3715qamqi3is+n4+Pf/zjXHDBBXzxi1/U7AHT2dnJCy+8wN13382DDz446f3nnnuOW265BUmS2LRpE8PDw0iSVCmE6JnquVNKZiW1otPpJlV0BZYuRaxfj37XLsScOSBJfC7fx+fYA0aBZO1DlJYy8MHz2b9/P8uWLUu5dCuMd+soY3FCAyU6nY7i4mKKi4vVG72/v18tUy0vL6esrCwhQra3tzM4OMjatWs1aSoI3HILrksuwfDkk+j/+lfw+5FXr8b/iU8gL1nC0aNH8fv9rFixIuYNLEkS+fn549/7L3+J+957yX74YWRA8vnAYEAnywQ+9CG8P/4xhAj5ybLMgQMHVM2tVEOxyDU1NVEDh36/n09+8pOsW7dOUyID/Mu//Avf/e53I0487erqYs6cOerf1dXVHDlypAo4vcl86NAh8vPzJ0mqyLKMDIgvfAHp0UfRvfbaeODEaASfb7xiafFiOm66ifaenrDE0hpCCNX1jIdYwTf6woULcTqd9Pf3s3//fiRJoqysjPLy8ogus5KecbvdrF69WtsI+axZ+P/t3/D/279NOF9TUxN6vZ76+vrEb2BJQvrqV3F/8YvoX3oJuaODMUmifcUKhrOyKOrupszne2+dLcs0NDRQVFSUFkkdhcixRuEEAgH++Z//mfr6er7yla9oSuTnn3+e8vJy1q1bxyuvvBJ2mwh1HZooXqSsaATGg1nBxxdC8Oabb6prGYUwUk8PujfegL4+yM8ncMYZHJMkHE4ny5cvT3n3jBJl1Sr15PF4GBgYoL+/H5/Pp0bGc3Nz1aWHEgxatGhRyteQioUMHqeq9fGVdbbNZiMnJwen08msWbOYN2+epucKh0AgwL59+5g9e3bUhpBAIMAdd9xBRUUF3/72tzX/Hr785S/z5JNPYjAYcLvdjI6OcvXVV/OLX/xC3eZTn/oUF1xwAR/+8IcBWLx4MUeOHJmthZudUjL7/f73ikROag2PjY3R29vL0NAQ2dnZlJeXU1paqvY9BwIBDh48SHZ2NgsXLkz5jZ5QlVUSUCLjAwMDaqR4dHSU0tLStKS6AoEADQ0NYceppgJ+v589e/ZgNBrx+Xzo9Xo1n51UYC8G4iWyLMt84QtfIDc3lwceeCDltQKvvPIKDzzwwKRo9gsvvMAPf/hDXnzxRXbu3Mkdd9zBrl27Tu8KsGAIIdRuKUW3WJkN1NfXp47JLC4upre3l+rqaqqiFB5oBUWfbPbs2Sk7n9FoVOVc3W63eqP39fXhdrtTEhlXoLieymjSVMPv96t66Mr5lLrxQ4cOad79FAgE2L9/v/r9RoIsy9x1112YTKa0EDkUP/7xjwH49Kc/zaWXXsqLL77IwoULsVgs/PSnP9XsPCm3zF6vF1mWx2uDo/x4yg+u9LIqUwhSpcaoFGfU1dUlPcg9ESjSNEpLoSIM19/fj9VqJScnR/VStAiEKXJLCVWRTQHKg6OqqioisZT8bn9/f9h8diJQiFxRURH1QSzLMv/5n/+J3W7nxz/+8SkXqo+A07vRAuCJJ56gtraW1atXR133KvpQK1asICcnR32a9/f3A8QMJiWKcKmnVEIZcFZfXx9W0UJNAZ1sGDCZTOpnTiZVpTw4YjUVaAWlQCNWFDkY4dbZSpovluZXvEQWQvCtb32Lnp4eHn/88ZTHXqaA05/MzzzzDL/61a9obm7mwgsv5Morr2TDhg0TUlQdHR0MDAywYsWKsDeux+Ohv7+f/v5+AoEAZWVlVFRUxJwxFAn9/f20trayatWqlEfI4b0HR2gnUjQokfGBgYG4IuOh+zY0NMSUwtEKigcwb968pGsAlBlQ/f39DA0NRV1ny7LM/v37KSsri9qAIoTgu9/9LseOHeOJJ55IuZbYFHH6k1mBy+Xij3/8I9u3b2f//v2cf/75XHbZZTz//PPccMMNrF27Ni73x+v1qhbb6/VSWlpKRUUFOTk5ca2/2tvbGRgYYOXKlSlXfIRxj6OlpWVKD45YkfFgKB5AujwOhchaDVRX4HK51KCh8pnLysrIycmhoaGB0tLSCbnaUAghePjhh9m7dy+/+tWv0vJbTxHTh8zB8Hg8PPvss9x5552Ul5ezZs0arr76as4+++yEvnQlStzf34/L5aKkpISKioqwgRWlU8fn802pdzYRdHd309XVNUnHbCoIXXMWFxdTXl5OYWEho6OjqgeQ6imM8J6sUKpd+eDPPDg4SF5eHrW1tRHX2UIIHn30UV5//XV+97vfpVw4XyNMTzID3HPPPaxatYrNmzfz97//nR07dvDGG2+wceNGtmzZwvnnn5/QjxAIBFRij42NUVxcTEVFBQUFBWqONScnJy11wTAug2Oz2Vi5cmXK1mmBQACr1Up/fz82mw2/309dXR2VlZUpf1gpqpaJdnYlC1mWaWxspLCwkNzc3AnrbEVlxGg0IoTg8ccf5+WXX2bHjh0pH2WjIaYvmcPB7/fz2muv8dRTT/GPf/yDNWvWsGXLFi688MKEXNTgm3xkZAS/309FRQV1dXUpv8lPhQeguPLz58/HZrOlJDIeDEUfLF1r8mAiB1eSha6zf/7zn+PxeOjo6ODll1+ecjzE7XZz3nnn4fF48Pv9XHPNNXw9REXllVde4corr1RLVa+++mruueeeZE43s8gcjEAgwJtvvsn27dv529/+xtKlS9myZQsXX3xx3IEvp9OpRjzdbjcjIyMp63iC9xoKsrKy0lLsAuMNGp2dnROmPgTf5IrYn1IzPlVLpQTXYk1+0AqKV5Wfnx+zkuyRRx7ht7/9LYWFhYyNjfHXv/51SssNIQQOh4Pc3Fx8Ph/nnHMODz/8MJs2vTfFI1JhSBKYPkUjiUKv13Puuedy7rnnIssyu3fv5qmnnuL+++9n4cKFXHHFFXzoQx+KGORR+oKDmzOUvG5fXx9Hjx4lLy9PTYVM1RVWhOJLSkrSNtqzs7OT/v7+SVMfggeKL1iwQI2MNzQ0AO+l+RLNBiiDx9PV8BLcpBGLyE899RTPP/88r7zyCrm5uYyNjU05biBJkpp98Pl8+Hy+tKnDJovT0jJHgizL7Nu3j+3bt/PSSy8xZ84crrjiCi699FLV5evr66OtrY2VK1dGbXIYHR1VXTSLxZK0W6pUkaVDKF5BW1sbw8PDrFixIqEHUSKR8WCkO0ouhJhQSx4Nzz77LI8++ijPP/+85t5CIBBg3bp1HDt2jM9+9rN85zvfmfD+K6+8wtatW9WKtwceeIBly5Ylc6qZ62bHA+UH3759Oy+88IKavjCZTDz44INxR8YVt7Svr4/BwUGysrJUtzTWMZT148KFCzVNzUS71mPHjuHxeKa8Jvf7/WrQMDQyHkxshciJ5MmnAiEEBw8exGKxxOx/fuGFF/je977HCy+8EFXMfqoYHh7mqquu4gc/+AHLly9XXx8dHVXliV588UW2bdvG0aNHkznF+5vMwQgEAtx+++3s2bMHs9lMXl4eV1xxBZs3b6asrCwh90ipFx8cHMRgMKhlpaHRdUWWdunSpWlZPwohOHz4MJIksXjxYk1dPlmWVRlkJbagPMyamppUDa1UQ1EkycrKijnA/OWXX+bb3/42L774Ylqq3L7+9a+Tk5PDnXfeGXGbefPm8c477yTzYJ+5a+ZE4XQ6WbFiBY899hiSJHH8+HF27NjBjTfeiMlk4oorruDKK69k1qxZMUmgKGLU1tZO6FHW6XTqetPlctHc3Jy2m1wZ26Pc5Fqv3ZTPFlwzrqzJi4qKsNvtmM3mlFZRBUsLxbLIf//737n33nt54YUXUkbkgYEBjEYjhYWFuFwu/vKXv/ClL31pwja9vb1UVFQgSRK7du1CluW0PFgiYUZY5khQBAd27NjBs88+iyzLbN68mS1btlBdnZiMr9vtpr+/n66uLlwuFzU1NVRVVaWkrS8YgUBATc2kozcY3tPQXrlyJYFAICWR8WAowglGozFmJuC1117jK1/5Ci+88AKzkhyHEw8aGhq49dZbVWno6667jnvuuWdCB9QPf/hDHn30UQwGA9nZ2Tz44IOcddZZyZwu42YnAiEEPT097Nixg2eeeQaXy8Vll13GlVdeGXcfc2dnJ729vdTX12Oz2ejv78fv96sWW2srrbQUxiuErwVsNhvNzc1hNbSV6REDAwMIIZKOjAdDWT4YDIaYRH7rrbe48847ef7559PSIptGZMg8FfT39/PMM8/w9NNPY7VaufTSS9myZUtY5Q8hBK2trYyOjk6KIPt8PgYGBujr68Pj8ag3eKwIcSykc5yqgqGhIVUMP5bl9Xq9ajOIUief6OdWiKzX62PK777zzjvccccd/P73v0+LyEKakSGzVhgaGuK5555jx44d9Pb28k//9E9cddVV1NfXqxVIZrOZJUuWRI0gB0eInU4nJSUllJeXk5+fnxCxI45TTSEGBwc5fvw4a9asSbieOd7IeDAUqSZJkmJKJ+3bt4/PfOYzPPPMM2lR+DwFyJA5FRgeHuYPf/gDTz/9NMePH8dgMHDRRRdxzz33JJTTDQQCaoOA3W6P6waHCONUU4yBgQFaW1tZvXr1lBsTZFmeUE6bn5+vVt0p359S9iqEiBmZP3DgALfddhvbt29nUYxRtNMYGTKnEna7nSuvvJJ58+Zht9tpbm7moosu4sorr2T9+vUJ5XiVG7yvr4/R0VEKCwspLy+f1PkTdpxqitHX10d7e/uEklCtIIRgZGRELc5RGiNGRkYAYhK5qamJj33sY/zmN79h6dKlml7baYYMmVMJh8PBG2+8wSWXXAJM7MluaGjg/PPP58orr2TTpk0JWWxFSL+vr4/h4WHVcik53XQVZ8B4aqWjoyMlRA6FUpxz+PBhnE6nWk4bKTJ+5MgRbrnlFn75y1+qU0VmME4/Ms+bN4+8vDz0ej0Gg4F33nkHq9XK9ddfT1tbG/PmzeN3v/tdSqt10gG3282f//xntm/fzrvvvstZZ53FVVddxdlnn51QLlaxXIpoQnFxMbNnz6a0tDTlEjfd3d10d3ezevXqtKhwKNVrPp+P+vp6NdUXLjLe2trKjTfeyM9+9jPWrJn66KF4OqBSOQMqDpyeZA6tgPn3f/93iouLueuuu7j//vux2WyTalynM7xeL3//+9/Zvn07b731ltqTfd5558W1/lQCTytXrlQjxMEyxMH64lqhq6uL3t7emNpsWkEIwfHjx9Uy1FDXWlGQ2b9/P3fffTc+n49vfOMbfOQjH9GkQCaeDqgXX3yRH/zgB6oE7rZt2zSbARUHpgeZFy9ezCuvvEJlZSU9PT1ccMEF6ljPmQa/38+rr77KU089xWuvvab2ZH/gAx8I21+rjMIJDTwpN59SVmoymaioqIirXjwWFM21VatWpU3g7vjx47jd7pjzmLu6uvjwhz/MFVdcQVNTE1VVVTzwwAOaXovT6eScc87h0Ucf5YwzzlBfDydOr9y3acDpV84pSRKXXHIJkiTxqU99ittvv52+vj71C6msrFQVN2ciDAYDF154IRdeeCGBQIA33niDHTt28LWvfY1ly5axZcsWPvjBD2KxWDh48CButzvsKByl/S43N5cFCxbgcDjo7+9n7969ar14MlVY7e3tWK1W7cfhREFLSwsul4tly5ZFJXJvby/XX389Dz30EOedd57m1xHaARVMZAg/A6qrqyttOX4toCmZ33jjDWbPnk1/fz8XX3wxS5Ys0fLw0wp6vZ7zzjuP8847D1mW2bVrF9u3b+fb3/42ubm56PV6fvvb38blQufkjA96nz9/Pi6XS+1PliRJbQSJpazR1tbGyMgIK1euTBuRlXnPy5cvj0rk/v5+rr32Wv7rv/4rJUSG8d9j3759agfUgQMHJnRAhfNQT/f+5VBo+qsqUwzKy8u56qqr2LVrFxUVFfT0jI/R6enpSctI1tMNOp2OTZs28cADD7B161ZycnI488wzufTSS/nwhz/Mr3/9azVdEwvZ2dnMnTuXDRs2qJMcDx48yK5du2hra8PpdE7ap6WlRa1eSyeR7XZ7TIs8ODjItddey7333stFF12U8usqLCzkggsu4I9//OOE16urq+no6FD/7uzsTMsUEC2h2S/rcDjUMZYOh4OXX36Z5cuXc8UVV/DEE08A46L4V155pVannJbYuHEjL730Et/5znfYs2cP3/rWtzhx4gSbN29m69at/PznP8dqtcZ1LLPZzJw5c1i3bp2aXjp8+DA7d+6kpaUFu93OsWPH1K6ydBG5ra0Nu93O8uXLo57TZrNx7bXXcs899/ChD30oZdczMDDA8PAwgNoBFeo1XnHFFfz85z9HCMHbb79NQUHBtHKxQcMAWEtLC1dddRUwHgi68cYbufvuuxkaGuK6666jvb2dmpoannrqqbgrm4aHh7nttts4cOAAkiTxk5/8hMWLF8+4VBe8V6e8fft2VTXjiiuu4PLLL0+4J1upF29tbcXn81FVVRVRhlhrnDhxQlVBiUbkkZERtm7dyp133snVV1+d0muKpwNKCMHnPvc5/vjHP6ozoNavX5/S6wrC6RfN1hq33nor5557Lrfddhterxen08l99903o1Nd8F4qZ8eOHTz33HOYzWY2b94cd0+2Ui4pyzILFy5Uq88cDodaL15QUKA5sdvb27HZbDGJbLfbueaaa/jc5z7H9ddfr+k1TFPMbDKPjo6yatUqWlpaJtx076dUF0zsyX7mmWcAuPzyyyP2ZCsWXqfTTWpgUGSI+/r6sNvtFBUVqWWlUyW2EimPFWBzOBxcd911fPzjH+fmm2+e0jlnEGY2mfft28ftt9/O0qVL2b9/P+vWrePhhx+mqqpKXf8AFBUVYbPZTtVlphXBPdlPP/00brebyy+/XNVulmWZhoYGcnNzY/YGK4Pb+vr6VKmgioqKpCYydnR0MDg4yKpVq6Lu63K5uO666/jIRz7Cxz/+8YTOMcMxs8n8zjvvsGnTJt544w3OOOMMtm3bRn5+Pj/4wQ/et2QOhhBiQk+2zWbDYDBwwQUXcPfddydESCGEKrZgs9nIy8ujoqJiQqdTJHR2dqpFKNHO6Xa7ufHGG9myZQuf+tSnpl3aJ8WY2WTu7e1l06ZNtLW1AeNyMffffz/Hjh17X7nZ8cDv93P99dcjyzJer5e+vr4JPdmJECdcp1NFRUXYenFFJyxWNZnX6+Wmm27ikksu4fOf/3yGyJMxs8kMcO655/J///d/LF68mK997Ws4HA4ASkpK1ACY1Wrlu9/97qm8zFMOu93OM888wy233AKMZwF+//vf8/TTT3PixAkuvvhitmzZknDBiDI3WtEAy8rKUond399PX19fTCL7fD4++tGPcvbZZ/PFL34xQ+TwmPlk3rdvnxrJrq2t5ac//amaWkgm1dXc3DwhetrS0sI3vvENbrnllhmZ7oJxor/wwgvs2LGDI0eOqD3Z69atS3htrIy96e7uxu/3s2DBAioqKiI2lPj9fj7xiU+wZs0avvzlL2eIHBkzn8ypRCAQoKqqip07d/LII4/M+HQXjDcZvPTSS+zYsYMDBw6oPdlnnHFG3E0XPT09dHd3s2jRIlVJRa/Xq2WlSr14IBDg05/+NAsXLuRrX/uaJkTu6Ojglltuobe3F51Ox+233862bdsmbKPhMLd0IkPmqeDll1/m61//Om+88cb7Lt0FE3uy9+zZo/Zkn3XWWRHrxRUih7ZOKr3JSgDtzTffpKOjg7lz53LfffdpZpF7enro6elh7dq12O121q1bx7PPPjtBhUTDYW7phCZfUHrq+05D/OY3v1Hb3d5PnV0KsrKy2Lx5M0888QTvvvsuV111FTt27OCss87i85//PH/961/xer3q9r29verw+FArnpWVRU1NDevXr2fVqlUcPnyYt99+m3/84x/89Kc/1eyaKysrVcGAvLw86uvr6erq0uz40x3vS8vs9XqZPXs2Bw8epKKigsLCwky66yRCe7LXrl1LRUUFdrud7373u1G7vGRZ5q677gLg+9//PsPDw7S1taVEsaOtrY3zzjtPHfmqQMNhbumENq6LECLavxmJZ599Vlx88cXq34sWLRLd3d1CCCG6u7vFokWLTtWlnVbw+/3im9/8pqiurharV68WN9xwg/j1r38tBgYGhMPhmPDPbreLf/3XfxWf/OQnRSAQSOl12e12sXbtWrFjx45J742MjAi73S6EEOKFF14QCxcuTOm1aIRYPIzr3/vSzf71r3+tuthAprMrAoQQtLS0cODAAd599122bdvG7t27ueiii7jlllt4+umnGRsbQwjBt771LaxWK48++mhKu7N8Ph9bt27lIx/5SNgGjfz8fFUQ8dJLL8Xn8zE4OJiy6zmtEIPtMw4Oh0MUFxeL4eFh9bXBwUFx4YUXioULF4oLL7xQDA0NJXzcBx98UCxdulQsW7ZM3HDDDcLlcomhoSHxwQ9+UCxcuFB88IMfFFarVcuPcsoQCATEO++8I770pS+J1atXi6VLl4otW7YIv9+f0vPKsixuvvlmsW3btojb9PT0CFmWhRBC7Ny5U8yZM0f9+zSGJpb5fblm1hpdXV2cc845HDp0iOzsbK677jouvfRSDh06NONTXrIs8/zzz3PhhRemXCL49ddf59xzz53QlXXffffR3t4OaD7MLZ3IrJlPF3R2dorq6moxNDQkfD6fuOyyy8Sf/vSnzFo8g3iRWTOfLqiqquLOO++kpqaGyspKCgoKuOSSS96XKa8MTh0yZNYANpuN5557jtbWVrq7u3E4HPziF7841ZeVwfsMGTJrgL/85S/Mnz9f1bW++uqrefPNNzNihhmkFRkya4CamhrefvttnE4nQgj++te/Ul9fn0l5ZZBWZMisAc444wyuueYa1q5dy4oVK5Blmdtvv5277rqLP//5z9TV1fHnP/9ZrY6KFw8//DDLly9n2bJlPPTQQwBYrVYuvvhi6urquPjii9+3lWoZTEYmNXWa4sCBA9xwww3s2rULk8nEhz70IR599FEee+yxGZ/ueh8i02gxk9HU1MSmTZuwWCwYDAbOP/98nnnmGZ577jluvfVWYFy99Nlnnz21F5rBaYMMmU9TLF++nFdffZWhoSGcTicvvvgiHR0d0z7d1dHRwQc+8AHq6+tZtmwZDz/88KRthBDccccdLFy4kJUrV7Jnz55TcKXTD6kfzJtBUqivr+dLX/oSF198Mbm5uaxatSotc5RTDYPBwH//939P6Em++OKLJ/Qkv/TSSxw9epSjR4+yc+dOPvOZz6RzvOq0RcYyn8b4xCc+wZ49e3j11VcpLi6mrq5u2qe74ulJfu6557jllluQJIlNmzYxPDysfuYMIiND5tMYigvd3t7O008/rc4uninprra2Nvbu3Rv3eNUMomP6+20zGFu3bmVoaAij0cgjjzxCUVERd911F9dddx2PP/64Kmg4HTE2NsbWrVt56KGHJogLwMwYr3oqECs1lcEMgSRJPwEuB/qFEMtPvlYM/BaYB7QB1wkhbCff+zLwCSAA3CGE+JOG12IEngf+JIR4MMz7/wO8IoT49cm/m4ELhBAZXzsKMm72+wc/A0Lnpt4F/FUIUQf89eTfSJK0FLgBWHZynx9JkhSffGcMSOMm9nGgKRyRT+L3wC3SODYBIxkix0bGzX6fQAjxqiRJ80JevhK44OT/PwG8Anzp5Ou/EUJ4gFZJko4BG4G3NLiUs4GbgUZJkvadfO0rQM3J6/wx8CJwKXAMcAIf0+C8Mx4ZMr+/UaFYPCFEjyRJSmi8Cng7aLvOk69NGUKI14lR8STG136f1eJ87ydk3OwMwiEc2TLBldMcGTK/v9EnSVIlwMn/KuVkncCcoO2qge40X1sGCSJD5vc3fg/cevL/bwWeC3r9BkmSzJIkzQfqgF2n4PoySACZNfP7BJIk/ZrxYFepJEmdwH8C9wO/kyTpE0A7cC2AEOKgJEm/Aw4BfuCzQojAKbnwDOJGJs+cQQYzBBk3O4MMZggyZM4ggxmCDJkzyGCGIEPmDDKYIciQOYMMZggyZM4ggxmCDJkzyGCGIEPmDDKYIfj/taQbDJ4u+H4AAAAASUVORK5CYII=\n",
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
    "ax.scatter(x_data[:,0],x_data[:,1],y_data,c = 'r',marker='o',s=100)\n",
    "x0 = x_data[:,0]\n",
    "x1 = x_data[:,1]\n",
    "x0,x1 = np.meshgrid(x0,x1)\n",
    "z = model.intercept_ + x0*model.coef_[0] + x1*model.coef_[1]\n",
    "\n",
    "ax.plot_surface(x0,x1,z)"
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
