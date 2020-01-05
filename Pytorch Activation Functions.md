# Activation Functions with
<br><img src=https://devblogs.nvidia.com/wp-content/uploads/2017/04/pytorch-logo-dark.png width='60%'  img> <img src=https://matplotlib.org/_static/logo2_compressed.svg width='30%' align=right img>
<br>
<br>
```
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
```

## ReLU


<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$ReLU(x)&space;=&space;max(0,&space;x)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$ReLU(x)&space;=&space;max(0,&space;x)$$" title="\large $$ReLU(x) = max(0, x)$$" height=20 /></a>
```
x = torch.arange(-10, 10, 0.2)
y = nn.ReLU()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('RELU')
plt.show()
```
<img src=https://media.discordapp.net/attachments/661234508078776333/661595342969503744/relu.png img>

## Leaky ReLU
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{LeakyReLU}(x)&space;=&space;\max(0,&space;x)&space;&plus;&space;\text{negative&space;slope}&space;*&space;\min(0,&space;x)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{LeakyReLU}(x)&space;=&space;\max(0,&space;x)&space;&plus;&space;\text{negative&space;slope}&space;*&space;\min(0,&space;x)$$" title="\large $$\text{LeakyReLU}(x) = \max(0, x) + \text{negative slope} * \min(0, x)$$" height=20 /></a>
```
x = torch.arange(-10, 10, 0.2)
y = nn.LeakyReLU(.2)(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
set_theme()
plt.title('Leaky ReLU (0.2 slope)')
plt.show()
```
<img src=https://media.discordapp.net/attachments/661234508078776333/661609983292735509/movie.gif img>


## Randomized Leaky ReLU


<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$&space;RReLU(x)&space;=&space;\begin{cases}&space;x&space;&&space;\text{if&space;}&space;x&space;\geq&space;0&space;\\&space;0.1&space;αx&space;&&space;\text{otherwise},\&space;α&space;=&space;random(0.125,&space;0.333)&space;\end{cases}&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$&space;RReLU(x)&space;=&space;\begin{cases}&space;x&space;&&space;\text{if&space;}&space;x&space;\geq&space;0&space;\\&space;0.1&space;αx&space;&&space;\text{otherwise},\&space;α&space;=&space;random(0.125,&space;0.333)&space;\end{cases}&space;$$" title="\large $$ RReLU(x) = \begin{cases} x & \text{if } x \geq 0 \\ 0.1 αx & \text{otherwise},\ α = random(0.125, 0.333) \end{cases} $$" height=75/></a>

```
x = torch.arange(-4, 4, 0.1)
y = nn.RReLU(-.2, -.1)(x)

fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()
```

<img src=https://media.discordapp.net/attachments/661234508078776333/662328213372141568/movie.gif img>


## RELU6

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{ReLU6}(x)&space;=&space;\min(\max(0,x),&space;6)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{ReLU6}(x)&space;=&space;\min(\max(0,x),&space;6)$$" title="\large $$\text{ReLU6}(x) = \min(\max(0,x), 6)$$" height=25/></a>

```
x = torch.arange(-10, 10, 0.2)
y = nn.ReLU6()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('ReLU6')
plt.show()
```

<img src=https://user-images.githubusercontent.com/46652050/71629309-c2cfae00-2bca-11ea-9de9-40fcb9bc4040.png img>

## SELU

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{SELU}(x)&space;=&space;\text{scale}&space;*&space;(\max(0,x)&space;&plus;&space;\min(0,&space;\alpha&space;*&space;(\exp(x)&space;-&space;1)))$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{SELU}(x)&space;=&space;\text{scale}&space;*&space;(\max(0,x)&space;&plus;&space;\min(0,&space;\alpha&space;*&space;(\exp(x)&space;-&space;1)))$$" title="\large $$\text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))$$" height=25 /></a>
```
x = torch.arange(-4, 4, 0.1)
y = nn.SELU()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('SELU')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71629913-0bd53180-2bce-11ea-9286-ec596870e616.png img>

## ELU

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{ELU}(x)&space;=&space;\max(0,x)&space;&plus;&space;\min(0,&space;\alpha&space;*&space;(\exp(x)&space;-&space;1))$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{ELU}(x)&space;=&space;\max(0,x)&space;&plus;&space;\min(0,&space;\alpha&space;*&space;(\exp(x)&space;-&space;1))$$" title="\large $$\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))$$" height=25/></a>

```
x = torch.arange(-5, 5, 0.2)
y = nn.ELU(alpha=.5)(x)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('ELU')
plt.show()
```

<img src=https://media.discordapp.net/attachments/661234508078776333/661964814767095827/movie.gif img>

## GLU
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$${GLU}(a,&space;b)=&space;a&space;\otimes&space;\sigma(b)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$${GLU}(a,&space;b)=&space;a&space;\otimes&space;\sigma(b)$$" title="\large $${GLU}(a, b)= a \otimes \sigma(b)$$" height=25/></a>

```
x = torch.linspace(-5, 5, 64)
y = nn.GLU()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x[::2], y)
plt.title('GLU')
plt.show()
```

<img src=https://user-images.githubusercontent.com/46652050/71643488-6337d800-2c88-11ea-8868-fb2e3180b174.png img>

## Sigmoid

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{Sigmoid}(x)&space;=&space;\frac{1}{1&space;&plus;&space;\exp(-x)}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{Sigmoid}(x)&space;=&space;\frac{1}{1&space;&plus;&space;\exp(-x)}$$" title="\large $$\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}$$" height=75/></a>
```
x = torch.linspace(-5, 5, 64)
y = nn.Sigmoid()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('Sigmoid')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71643495-8b273b80-2c88-11ea-8683-50dd2cc9a3b8.png img>

## Softmax
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{Softmax}(x_{i})&space;=&space;\frac{\exp(x_i)}{\sum_j&space;\exp(x_j)}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{Softmax}(x_{i})&space;=&space;\frac{\exp(x_i)}{\sum_j&space;\exp(x_j)}$$" title="\large $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$" height=75/></a>

```
x = torch.linspace(-5, 5, 15*15).view(15, 15)
y = nn.Softmax(1)(x) 
z = nn.Softmax(0)(x)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.title('Softmax')
plt.show()
```

<img src=https://user-images.githubusercontent.com/46652050/71643530-f83ad100-2c88-11ea-96cc-22683b70452c.png img>

## Softsign

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$&space;\text{SoftSign}(x)&space;=&space;\frac{x}{&space;1&space;&plus;&space;|x|}&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$&space;\text{SoftSign}(x)&space;=&space;\frac{x}{&space;1&space;&plus;&space;|x|}&space;$$" title="\large $$ \text{SoftSign}(x) = \frac{x}{ 1 + |x|} $$" height=75/></a>

```
x = torch.linspace(-4, 4, 50)
y = nn.Softsign()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('Softsign')
plt.show()
```

<img src=https://user-images.githubusercontent.com/46652050/71643550-2b7d6000-2c89-11ea-9d84-123c0886dc84.png img>

## Softplus
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$&space;\text{Softplus}(x)&space;=&space;\frac{1}{\beta}&space;*&space;\log(1&space;&plus;&space;\exp(\beta&space;*&space;x))&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$&space;\text{Softplus}(x)&space;=&space;\frac{1}{\beta}&space;*&space;\log(1&space;&plus;&space;\exp(\beta&space;*&space;x))&space;$$" title="\large $$ \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) $$" height=75/></a>

```
x = torch.arange(-5, 5, 0.2)
y = nn.Softplus()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('Softplus')
plt.show()
```

<img src=https://user-images.githubusercontent.com/46652050/71643567-5b2c6800-2c89-11ea-8390-6ed0c5d9a061.png img>

## Softshrink

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$&space;\text{SoftShrinkage}(x)&space;=&space;\begin{cases}&space;x&space;-&space;\lambda,&space;&&space;\text{&space;if&space;}&space;x&space;>&space;\lambda&space;\\&space;x&space;&plus;&space;\lambda,&space;&&space;\text{&space;if&space;}&space;x&space;<&space;-\lambda&space;\\&space;0,&space;&&space;\text{&space;otherwise&space;}&space;\end{cases}&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$&space;\text{SoftShrinkage}(x)&space;=&space;\begin{cases}&space;x&space;-&space;\lambda,&space;&&space;\text{&space;if&space;}&space;x&space;>&space;\lambda&space;\\&space;x&space;&plus;&space;\lambda,&space;&&space;\text{&space;if&space;}&space;x&space;<&space;-\lambda&space;\\&space;0,&space;&&space;\text{&space;otherwise&space;}&space;\end{cases}&space;$$" title="\large $$ \text{SoftShrinkage}(x) = \begin{cases} x - \lambda, & \text{ if } x > \lambda \\ x + \lambda, & \text{ if } x < -\lambda \\ 0, & \text{ otherwise } \end{cases} $$" height=75/></a>
```
x = torch.arange(-2, 2, 0.1)
y = nn.Softshrink()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('Softshrink')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71643577-77300980-2c89-11ea-8ff8-b2d8efed1415.png img>

## Hardshrink

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$&space;\text{HardShrink}(x)&space;=&space;\begin{cases}&space;x,&space;&&space;\text{&space;if&space;}&space;x&space;>&space;\lambda&space;\\&space;x,&space;&&space;\text{&space;if&space;}&space;x&space;<&space;-\lambda&space;\\&space;0,&space;&&space;\text{&space;otherwise&space;}&space;\end{cases}&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$&space;\text{HardShrink}(x)&space;=&space;\begin{cases}&space;x,&space;&&space;\text{&space;if&space;}&space;x&space;>&space;\lambda&space;\\&space;x,&space;&&space;\text{&space;if&space;}&space;x&space;<&space;-\lambda&space;\\&space;0,&space;&&space;\text{&space;otherwise&space;}&space;\end{cases}&space;$$" title="\large $$ \text{HardShrink}(x) = \begin{cases} x, & \text{ if } x > \lambda \\ x, & \text{ if } x < -\lambda \\ 0, & \text{ otherwise } \end{cases} $$" height=75/></a>
```
x = torch.arange(-4, 4, 0.1)
y = nn.Hardshrink(1)(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('Hardshrink')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71643596-9fb80380-2c89-11ea-959d-641a88f38282.png img>

## TANH
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{Tanh}(x)&space;=&space;\frac{e^x&space;-&space;e^{-x}}&space;{e^x&space;&plus;&space;e^{-x}}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{Tanh}(x)&space;=&space;\frac{e^x&space;-&space;e^{-x}}&space;{e^x&space;&plus;&space;e^{-x}}$$" title="\large $$\text{Tanh}(x) = \frac{e^x - e^{-x}} {e^x + e^{-x}}$$" height=75/></a>
```
x = torch.arange(-3, 3, 0.1)
y = nn.Tanh()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('TANH Shrink')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71643678-5d42f680-2c8a-11ea-9082-97c6eff831f4.png img>

## Tanhshrink
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{Tanhshrink}(x)&space;=&space;x&space;-&space;\text{Tanh}(x)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{Tanhshrink}(x)&space;=&space;x&space;-&space;\text{Tanh}(x)$$" title="\large $$\text{Tanhshrink}(x) = x - \text{Tanh}(x)$$" height=25/></a>
```
x = torch.arange(-3, 3, 0.1)
y = nn.Tanhshrink()(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('TANH Shrink')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71677740-95f9d300-2d51-11ea-881f-1280b642e3da.png img>

## Hardtanh

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{HardTanh}(x)&space;=&space;\begin{cases}&space;1&space;&&space;\text{&space;if&space;}&space;x&space;>&space;1&space;\\&space;-1&space;&&space;\text{&space;if&space;}&space;x&space;<&space;-1&space;\\&space;x&space;&&space;\text{&space;otherwise&space;}&space;\\&space;\end{cases}$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{HardTanh}(x)&space;=&space;\begin{cases}&space;1&space;&&space;\text{&space;if&space;}&space;x&space;>&space;1&space;\\&space;-1&space;&&space;\text{&space;if&space;}&space;x&space;<&space;-1&space;\\&space;x&space;&&space;\text{&space;otherwise&space;}&space;\\&space;\end{cases}$$" title="\large $$\text{HardTanh}(x) = \begin{cases} 1 & \text{ if } x > 1 \\ -1 & \text{ if } x < -1 \\ x & \text{ otherwise } \\ \end{cases}$$" height=100/></a>
```
x = torch.arange(-4, 4, 0.1)
y = nn.Hardtanh(-2, 1.5)(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('Hard Tanh')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71677868-d3f6f700-2d51-11ea-86a1-9717877cfeae.png img>

## CELU 
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{CELU}(x)&space;=&space;\max(0,x)&space;&plus;&space;\min(0,&space;\alpha&space;*&space;(\exp(x/\alpha)&space;-&space;1))$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{CELU}(x)&space;=&space;\max(0,x)&space;&plus;&space;\min(0,&space;\alpha&space;*&space;(\exp(x/\alpha)&space;-&space;1))$$" title="\large $$\text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))$$" height=12/></a>
```
x = torch.arange(-4, 4, 0.1)
y = nn.CELU(-2)(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('CELU')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71677956-fbe65a80-2d51-11ea-9266-1ab125d3a745.png img>

## Log Sigmoid
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$&space;\text{LogSigmoid}(x)&space;=&space;\log\left(\frac{&space;1&space;}{&space;1&space;&plus;&space;\exp(-x)}\right)&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$&space;\text{LogSigmoid}(x)&space;=&space;\log\left(\frac{&space;1&space;}{&space;1&space;&plus;&space;\exp(-x)}\right)&space;$$" title="\large $$ \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right) $$" height=25/></a>
```
x = torch.arange(-5, 5, 0.2)
y = nn.LogSigmoid()(x)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('Log Sigmoid')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71678033-233d2780-2d52-11ea-908c-6c110ca11c4e.png img>

## Threshold
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$&space;y&space;=&space;\begin{cases}&space;x,&space;&\text{&space;if&space;}&space;x&space;>&space;\text{threshold}&space;\\&space;\text{value},&space;&\text{&space;otherwise&space;}&space;\end{cases}&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$&space;y&space;=&space;\begin{cases}&space;x,&space;&\text{&space;if&space;}&space;x&space;>&space;\text{threshold}&space;\\&space;\text{value},&space;&\text{&space;otherwise&space;}&space;\end{cases}&space;$$" title="\large $$ y = \begin{cases} x, &\text{ if } x > \text{threshold} \\ \text{value}, &\text{ otherwise } \end{cases} $$" height=75/></a>
```
x = torch.arange(-5, 5, 0.2)
y = nn.Threshold(value=1, threshold=-1)(x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
plt.title('Threshold (value=1, threshold=-1)')
plt.show()
```
<img src=https://user-images.githubusercontent.com/46652050/71678087-4a93f480-2d52-11ea-89c7-b1bad6f63cdd.png img>
