# Activation Functions with
<br><img src=https://devblogs.nvidia.com/wp-content/uploads/2017/04/pytorch-logo-dark.png width='60%'  img> <img src=https://matplotlib.org/_static/logo2_compressed.svg width='30%' align=right img>
<br>
<br>


## ReLU


<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$ReLU(x)&space;=&space;max(0,&space;x)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$ReLU(x)&space;=&space;max(0,&space;x)$$" title="\large $$ReLU(x) = max(0, x)$$" height=25 /></a>

<img src=https://media.discordapp.net/attachments/661234508078776333/661595342969503744/relu.png img>

## Leaky ReLU

<img src=https://media.discordapp.net/attachments/661234508078776333/661609983292735509/movie.gif img>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{LeakyReLU}(x)&space;=&space;\max(0,&space;x)&space;&plus;&space;\text{negative&space;slope}&space;*&space;\min(0,&space;x)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{LeakyReLU}(x)&space;=&space;\max(0,&space;x)&space;&plus;&space;\text{negative&space;slope}&space;*&space;\min(0,&space;x)$$" title="\large $$\text{LeakyReLU}(x) = \max(0, x) + \text{negative slope} * \min(0, x)$$" height=25 /></a>

## Randomized Leaky ReLU
```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.arange(-4, 4, 0.1)
y = nn.RReLU(-.2, -.1)(x)

fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()
```

<img src=https://media.discordapp.net/attachments/661234508078776333/661620841628565504/movie.gif img>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$&space;RReLU(x)&space;=&space;\begin{cases}&space;x&space;&&space;\text{if&space;}&space;x&space;\geq&space;0&space;\\&space;0.1&space;αx&space;&&space;\text{otherwise},\&space;α&space;=&space;random(0.125,&space;0.333)&space;\end{cases}&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$&space;RReLU(x)&space;=&space;\begin{cases}&space;x&space;&&space;\text{if&space;}&space;x&space;\geq&space;0&space;\\&space;0.1&space;αx&space;&&space;\text{otherwise},\&space;α&space;=&space;random(0.125,&space;0.333)&space;\end{cases}&space;$$" title="\large $$ RReLU(x) = \begin{cases} x & \text{if } x \geq 0 \\ 0.1 αx & \text{otherwise},\ α = random(0.125, 0.333) \end{cases} $$" height=75/></a>

## RELU6

<img src=https://user-images.githubusercontent.com/46652050/71629309-c2cfae00-2bca-11ea-9de9-40fcb9bc4040.png img>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{ReLU6}(x)&space;=&space;\min(\max(0,x),&space;6)$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{ReLU6}(x)&space;=&space;\min(\max(0,x),&space;6)$$" title="\large $$\text{ReLU6}(x) = \min(\max(0,x), 6)$$" height=25/></a>

## SELU
<img src=https://user-images.githubusercontent.com/46652050/71629913-0bd53180-2bce-11ea-9286-ec596870e616.png img>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\large&space;$$\text{SELU}(x)&space;=&space;\text{scale}&space;*&space;(\max(0,x)&space;&plus;&space;\min(0,&space;\alpha&space;*&space;(\exp(x)&space;-&space;1)))$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;$$\text{SELU}(x)&space;=&space;\text{scale}&space;*&space;(\max(0,x)&space;&plus;&space;\min(0,&space;\alpha&space;*&space;(\exp(x)&space;-&space;1)))$$" title="\large $$\text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))$$" height=25 /></a>

## ELU

