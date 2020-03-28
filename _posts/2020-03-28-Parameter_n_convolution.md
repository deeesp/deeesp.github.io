---
title:  "[03-1] Visualization of neural networks parameter transformation and fundamental concepts of convolution"
excerpt: "CNN 개념잡기"
categories:
  - Deep Learning
  
tags:
  - Deep Learning
  - 딥러닝
  - 인공지능
  
last_modified_at: 2020-03-28

toc: true
toc_sticky: true
---


## [신경망 시각화](https://youtu.be/FW5gFiJb-ig)

<!--
In this section we will visualise the inner workings of a neural network.
-->
이번 절에서는 신경망이 어떻게 동작하는지 내부를 시각화 해볼 것이다.

<center><img src="{{site.baseurl}}/images/week03/03-1/Network.png" alt="Network" style="zoom:35%;" /><br>
그림. 1 신경망 구조</center><br>  
  
  
<!--Figure 1 depicts the structure of the neural network we would like to visualise. Typically, when we draw the structure of a neural network, the input appears on the bottom or on the left, and the output appears on the top side or on the right. In Figure 1, the pink neurons represent the inputs, and the blue neurons represent the outputs. In this network, we have 4 hidden layers (in green), which means we have 6 layers in total (4 hidden layers + 1 input layer + 1 output layer). In this case, we have 2 neurons per hidden layer, and hence the dimension of the weight matrix ($W$) for each layer is 2-by-2. This is because we want to transform our input plane into another plane that we can visualize.
-->
그림 1 은 우리가 보고자 하는 신경망의 구조를 묘사한 것이다. 일반적으로 신경망의 구조를 그릴 때, 입력은 아래나 왼쪽에 나타나고, 출력은 위나 오른쪽에 나타난다. 그림 1을 보면, 분홍색 뉴런은 입력을 나타내고, 파란색 뉴런은 출력을 나타낸다. 이 신경망은 4개 층의 초록색 은닉층<sup>Hidden Layers</sup>을 가지고 있다. 즉, 총 6개(4개의 은닉 층과 1개의 입력 층, 1개의 출력 층) 계층을 가지고 있다. 여기서는, 각 은닉층마다 2개의 뉴런을 가지고 있으므로, 각 층별 가중치 행렬 ($W$)의 차원은 2X2가 된다. 이는 입력 평면을 시각화 하고자 하는 평면으로 변환하기 위함이다.  
  
  
<center><img src="{{site.baseurl}}/images/week03/03-1/Visual1.png" alt="Network" style="zoom:35%;" /><br>
그림. 2 접힘 평면<sup>Folding space</sup></center><br>  
  
<!-- The transformation of each layer is like folding our plane in some specific regions as shown in Figure 2. This folding is very abrupt, this is because all the transformations are performed in the 2D layer. In the experiment, we find that if we have only 2 neurons in each hidden layer, the optimization will take longer; the optimization is easier if we have more neurons in the hidden layers. This leaves us with an important question to consider: Why is it harder to train the network with fewer neurons in the hidden layers? You should consider this question yourself and we will return to it after the visualization of $\texttt{ReLU}$.
-->
각 층의 변환은 그림 2에서 보여지는 것처럼 특정 지역에서 평면을 접는 것과 같다. 2차원의 계층에서 모든 변환이 이루어지기 때문에, 매우 갑작스럽게 접히게 된다. 우리는 실험을 통해 만약 2개의 뉴런만 각 은닉층에 있다면 최적화가 더 오래 걸리고, 은닉층에 더 많은 뉴런이 있을 수록 최적화가 더 쉬워진다는 것을 발견하였다. 여기서 우리는 "왜 더 적은 뉴런으로 신경망을 학습시키는 것이 더 어려울까?"라는 의문점을 남기게 될 것이다. 이러한 질문에 우리는 스스로 생각해 보아야 하며, $\texttt{ReLU}$를 시각화 해본 뒤에 다시 돌아와 볼 것이다.  
  

| <img src="{{site.baseurl}}/images/week03/03-1/Visual2a.png" alt="Network" style="zoom:45%;" /> | <img src="{{site.baseurl}}/images/week03/03-1/Visual2b.png" alt="Network" style="zoom:45%;" /> |
|(a)|(b)|
<center>그림. 3 ReLU 연산</center><br>
   
   
<!--
When we step through the network one hidden layer at a time, we see that with each layer we perform some affine transformation followed by applying the non-linear ReLU operation, which eliminates any negative values. In Figures 3(a) and (b), we can see the visualisation of ReLU operator. The ReLU operator helps us to do non-linear transformations. After mutliple steps of performing an affine transformation followed by the ReLU operator, we are eventually able to linearly separate the data as can be seen in Figure 4.
-->
신경망의 은닉층을 하나씩 살펴보면, 각 계층마다 어떤 아핀 변환<sup>Affine Transformation</sup>을 수행한 다음, 비선형 ReLU 연산을 하여 음수 값을 모두 제거하는 것을 볼 수 있다. 그림 3(a)와 (b)는 ReLU 연산을 시각화한 것이다. ReLU 연산은 비선형 변환하는 데에 쓰인다. ReLU 연산 후 아핀 변환을 수행하는 과정을 여러번 거친 후, 그림 4에서 볼 수 있듯이 데이터를 선형적으로 분리할 수 있다.  
  
<br><center><img src="{{site.baseurl}}/images/week03/03-1/Visual3.png" alt="Network" style="zoom:30%;" /><br>
그림. 4 출력 시각화</center><br>  
  
  
<!--
This provides us with some insight into why the 2-neuron hidden layers are harder to train. Our 6-layer network has one bias in each hidden layers. Therefore if one of these biases moves points out of top-right quadrant, then applying the ReLU operator will eliminate these points to zero. After that, no matter how later layers transform the data, the values will remain zero. We can make a neural network easier to train by making the network "fatter" - i.e. adding more neurons in hidden layers - or we can add more hidden layers, or a combination of the two methods. Throughout this course we will explore how to determine the best network architecture for a given problem, stay tuned.
-->
이로부터 왜 2개 뉴런으로 이루어진 은닉층이 학습하기 어려운 지에 대해 감이 올 것이다. 6-계층 신경망에는 각 은닉층마다 하나의 편향<sup>Bias</sup>이 있다. 따라서, 이러한 편향 중 하나가 점을 우상단 사분면 밖으로 이동시켰을 때 ReLU 연산을 하면 그 점은 0으로 제거된다. 그 후, 나중에 각 층들이 어떻게 데이터를 변환하든지, 값은 0으로 유지된다. 은닉층에 뉴런을 추가하는 등 네트워크를 "더 무겁게" 만들어 신경망을 보다 더 쉽게 학습시킬 수 있다. 또는 은닉층을 더 추가하거나 앞의 두 방법을 조합하는 방법도 있다. 이러한 과정을 통해 우리는 주어진 문제에 가장 적합한 신경망 아키텍처를 결정하는 방법을 계속 탐구할 것이다.  
   
   
## [매개변수 변환](https://www.youtube.com/watch?v=FW5gFiJb-ig&t=477s)

<!--
General parameter transformation means that our parameter vector $w$ is the output of a function. By this transformation, we can map original parameter space into another space. In Figure 5, $w$ is actually the output of $H$ with the parameter $u$. $G(x,w)$ is a network and $C(y,\bar y)$ is a cost function. The backpropagation formula is also adapted as follows,
-->
일반적으로 매개변수 변환은 매개변수 벡터 $w$가 함수의 출력임을 의미한다. 이 변환을 통해 원래 매개변수 공간을 다른 공간에 맵핑할 수 있다. 그림 5를 보면, $w$는 실제 매개변수 $u$를 가진 $H$의 출력이다. $G(x,w)$는 신경망이며 $C(y,\bar y)$는 비용 함수<sup>Cost Function</sup>이다. 역전파<sup>Backpropagation</sup> 공식도 다음과 같이 조정된다.  

$$u \leftarrow u - \eta\frac{\partial H}{\partial u}^\top\frac{\partial C}{\partial w}^\top$$

$$w \leftarrow w - \eta\frac{\partial H}{\partial u}\frac{\partial H}{\partial u}^\top\frac{\partial C}{\partial w}^\top$$

<!--
These formulas are applied in a matrix form. Note that the dimensions of the terms should be consistent. The dimension of $u$,$w$,$\frac{\partial H}{\partial u}^\top$,$\frac{\partial C}{\partial w}^\top$ are $[N_u \times 1]$,$[N_w \times 1]$,$[N_u \times N_w]$,$[N_w \times 1]$, respectively. Therefore, the dimension of our backpropagation formula is consistent.
-->
이 공식들은 행렬 형태로 적용된다. Terms의 차원은 일정해야 한다는 것을 꼭 명심해야 한다. $u$,$w$,$\frac{\partial H}{\partial u}^\top$,$\frac{\partial C}{\partial w}^\top$의 차원은 각각 $[N_u \times 1]$,$[N_w \times 1]$,$[N_u \times N_w]$,$[N_w \times 1]$이다. 따라서 역전파 공식의 차원은 consistent 하다.  
  
  
<center><img src="{{site.baseurl}}/images/week03/03-1/PT.png" alt="Network" style="zoom:35%;" /><br>
그림. 5 일반적인 매개변수 변환 형태</center><br>


### 간단한 매개변수 변환법: 가중치 공유

<!--
A Weight Sharing Transformation means $H(u)$ just replicates one component of $u$ into multiple components of $w$. $H(u)$ is like a **Y** branch to copy $u_1$ to $w_1$, $w_2$. This can be expressed as,
-->
가중치 공유 변환법은 $H(u)$가 $u$의 한 구성 요소를 여러 개의 $w$ 구성 요소로 복제하는 것을 의미한다. $H(u)$는 $u_1$를 $w_1$, $w_2$에 복사하는 **Y** branch와 같다. 이는 다음과 같이 표현 될 수 있다.

$$
w_1 = w_2 = u_1, w_3 = w_4 = u_2
$$

We force shared parameters to be equal, so the gradient w.r.t. to shared parameters will be summed in the backprop. For example the gradient of the cost function $C(y, \bar y)$ with respect to $u_1$ will be the sum of the gradient of the cost function $C(y, \bar y)$ with respect to $w_1$ and the gradient of the cost function $C(y, \bar y)$ with respect to $w_2$.



### Hypernetwork

A hypernetwork is a network where the weights of one network is the output of another network. Figure 6 shows the computation graph of a "hypernetwork". Here the function $H$ is a network with parameter vector $u$ and input $x$. As a result, the weights of $G(x,w)$ are dynamically configured by the network $H(x,u)$. Although this is an old idea, it remains very powerful.

<br><center><img src="{{site.baseurl}}/images/week03/03-1/HyperNetwork.png" alt="Network" style="zoom:35%;" /><br>
Fig. 6 "Hypernetwork"</center><br>

### Motif detection in sequential data

Weight sharing transformation can be applied to motif detection. Motif detection means to find some motifs in sequential data like keywords in speech or text. One way to achieve this, as shown in Figure 7, is to use a sliding window on data, which moves the weight-sharing function to detect a particular motif (i.e. a particular sound in speech signal), and the outputs (i.e. a score) goes into a maximum function.

<br><center><img src="{{site.baseurl}}/images/week03/03-1/Motif.png" alt="Network" style="zoom:30%;" /><br>
Fig. 7 Motif Detection for Sequential Data</center><br>

In this example we have 5 of those functions. As a result of this solution, we sum up five gradients and backpropagate the error to update the parameter $w$. When implementing this in PyTorch, we want to prevent the implicit accumulation of these gradients, so we need to use `zero_grad()` to initialize the gradient.

### Motif detection in images

The other useful application is motif detection in images. We usually swipe our "templates" over images to detect the shapes independent of position and distortion of the shapes. A simple example is to distinguish between "C" and "D", as Figure 8 shows. The difference between "C" and "D" is that "C" has two endpoints and "D" has two corners. So we can design "endpoint templates" and "corner templates". If the shape is similar to the "templates", it will have thresholded outputs. Then we can distinguish letters from these outputs by summing them up. In Figure 8, the network detects two endpoints and zero corners, so it activates "C".

<br><center><img src="{{site.baseurl}}/images/week03/03-1/MotifImage.png" alt="Network" style="zoom:35%;" /><br>
Fig. 8 Motif Detection for Images</center><br>

It is also important that our "template matching" should be shift-invariant - when we shift the input, the output (i.e. the letter detected) shouldn't change. This can be solved with weight sharing transformation. As Figure 9 shows, when we change the location of "D", we can still detect the corner motifs even though they are shifted. When we sum up the motifs, it will activate the "D" detection.

<br><center><img src="{{site.baseurl}}/images/week03/03-1/ShiftInvariance.png" alt="Network" style="zoom:35%;" /><br>
Fig. 9 Shift Invariance</center><br>

This hand-crafted method of using local detectors and summation to for digit-recognition was used for many years. But it presents us with the following problem: How can we design these "templates" automatically? Can we use neural networks to learn these "templates"? Next, We will introduce the concept of **convolutions** , that is, the operation we use to match images with "templates".