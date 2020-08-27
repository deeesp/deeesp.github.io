---

title: "[DL] Activation Functions (활성화 함수)"
excerpt: "다양한 활성화 함수들에 대한 고찰"
categories:

- Deep Learning

tags:

- 딥러닝
- Deep Learning
- 활성화 함수

last_modified_at: 2020-08-27-23:00:00

toc: true
toc_sticky: true

---

# DS-GA 1008 SPRING 2020 11-1

> Yann LeCun 교수님의 NYU Deep Learning 강의 11-1 번역

## [활성화 함수](https://www.youtube.com/watch?v=bj1fh3BvqSU&t=15s)

<!-- In today's lecture, we will review some important activation functions and their implementations in PyTorch. They came from various papers claiming these functions work better for specific problems.-->

몇 가지 중요한 활성화 함수와 파이토치에서의 구현에 대해 복습할 것이다. 다양한 논문에서 어떤 특정 문제들에 대해서 여기서 다룰 활성화 함수들이 효과적이라고 동작한다고 주장했다.

### ReLU - `nn.ReLU()`

$$\text{ReLU}(x) = (x)^{+} = \max(0,x)$$

<center>
<img src="/images/week11/11-1/ReLU.png" height="400px" /><br>
<b>그림. 1</b>: ReLU 함수
</center>

### RReLU - `nn.RReLU()`

<!-- There are variations in ReLU. The Random ReLU (RReLU) is defined as follows. -->
ReLU 함수에서 변형을 준 것이다. Random ReLU(RReLU)는 다음과 같이 정의한다.

$$\text{RReLU}(x) = \begin{cases}
x, & \text{if $x \geq 0$}\\
ax, & \text{otherwise}
\end{cases}$$

<center>
<img src="/images/week11/11-1/RRelU.png" width="700" /><br>
<b>그림. 2</b>: ReLU, Leaky ReLU/PReLU, RReLU
</center>

<!--Note that for RReLU, $a$ is a random variable that keeps samplings in a given range during training, and remains fixed during testing. For PReLU , $a$ is also learned. For Leaky ReLU, $a$ is fixed.-->

RReLU에서 주의해야 할 점은, $a$는 훈련 중에 주어진 범위 내에서 추출되는 확률 변수이며, 테스트할 때에는 고정 값이 된다. PReLU에서도 또한 $a$는 학습되지만, Leaky ReLU에서는 $a$는 고정 값이다.

### LeakyReLU - `nn.LeakyReLU()`

$$\text{LeakyReLU}(x) = \begin{cases}
x, & \text{if $x \geq 0$}\\
a_\text{negative slope}x, & \text{otherwise}
\end{cases}$$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/LeakyReLU.png" height="400px" /><br>
<b>그림. 3</b>: LeakyReLU
</center>

<!-- Here $a$ is a fixed parameter. The bottom part of the equation prevents the problem of dying ReLU which refers to the problem when ReLU neurons become inactive and only output 0 for any input. Therefore, its gradient is 0. By using a negative slope, it allows the network to propagate back and learn something useful. -->

여기서 $a$는 고정된 매개 변수이다. 수식의 아래부분은 ReLU 뉴런이 비활성화 되는 것 즉, 어떤 입력에 대해서도 출력이 0이 되어 버리는 "Dying ReLU" 문제를 막아준다. 그러므로, 경사가 0이 된다. `음의 입력에 대한 작은 값의 기울기<sup>Negative slope</sup>`를 사용함으로써, 신경망이 역전파되어 잘 학습할 수 있도록 한다.

<!-- LeakyReLU is necessary for skinny network, which is almost impossible to get gradients flowing back with vanilla ReLU. With LeakyReLU, the network can still have gradients even we are in the region where everything is zero out.-->
LeakyReLU is necessary for skinny network, which is almost impossible to get gradients flowing back with vanilla ReLU.

LeakyReLU는 바닐라 ReLU를 쓴다면 경사를 역전파하는 것이 거의 불가능한 `skinny network`에서 필요하다. 즉, 모든 값이 0이 되어 훈련이 힘든 영역에서도 LeakyReLU를 사용하면 `skinny network`는 경사를 가질 수 있다.

### PReLU - `nn.PReLU()`

$$\text{PReLU}(x) = \begin{cases}
x, & \text{if $x \geq 0$}\\
ax, & \text{otherwise}
\end{cases}$$

<!-- Here $a$ is a learnable parameter.-->

여기서 $a$는 학습 가능한 매개변수이다.

<center>
<img src="{{site.baseurl}}/images/week11/11-1/PReLU.png" height="400px" /><br>
<b>그림. 4</b>: PReLU
</center>

<!--The above activation functions (*i.e.* ReLU, LeakyReLU, PReLU) are scale-invariant.-->

위의 활성화 함수들(ReLU, LeakyReLU, PReLU)은 스케일 불변<sup>Scale-invariant</sup>하다.

### Softplus - `Softplus()`

$$\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))$$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/Softplus.png" height="400px" /><br>
<b>그림. 5</b>: Softplus
</center>

<!--Softplus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.

The function will become more like ReLU, if the $\beta$ gets larger and larger. -->

Softplus는 ReLU 함수를 부드럽게 근사한 것(미분 가능)으로, 출력을 항상 양수로 제한하는 데에 사용할 수 있다. $\beta$가 커질수록 ReLU와 비슷해진다.

### ELU - `nn.ELU()`

$$\text{ELU}(x) = \max(0, x) + \min(0, \alpha * (\exp(x) - 1) $$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/ELU.png" height="400px" /><br>
<b>그림. 6</b>: ELU
</center>

<!--Unlike ReLU, it can go below 0 which allows the system to have average output to be zero. Therefore, the model may converge faster. And its variations (CELU, SELU) are just different parametrizations.-->

ReLU와는 달리, 0 이하로 내려가면 시스템의 평균 출력이 0이 될 수 있어, 모델이 더 빠르게 수렴할 것이다. CELU와 SELU는 ELU를 변형한 것으로, 단지 다른 방식으로 매개변수화한 것이다.

### CELU - `nn.CELU()`

$$\text{CELU}(x) = \max(0, x) + \min(0, \alpha * (\exp(x/\alpha) - 1) $$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/CELU.png" height="400px" /><br>
<b>그림. 7</b>: CELU
</center>

### SELU - `nn.SELU()`

$$\text{SELU}(x) = \text{scale} * (\max(0, x) + \min(0, \alpha * (\exp(x) - 1)) $$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/SELU.png" height="400px" /><br>
<b>그림. 8</b>: SELU
</center>

### GELU - `nn.GELU()`

$$\text{GELU(x)} = x * \Phi(x) $$

<!--where $\Phi(x)$ is the Cumulative Distribution Function for Gaussian Distribution.-->

여기서 $\Phi(x)$는 가우시안 분포의 누적확률분포함수(CDF)<sup>Cumulative Distribution Function</sup>이다.

<center>
<img src="{{site.baseurl}}/images/week11/11-1/GELU.png" height="400px" /><br>
<b>그림. 9</b>: GELU
</center>

### ReLU6 - `nn.ReLU6()`

$$\text{ReLU6}(x) = \min(\max(0,x),6) $$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/ReLU6.png" height="400px" /><br>
<b>그림. 10</b>: ReLU6
</center>

<!--This is ReLU saturating at 6. But there is no particular reason why picking 6 as saturation, so we can do better by using Sigmoid function below.-->

ReLU6는 6에서 포화상태가 되는<sup>Saturated</sup> ReLU이다. 그러나 6을 포화상태로 하는 데에는 특별한 이유가 없기 때문에, 아래 Sigmoid 함수를 사용하는 것이 더 좋을 수 있다.

### Sigmoid - `nn.Sigmoid()`

$$\text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)} $$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/Sigmoid.png" height="400px" /><br>
<b>그림. 11</b>: Sigmoid
</center>

<!--If we stack sigmoids in many layers, it may be inefficient for the system to learn and requires careful initialization. This is because if the input is very large or small, the gradient of the sigmoid function is close to 0. In this case, there is no gradient flowing back to update the parameters, known as saturating gradient problem. Therefore, for deep neural networks, a single kink function (such as ReLU) is preferred.-->

여러 계층에 Sigmoid를 쌓으면, 학습하는데 비효율적일 수 있고 초기화에 주의해야 한다. 이는 입력이 매우 크거나 작으면 Sigmoid의 기울기가 0에 가까워지기 때문이다. 이 경우, 매개변수를 업데이트 하기 위해 역전파 되는 경사가 없게 되는데, 이를 경사포화문제<sup>Saturating gradient problem</sup>라고 한다. 그러므로, 심층 신경망에서는 ReLU와 같은 단일 Kink 함수<sup>A single kink function</sup>(미분 불가능한 점이 하나 존재하는 함수)가 선호된다.

### Tanh - `nn.Tanh()`

$$\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}
$$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/Tanh.png" height="400px" /><br>
<b>그림. 12</b>: Tanh
</center>

<!--Tanh is basically identical to Sigmoid except it is centred, ranging from -1 to 1. The output of the function will have roughly zero mean. Therefore, the model will converge faster. Note that convergence is usually faster if the average of each input variable is close to zero. One example is Batch Normalization.-->

Tanh는 -1에서 1까지의 범위에 집중되어 있다는 점을 제외하면 기본적으로 Sigmoid와 동일하다. 함수 출력의 평균은 거의 0에 가까우므로, 모델이 빨리 수렴하게 된다. 일반적으로 각 입력 변수의 평균이 0에 가까울수록 수렴속도가 빠르다는 것에 유의해야 한다. 배치 정규화<sup>Batch Normalization</sup>가 그 예시이다.

### Softsign - `nn.Softsign()`

$$\text{SoftSign}(x) = \frac{x}{1 + |x|}
$$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/Softsign.png" height="400px" /><br>
<b>그림. 13</b>: Softsign
</center>

<!--It is similar to the Sigmoid function but gets to the asymptote slowly and alleviate the gradient vanishing problem (to some extent).-->

Sigmoid 함수와 비슷하지만, 점근선<sup>Asymptote</sup>에 천천히 도달하여 경사소실문제<sup>Gradient Vanishing Problem</sup>를 어느정도 완화해준다.

### Hardtanh - `nn.Hardtanh()`

$$\text{HardTanh}(x) = \begin{cases}
1, & \text{if $x > 1$}\\
-1, & \text{if $x < -1$}\\
x, & \text{otherwise}
\end{cases}$$

<!--The range of the linear region [-1, 1] can be adjusted using `min_val` and `max_val`.-->

선형 영역[-1, 1]의 범위는 `min_val` 과 `max_val`를 이용하여 조정할 수 있다.

<center>
<img src="{{site.baseurl}}/images/week11/11-1/Hardtanh.png" height="400px" /><br>
<b>그림. 14</b>: Hardtanh
</center>

<!--It works surprisingly well especially when weights are kept within the small value range.-->

Hardtanh는 놀랍게도 가중치가 작은 값 범위 내에 있을 때 놀랍게도 잘 작동한다.

### Threshold - `nn.Threshold()`

$$y = \begin{cases}
x, & \text{if $x > \text{threshold}$}\\
v, & \text{otherwise}
\end{cases}$$

<!--It is rarely used because we cannot propagate the gradient back. And it is also the reason preventing people from using back-propagation in 60s and 70s when they were using binary neurons.-->

Threshold는 경사를 역전파할 수 없기 때문에 거의 사용되지 않는다. 60~70년대에 이진뉴런을 사용할 때, 역전파를 사용하지 못했던 이유이기도 하다.

### Tanhshrink - `nn.Tanhshrink()`

$$\text{Tanhshrink}(x) = x - \tanh(x) $$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/Tanhshrink.png" height="400px" /><br>
<b>그림. 15</b>: Tanhshrink
</center>

<!--It is rarely used except for sparse coding to compute the value of the latent variable.-->

Thanshrink는 잠재변수 값을 계산하기 위한 희소 코딩<sup>Sparse coding</sup> 외에는 거의 사용되지 않는다.

### Softshrink - `nn.Softshrink()`

$$\text{SoftShrinkage}(x) = \begin{cases}
x - \lambda, & \text{if $x > \lambda$}\\
x + \lambda, & \text{if $x < -\lambda$}\\
0, & \text{otherwise}
\end{cases}$$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/Softshrink.png" height="400px" /><br>
<b>그림. 16</b>: Softshrink
</center>

<!--This basically shrinks the variable by a constant towards 0, and forces to 0 if the variable is close to 0. You can think of it as a step of gradient for the $\ell_1$ criteria. It is also one of the step of the Iterative Shrinkage-Thresholding Algorithm (ISTA). But it is not commonly used in standard neural network as activations.-->

Softshrink는 기본적으로 변수를 0을 향해 상수로 수축시키고, 변수가 0에 가까워지면 출력은 0이 된다. $\ell_1$ 기준에 대한 경사라고 생각하면 된다. Softshrink를 Iterative Shrinkage-Thresholding Algorithm (ISTA)의 특정 단계에서 쓰이기도 하다. 그러나 일반적인 신경망에서 활성화로는 잘 사용되지 않는다.

### Hardshrink - `nn.Hardshrink()`

$$\text{HardShrinkage}(x) = \begin{cases}
x, & \text{if $x > \lambda$}\\
x, & \text{if $x < -\lambda$}\\
0, & \text{otherwise}
\end{cases}$$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/Hardshrink.png" height="400px" /><br>
<b>그림. 17</b>: Hardshrink
</center>

<!-- It is rarely used except for sparse coding. -->

Hardshrink 또한 희소 코딩<sup>Sparse coding</sup>을 제외하고는 거의 사용되지 않는다.

### LogSigmoid - `nn.LogSigmoid()`

$$\text{LogSigmoid}(x) = \log\left(\frac{1}{1 + \exp(-x)}\right) $$

<center>
<img src="{{site.baseurl}}/images/week11/11-1/LogSigmoid.png" height="400px" /><br>
<b>그림. 18</b>: LogSigmoid
</center>

<!--It is mostly used in the loss function but not common for activations.-->

LogSigmoid는 주로 활성화함수 보다는 손실함수에 사용된다.

### Softmin - `nn.Softmin()`

$$\text{Softmin}(x_i) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)} $$

<!--It turns numbers into a probability distribution.-->

Softmin은 숫자를 확률분포로 변환한다.

### Soft(arg)max - `nn.Softmax()`

$$\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} $$

### LogSoft(arg)max - `nn.LogSoftmax()`

$$\text{LogSoftmax}(x_i) = \log\left(\frac{\exp(x_i)}{\sum_j \exp(x_j)}\right) $$

<!--It is mostly used in the loss function but not common for activations.-->

LogSoft(arg)max 또한 활성화 함수에는 잘 안쓰이고 주로 손실함수에 사용된다.

## [Q&A 활성화 함수](https://www.youtube.com/watch?v=bj1fh3BvqSU&t=861s)

### `nn.PReLU()` 관련된 질문들

<!--
- Why would we want the same value of $a$ for all channels?

- Different channels could have different $a$. You could use $a$ as a parameter of every unit. It could be shared as a feature map as well.

-->

- 왜 모든 채널에 대해 같은 $a$ 값이 필요한가요?

    > 각 채널들은 다른 $a$ 값을 가질 수 있다. $a$를 모든 유닛의 매개변수로 사용할 수 있다. 또한, 특징지도<sup>Feature map</sup>로 공유될 수 있다.

<!--

- Do we learn $a$? Is learning $a$ advantageous?

    > You can learn $a$ or fix it.
    The reason for fixing is to ensure that nonlinearity gives you a non-zero gradient even if it's in a negative region.
    Making $a$ learnable allows the system to turn nonlinearity into either linear mapping or full rectification. It could be useful for some applications like implementing an edge detector regardless of the edge polarity.

-->

- $a$를 학습하나요? $a$를 학습함으로써 얻는 이점이 있나요?

    > $a$를 학습할 수도 있고 고정시킬 수도 있다. 고정시키는 이유는 음의 영역에서도 0이 아닌 경사를 갖는 비선형성을 보장하기 때문이다. $a$를 학습가능하도록 만듦으로써 시스템을 비선형성에서 선형 매핑 및 `완전정류`<sup>Full rectification</sup>로 변환할 수 있다. 이는 윤곽선<sup>Edge</sup>의 극성에 무관한 윤곽선 검출을 구현과 같은 응용에 사용될 수 있다.

<!--

- How complex do you want your non-linearity to be?

    > Theoretically, we can parametrise an entire nonlinear function in very complicated way, such as with spring parameters, Chebyshev polynomial, etc. Parametrising could be a part of learning process.

-->

- 얼마나 비선형성을 복잡하게 만들 수 있나요?

    > 이론적으로, `스프링 매개변수<sup>Spring parameter</sup>`와 체비셰프 다항식 등과 같이 전체 비선형 함수를 아주 복잡한 방법으로 매개변수화 시킬 수 있다. 매개변수화 시키는 것은 일종의 학습 방법이다.

<!--

- What is an advantage of parametrising over having more units in your system?

    > It really depends on what you want to do. For example, when doing regression in a low dimensional space, parametrisation might help. However, if your task is in under a high dimensional space such as image recognition, just "a" nonlinearity is necessary and monotonic nonlinearity will work better.
    In short, you can parametrize any functions you want but it doesn't bring a huge advantage.

-->

- 시스템에서 유닛을 많이 가지고 있는 것보다 매개변수화 하는 것의 이점이 무엇인가요?

    > 무엇을 하고 싶은지에 달려있다. 예를 들어, 저차원 공간에서 회귀 분석을 수행할 때에는, 매개변수화 하는 것이 도움이 될 수 있다. 그러나 이미지 인식과 같이 고차원 공간을 다루는 경우, "a" 비선형성이 필요할 것이고 단조 비선형성<sup>Monotonic nonlinearity</sup>이 더 효과적일 것이다. 간단히 말해, 어떠한 함수도 매개변수화할 수 있지만, 큰 이점을 가져오지는 않을 수 있다.

### Kink related questions

<!--

- One kink versus double kink

    > Double kink is a built-in scale in it. This means that if the input layer is multiplied by two (or the signal amplitude is multiplied by two), then outputs will be completely different. The signal will be more in nonlinearity, thus you will get a completely different behaviour of the output. Whereas, if you have a function with only one kink, if you multiply the input by two, then your output will be also multiplied by two.

-->

- 단일 Kink vs 이중 Kink

> `이중 kink가 기본 스케일이다.` 이말인 즉슨, 입력 층에 2가 곱해지면(또는 신호의 진폭에 2를 곱하면), 출력은 완전히 달라질 것이다. 이 신호는 비선형에 더욱 가까워질 것이고, 완전히 다른 특성을 갖는 출력을 얻을 것이다. 반면에 단일 kink 함수라면, 입력에 2를 곱해도 출력 또한 2를 곱한 값이 될 것이다.

<!--

- Differences between a nonlinear activation having kinks and a smooth nonlinear activation. Why/when one of them is preferred?

    > It is a matter of scale equivariance. If kink is hard, you multiply the input by two and the output is multiplied by two. If you have a smooth transition, for example, if you multiply the input by 100, the output looks like you have a hard kink because the smooth part is shrunk by a factor of 100. If you divide the input by 100, the kink becomes a very smooth convex function. Thus, by changing the scale of the input, you change the behaviour of the activation unit.

    > Sometimes this could be a problem. For example, when you train a multi-layer neural net and you have two layers that are one after the other. You do not have a good control for how big the weights of one layer is relative to the other layer's weights. If you have nonlinearity that cares about scales, your network doesn't have a choice of what size of weight matrix can be used in the first layer because this will completely change the behaviour.

    > One way to fix this problem is setting a hard scale on the weights of every layer so you can normalise the weights of layers, such as batch normalisation. Thus, the variance that goes into a unit becomes always constant. If you fix the scale, then the system doesn't have any way of choosing which part of the nonlinearity will be using in two kink function systems. This could be a problem if this 'fixed' part becomes too 'linear'. For example, Sigmoid becomes almost linear near zero, and thus batch normalisation outputs (close to 0) could not be activated 'non-linearly'.It is not entirely clear why deep networks work better with single kink functions. It's probably due to the scale equivariance property.

-->

- 꺾인 비선형 활성화와 부드러운 비선형 활성화의 차이점, 언제 어떤 것이 왜 선호되나요?

> 이는 `Scale equivariance`와 관련된 문제이다. `kink가 hard하면,` 입력에 2를 곱했을 때 출력에도 2가 곱해진다. 부드러운 활성화의 경우로 예를 들자면, 입력에 100을 곱한다면 출력은 hard kink처럼 보이게 된다. 이는 부드러운 부분이 100만큼 줄어들기 떄문이다. 반대로 입력을 100으로 나누면, kink는 아주 매끄러운 볼록함수<sup>Convex function</sup>가 된다. 따라서 입력의 스케일을 변경하여 활성화 유닛의 특성을 꿔야 한다.

> 하지만 이 방법은 종종 문제가 될 때가 있다. 예를 들어, 다층 신경망을 학습시킬 때 연달아 있는 두 개의 층의 경우가 그렇다. 한 층의 가중치들이 다른 층의 가중치들에 얼마나 크게 상관이 있는지를 제어할 수 없다. 스케일에 영향이 있는 비선형은 특성을 완전히 바꿔 버리기 때문에, 신경망의 첫 번째 층에서 어떤 크기의 가중치 행렬을 사용할 지 정할 수 없다.

> 이 문제를 해결할 수 있는 방법 중 하나는 배치 정규화<sup>Batch normalization</sup>와 같이 각 층의 가중치를에 `hard scale을 설정`해서 정규화 할 수 있다. 따라서 한 유닛으로 들어가는 분산이 항상 상수가 된다. `스케일을 고정하게 되면, 시스템은 이중 kink 함수 시스템에서 비선형성의 어떤 부분을 사용할지 선택할 수 있는 방법이 없다`. '고정된' 부분이 너무 선형에 가까워지면 문제가 된다. 예를 들어, Sigmoid는 0 부근에서 거의 선형이 되어서 (0에 가까운) 배치 정규화된 출력은 '비선형성'을 활성화 시킬 수 없다. 심층 신경망이 왜 단일 kink 함수에서 잘 작동하는지에 대한 이유는 명확하지는 않다. 아마도 scale equivariance 특성 때문인 것 같다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a1d4de9-b572-4835-ab68-0d75851c6dcf/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a1d4de9-b572-4835-ab68-0d75851c6dcf/Untitled.png)

출처: [http://parker.ad.siu.edu/Olive/ch9.pdf](http://parker.ad.siu.edu/Olive/ch9.pdf)

<!--

### Temperature coefficient in a soft(arg)max function

-->

### Soft(arg)max 함수에서의 온도계수<sup>Temperature coefficient</sup>

<!--

- When do we use the temperature coefficient and why do we use it?

    > To some extent, the temperature is redundant with incoming weights. If you have weighted sums coming into your softmax, the $\beta$ parameter is redundant with the size of weights.

    > Temperature controls how hard the distribution of output will be. When $\beta$ is very large, it becomes very close to either one or zero. When $\beta$ is small, it is softer. When the limit of $\beta$ equals to zero, it is like an average. When $\beta$ goes infinity, it behaves like argmax. It's no longer its soft version. Thus, if you have some sort of normalisation before the softmax then, tuning this parameter allows you to control the hardness.
    Sometimes, you can start with a small $\beta$ so that you can have well-behaved gradient descents and then, as running proceeds, if you want a harder decision in your attention mechanism, you increase $\beta$. Thus, you can sharpen the decisions. This trick is called as annealing. It can be useful for a mixture of experts like a self attention mechanism.

-->

- 언제 온도계수를 사용하고, 왜 사용하나요?

> 어느 정도까지는, 온도는 입력 가중치에 `redundant`하다. 가중합이 Softmax로 들어가면, $\beta$ 매개변수는 가중치의 크기와 `redundant`하다.

> 온도는 출력분포의 `hard`한 정도를 제어한다. $\beta$가 아주 클 때는 출력이 1이나 0에 아주 가깝게 된다. $\beta$가 아주 작으면 `soft`해진다. $\beta$가 0에 가깝게 한정되면 평균에 가까워진다. $\beta$가 무한대로 발산하게 되면, argmax와 같은 특성을 가져 더이상 `soft`한 버전이 아니게 된다. 그러므로 softmax에 넣어주기 전에 정규화<sup>Normalization</sup>를 거치면 매개변수 조정을 통해 `hardness`를 조절할 수 있다.

> 보통 작은 $\beta$에서 시작하면, 경사하강법이 적절하게 동작하도록 할 수 있고, 어텐션 메커니즘<sup>Attention mechanism</sup>과 같은  `harder decision`을 원한다면 $\beta$를 증가시키면 된다. 따라서, 결정을 `sharpen` 하게 할 수 있다. 이러한 방법을 담금질<sup>Annealing</sup>이라고 한다. Self-attention mechanism과 같은 experts의 mixture에 유용하게 사용할 수 있다.

returning a probability distribution

```python
def softmax_with_temperature(z, beta) : 
    z = np.array(z)
    z = z * beta
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y
```

출처:

[https://3months.tistory.com/491](https://3months.tistory.com/491)

[Deep Play]

## 그렇다면, 왜 상수 'e'를 사용하는가?

1. 아래와같이 미분이 아주 깔끔해서 입니다.

    ![https://blog.kakaocdn.net/dn/Mhboo/btqy7Kjz1L6/FTTWidzaqT5WwVQ74Vz450/img.png](https://blog.kakaocdn.net/dn/Mhboo/btqy7Kjz1L6/FTTWidzaqT5WwVQ74Vz450/img.png)

2. 'soft'max의 관점에서, 입력값에 대한 결과값을 보면 'hard'하게 잘려있기 보다는, 'soft'하게 나눠지는 것을 볼 수 있습니다. 그래서 max값만 출력되는, one-hot vector의 형태로 출력되는 argmax와 달리, softmax는 max에 근접한 값들도 출력이되며, 이를 이용해 손실을 계산하는것이 유익합니다. (참고로 hardmax는 미분이 불가능하며, soft는 어디에서든지 미분이 가능합니다.)
    - argmax ([1, 3, 0, 2]) = [0, 1, 0, 0]
    - soft arg max([1, 3, 0, 2]) ≈ [0.087 0.644 0.032 0.24]
3. soft'max'의 관점에서, max이려면 차이가 얼마 안나도 가장 큰 값이 **확실하게 구별될 정도로** 크게 나타나야하기에, e를 사용하면, 지수승으로 표현이 가능해서 값의 비교가 명확해짐. 다른말로 하면, 엔트로피를 최대화하는 손실함수가 softmax입니다.
