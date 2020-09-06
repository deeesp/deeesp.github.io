---

title: "[DL] Loss Functions (손실함수)"
excerpt: "정의와 예시"
categories:

- Machine Learning

tags:

- 딥러닝
- Deep Learning
- 머신러닝
- Machine Learning
- Loss function
- 손실함수

last_modified_at: 2020-09-05 22:00:00

toc: true
toc_sticky: true

---


# [손실함수 <sup>Loss functions</sup>](https://www.youtube.com/watch?v=bj1fh3BvqSU&t=1990s)

<!--PyTorch also has a lot of loss functions implemented. Here we will go through some of them.-->
파이토치에는 다양한 손실함수가 구현되어 있다. 그 중 일부를 여기서 다루어 볼 것이다.

## [1] `nn.MSELoss()`

<!--This function gives the mean squared error (squared L2 norm) between each element in the input $x$ and target $y$. It is also called L2 loss.-->
이 함수는 입력 $x$와 타겟<sup>Target</sup> $y$의 원소들 사이에 평균제곱오차<sup>Mean Squared Error (MSE, Squared L2 Norm)</sup>를 계산한다.
 
<!--If we are using a minibatch of $n$ samples, then there are $n$ losses, one for each sample in the batch. We can tell the loss function to keep that loss as a vector or to reduce it.-->
만약 $n$ 샘플의 미니배치<sup>Minibatch</sup>를 사용한다면, 이 배치에서 각 샘플당 하나씩 $n$개의 loss를 가지게 된다. loss를 벡터로 가지거나 이를 줄이는 것을 손실함수라고 한다.

<!--If unreduced (*i.e.* set `reduction='none'`), the loss is-->
만약 줄어들지 않은 상태 즉,  `reduction='none'`로 설정되어 있다면, loss는 다음과 같다.

$$l(x,y) = L = \{l_1, \dots, l_N\}^\top, l_n = (x_n - y_n)^2$$

<!--where $N$ is the batch size, $x$ and $y$ are tensors of arbitrary shapes with a total of n elements each.-->
여기서 $N$은 배치의 크기이고, $x$와 $y$는 각각 총 n개 원소를 가진 임의의 모형<sup>Shape</sup>을 가진 텐서<sup>Tensor</sup>이다.


<!--The reduction options are below (note that the default value is `reduction='mean'`).-->
Reduction 옵션은 아래와 같다. 기본값이 `reduction='mean'`인 것에 주의해야 한다.

$$l(x,y) = \begin{cases}\text{mean}(L), \quad &\text{if reduction='mean'}\\
\text{sum}(L), \quad &\text{if reduction='sum'}
\end{cases}$$

<!--The sum operation still operates over all the elements, and divides by $n$.

The division by $n$ can be avoided if one sets ``reduction = 'sum'``.-->

모든 원소에 대해 Sum 연산을 하고, $n$ 으로 나눈다.


## [2] `nn.L1Loss()`

<!--This measures the mean absolute error (MAE) between each element in the input $x$ and target $y$ (or the actual output and desired output).

If unreduced (*i.e.* set `reduction='none'`), the loss is-->

L1 loss는 입력 $x$와 타겟<sup>Target</sup> $y$의 원소들(또는 실제 출력과 원하는 출력) 사이에 평균절대오차<sup> mean absolute error (MAE)</sup>를 계산한다.

만약 줄어들지 않은 상태 즉,  `reduction='none'`로 설정되어 있다면, loss는 다음과 같다.

$$l(x,y) = L = \{l_1, \dots, l_N\}^\top, l_n = \vert x_n - y_n\vert$$

<!--, where $N$ is the batch size, $x$ and $y$ are tensors of arbitrary shapes with a total of n elements each.-->
여기서 $N$은 배치의 크기이고, $x$와 $y$는 각각 총 n개 원소를 가진 임의의 모형<sup>Shape</sup>을 가진 텐서<sup>Tensor</sup>이다.

<!--It also has `reduction` option of `'mean'` and `'sum'` similar to what `nn.MSELoss()` have.-->

`nn.MSELoss()` 에서와 비슷하게 `'mean'` and `'sum'`에 `reduction` 옵션이 있다.

<!--**Use Case:** L1 loss is more robust against outliers and noise compared to L2 loss. In L2, the errors of those outlier/noisy points are squared, so the cost function gets very sensitive to outliers.

**Problem:** The L1 loss is not differentiable at the bottom (0). We need to be careful when handling its gradients (namely Softshrink). This motivates the following SmoothL1Loss.-->

**사용 사례:** L1 loss는 L2 loss에 비하여 이상치와 잡음 <sup>Outliers and Noise</sup>에 상대적으로 견고<sup>Robust</sup>하다. L2는 이상치와 잡음이 있는 점들에 대해 제곱을 하여, 비용함수가 이상치에 대해 아주 민감하게 된다.

**문제점:**  L1 loss는 0의 값을 갖는 바닥 지점<sup>Bottom</sup>에서 미분이 불가능하기 때문에 경사를 다룰 때 주의를 기울여야 한다. (Softshrink라고 불린다.) 여기서 영감을 받은 것이 앞으로 나올 SmoothL1Loss이다.


## [3] `nn.SmoothL1Loss()`

<!--This function uses L2 loss if the absolute element-wise error falls below 1 and L1 loss otherwise.-->
이 함수는 요소별<sup>Element-wise</sup> 절대오차가 1 아래로 떨어질 때 L2 loss를 이용하고 나머지는 L1 loss를 사용한다.

$$\text{loss}(x, y) = \frac{1}{n} \sum_i z_i$$

<!--, where $z_i$ is given by-->
여기서 $z_i$는 다음과 같이 주어진다.

$$z_i = \begin{cases}0.5(x_i-y_i)^2, \quad &\text{if } |x_i - y_i| < 1\\
|x_i - y_i| - 0.5, \quad &\text{otherwise}
\end{cases}$$

<!--It also has `reduction` options.

This is advertised by Ross Girshick ([Fast R-CNN](https://arxiv.org/abs/1504.08083)). The Smooth L1 Loss is also known as the Huber Loss or  the Elastic Network when used as an objective function,.-->
이 또한 `reduction` 옵션이 있다.

SmoothL1Loss는 Ross Girshick의 [Fast R-CNN](https://arxiv.org/abs/1504.08083)에 의해서 알려졌다. Smooth L1 Loss는 Huber Loss라고도 불리며, 목적함수로 사용될 때에는 Elastic Network 라고 불린다.

<!--**Use Case:** It is less sensitive to outliers than the `MSELoss` and is smooth at the bottom. This function is often used in computer vision for protecting against outliers.

**Problem:** This function has a scale ($0.5$ in the function above).-->

**사용 사례:** `MSELoss` 보다는 이상치에 대해 덜 민감하고, 0인 지점에서 부드럽다. 이 함수는 주로 컴퓨터 비전 분야에서 이상치로부터 보호하기 위해 주로 쓰인다.

**문제점:** 이 함수는 위 함수에서 보이는 처럼 $0.5$의 스케일을 가지고 있다.


## [4] 컴퓨터 비전에서의 L1 *vs.* L2

<!--In making predictions when we have a lot of different $y$'s: -->
예측하는 과정에서, 다음과 같이 다양한 $y$를 가질 수 있다.
<!-- * If we use MSE (L2 Loss), it results in an average of all $y$, which in CV it means we will have a blurry image. -->
<!-- * If we use L1 loss, the value $y$ that minimize the L1 distance is the medium, which is not blurry, but note that medium is difficult to define in multiple dimensions. -->
* MSE(L2 Loss)를 사용한다면, 모든 $y$에 대해서 평균이 나온다. 컴퓨터 비전에서는 blurry 이미지를 얻게 된다는 뜻이다.
* L1 loss를 사용한다면, L1 거리를 최소화 하는 $y$ 값은 중간값<sup>Medium</sup>이다. Blurry하진 않지만, 다차원에서는 중간값을 정의하기 힘들다.

<!--Using L1 results in sharper image for prediction.-->
L1을 사용하면 더 날카로운 이미지를 예측해낸다.


## [5] `nn.NLLLoss()`

<!--It is the negative log likelihood loss used when training a classification problem with C classes.-->
음의 로그우도<sup>Negative Log Likelihood</sup> loss는 C개의 클래스를 가진 분류 문제를 학습시킬 때 사용된다.

<!--Note that, mathematically, the input of `NLLLoss` should be (log) likelihoods, but PyTorch doesn't enforce that. So the effect is to make the desired component as large as possible.-->
수학적으로 `NLLLoss` 의 입력은 (로그) 우도여야 하지만, 파이토치는 이를 강제하지는 않는다는 점에 유의해야 한다. 따라서 원하는 성분을 가능한 한 크게 만드는 효과가 있다.

<!--The unreduced (*i.e.* with :attr:`reduction` set to ``'none'``) loss can be described as:-->
감소되지 않은 loss 즉, :attr:`reduction` 이 ``'none'``으로 설정 되어 있으면, 다음과 같이 정의될 수 있다.

$$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\}$$
        
<!--,where $N$ is the batch size.-->
여기서 $N$은 배치 크기이다.

<!--If `reduction` is not ``'none'`` (default ``'mean'``), then-->
`reduction`이 ``'none'`` (기본 설정은 ``'mean'``)이 아니라면, 다음과 같다.

$$\ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{'sum'.}
        \end{cases}$$

<!--This loss function has an optional argument `weight` that can be passed in using a 1D Tensor assigning weight to each of the classes. This is useful when dealing with imbalanced training set.-->
이 손실함수는 각 클래스에 가중치를 할당하는 optional 인자인 `weight`을 가지고 있다. 이는 1D 텐서를 사용하여 전달할 수 있으며, 불균형한 훈련 세트를 다룰 때 유용하다. 


### 가중치와 불균형한<sup>Imbalanced</sup> 클래스

<!--Weight vector is useful if the frequency is different for each category/class. For example, the frequency of the common flu is much higher than the lung cancer. We can simply increase the weight for categories that has small number of samples.-->
가중치 벡터는 각 범주 또는 클래스의 빈도가 다를 때 유용하다. 예를 들어, 일반 독감의 빈도는 폐암보다 훨씬 높다. 간단하게 적은 샘플을 가진 범주에 가중치를 늘려줄 수 있다.

<!--However, instead of setting the weight, it's better to equalize the frequency in training so that we can exploits stochastic gradients better.-->
그러나, 가중치를 주는 대신에 확률적 경사<sup>Stochastic Gradients</sup>를 더 잘 활용할 수 있도록, 훈련 빈도를 균등하게 하는 것이 더 좋다. 

<!--To equalize the classes in training, we put samples of each class in a different buffer. Then generate each minibatch by picking the same number samples from each buffer. When the smaller buffer runs out of samples to use, we iterate through the smaller buffer from the beginning again until every sample of the larger class is used. This way gives us equal frequency for all categories by going through those circular buffers. We should never go the easy way to equalize frequency by **not** using all samples in the majority class. Don't leave data on the floor!-->
훈련 과정에서 각 클래스들을 균등하게 해주기 위해서, 다른 버퍼에 각 클래스의 샘플들을 넣어준다. 그리고 나서, 각 버퍼로부터 같은 수의 샘플을 추출하여 각 미니배치를 생성해준다. 작은 버퍼에서 사용할 샘플이 부족하면, 큰 클래스의 모든 샘플이 사용될 때까지 다시 처음부터 작은 버퍼를 반복한다. 이 방법은 순환 버퍼를 통해 모든 범주에 대해서 같은 빈도를 갖게 하는 방법이다. 빈도수를 균등하게 해주려고 주된 클래스의 모든 샘플을 사용하지 않는 것과 같은 쉬운 방법을 택해선 **절대** 안된다. 데이터를 남기지 말아야 한다.

<!--An obvious problem of the above method is that our NN model wouldn't know the relative frequency of the actual samples. To solve that, we fine-tune the system by running a few epochs at the end with the actual class frequency, so that the system adapts to the biases at the output layer to favour things that are more frequent.-->
위 방법의 명백한 문제점은 신경망 모델은 실제 샘플들의 상대적 빈도수를 모른다는 것이다. 이러한 문제를 풀기 위해서, 마지막에 실제 클래스 빈도로 약간의 epoch을 돌려 시스템을 미세조정<sup>Fine-tune</sup>한다. 그러면, 더 빈번한 것을 택하기 위해 시스템은 출력층의 편향<sup>Biases</sup>에 맞춘다. 

<!--To get an intuition of this scheme, let's go back to the medical school example: students spend just as much time on rare disease as they do on frequent diseases (or maybe even more time, since the rare diseases are often the more complex ones). They learn to adapt to the features of all of them, then correct it to know which are rare.-->
이 방법에 대한 직관을 얻기 위해, 의과 대학의 예시로 돌아가 보자. 학생들은 자주 접하는 질병에 투자하는 시간 만큼 희귀 질병에  많은 시간을 할애한다. (혹은 희귀 질병이 보통 더 복잡하기 때문에 더 많은 시간을 투자한다.) 학생들은 모든 질병의 특징에 적응하기 위해 공부를 하고나서, 어떤 질병이 희귀한지 수정한다.

## [6] `nn.CrossEntropyLoss()`

<!--This function combines `nn.LogSoftmax` and `nn.NLLLoss` in one single class. The combination of the two makes the score of the correct class as large as possible.-->
이 함수는 `nn.LogSoftmax`와  `nn.NLLLoss`가 하나의 클래스로 구성되어 있다. 이 둘의 조합은 정확한 크래스에 대해서 더 큰 점수를 부여한다.

<!--The reason why the two functions are merged here is for numerical stability of gradient computation. When the value after softmax is close to 1 or 0, the log of that can get close to 0 or $-\infty$. Slope of log close to 0 is close to $\infty$, causing the intermediate step in backpropagation to have numerical issues. When the two functions are combined, the gradients is saturated so we get a reasonable number at the end.-->
여기서 두 함수를 합친 이유는 경사를 연산함에 있어서 수치적 안정성을 위한 것이다. 소프트맥스를 거친 값이 1이나 0에 가까우면, 그 값에 로그를 취한 것은 0이나 $-\infty$에 가까워 질 수 있다. 0에 가까운 로그 값의 기울기는 $\infty$에 가까워지며, 역전파의 중간단계에서 수치적 문제를 발생시킨다. 두 함수가 합쳐지면, 경사는 포화되어 적당한 숫자를 얻게 된다.

<!--The input is expected to be unnormalised score for each class.

The loss can be described as:-->
각 클래스에 대한 입력 값은 정규화 되지 않은 점수일 것이다.

그 Loss는 다음과 같이 표현될 수 있다.

$$\text{loss}(x, c) = -\log\left(\frac{\exp(x[c])}{\sum_j \exp(x[j])}\right)
= -x[c] + \log\left(\sum_j \exp(x[j])\right)$$

<!--or in the case of the `weight` argument being specified:-->
또는 `weight` 인자가 구체화 된 경우에는 다음과 같다.

$$\text{loss}(x, c) = w[c] \left(-x[c] + \log\left(\sum_j\exp(x[j])\right)\right)$$

<!--The losses are averaged across observations for each minibatch.-->
이 loss들은 각 미니배치에 대한 관찰을 통해 평균을 낸다.

<!--A physical interpretation of the Cross Entropy Loss is related to the Kullback–Leibler divergence (KL divergence), where we are measuring the divergence between two distributions. Here, the (quasi) distributions are represented by the x vector (predictions) and the target distribution (a one-hot vector with 0 on the wrong classes and 1 on the right class).

Mathematically,-->
크로스 엔트로피<sup>Cross Entropy</sup> Loss에 대한 물리적 해석은 두 분포 사이의 차이<sup>Divergence</sup>를 측정하는 쿨백-라이블러 발산<sup>KL divergence (Kullback–Leibler divergence )</sup>와 연관이 있다.

수학적으로는, 다음과 같다.

$$H(p,q) = H(p) + \mathcal{D}_{KL} (p \mid\mid q)$$

$$H(p,q) = - \sum_i p(x_i) \log (q(x_i))$$

이는 (두 분포 사이의) 크로스 엔트로피이다.

$$H(p) = - \sum_i p(x_i) \log (p(x_i))$$ 는 엔트로피이고,

$$\mathcal{D}_{KL} (p \mid\mid q) = \sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)}$$는 KL divergence이다.


## [7] `nn.AdaptiveLogSoftmaxWithLoss()`

<!--This is an efficient softmax approximation of softmax for large number of classes (for example, millions of classes). It implements tricks to improve the speed of the computation.-->
이는 아주 많은(ex, 백만개) 클래스에 대한 소프트맥스의 Efficient softmax approximation이다. 이는 연산 속도를 향상시키기 위해서 몇가지 트릭을 이용해 구현한다.

<!-- Details of the method is described in [Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309) by Edouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou. -->
자세한 방법은 Edouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou의 [Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309)에 기술되어있다.
