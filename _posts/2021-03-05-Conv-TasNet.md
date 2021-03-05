---
title:  "[Speech] Conv-TasNet 톺아보기"
excerpt: "Time-domain single-channel speech separation, Conv-TasNet 분석, 정리"
categories:
  - Speech
  
tags:
  - Conv-TasNet
  - Speech
  - Speech separation
  - 컨브테스넷
  - 음성신호처리
  - 음성
  - 음원분리
  - 디지털신호처리
  - DSP
  - Deep Learning
  - 딥러닝

last_modified_at: 2021-03-05-14:00:00

toc: true
toc_sticky: true

---



Speech separation 분야의 역사에 한 획을 그은 [Conv-TasNet](https://ieeexplore.ieee.org/abstract/document/8707065)은 2019년도 IEEE/ACM TASLP (Transactions on Audio, speech, and language processing) 저널에 출판된 논문으로, 여전히 separation과 enhancement 분야에서 base-line이 되고 있습니다. Conv-TasNet을 읽고, 분석 및 정리해 보았습니다.


## TL;DR

<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c7ae61e5-958f-403b-85e1-84b16c282861/speech_separation_on_wsj0-2mix.jpeg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T031636Z&X-Amz-Expires=86400&X-Amz-Signature=e6d0a19fff2e5255bcfda31f10033c63cee0cf33631f611dd62045b7ebb2c957&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22speech_separation_on_wsj0-2mix.jpeg%22" height="150px" /><br>
<b>Fig. 1</b>: Speech separation SOTA performance on wsj0-2mix <br> (출처 : Papers with code)
</center>
 <br>
 
1.  Conv-TasNet은 각 speech source에 대한 mask를 time-domain에서 직접 estimation하는 speech seaparation<sup>음원 분리</sup> 모델로, 성능적으로 상당한 breakthrough를 이뤄낸 모델이다. 이후에도 이 분야에서 baseline이 되고 있고, [DPRNN](https://ieeexplore.ieee.org/abstract/document/9054266) (ICASSP 2020), [DPTNet](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2205.pdf) (Interspeech 2020), [SepFormer](https://arxiv.org/abs/2010.13154) (ICASSP 2021) 등 변형 모델로 성능향상을 보이고 있다.
	- wsj0-2mix dataset 기준 SI-SNRi 15.3dB, 기존 state-of-the-art 모델에서 4dB에 가까운 성능 향상
 <br>
2. Speech separation task에서는 기존 접근 방법처럼 mixture signal을 time-frequency representation (즉, STFT를 통한 spectrogram representation)에서 처리하면 다음과 같은 이유로 suboptimal하기 때문에, time-domain approach로 다음 문제를 해결하였다.
	- Spectrogram 계산 시 high-resolution frequency를 필요로 하기 때문에, 긴 temporal window를 이용해 STFT를 하게 되고, long latency를 야기한다.
	- 또한, signal의 phase와 magnitude가 decoupling되기 때문에, clean source들의 phase reconstruction 시 정확도에 upper bound를 야기한다.
 <br>
3. Speech waveform을 speech separation task에 optimal한 latent represenatation으로 만들어 End-to-End deep learning framework을 구축하였다.
	- Latent representation을 만들어 주는 linear autoencoder는 SI-SNR loss를 최소화 해주도록 separator와 jointly training하였고, 실제 구현 시에는 1-D convolution과 1-D transposed convolution을 이용해 구현하였다.
 <br>
4. 물론, 다음과 같은 제약이 상당히 들어가 있어 추가 연구가 필요하다.
	- Microphone이 하나인 single-channel 모델이다.
	- Clean speech 환경이다. (noisy하고 reverbrant한 환경에서는 성능 안좋음)
	- 겹치는 source의 수를 알고 있어야 한다. (Unknown number of speakers)
	- Real-world에서 회의 상황과 같은 continuous speech spearation 문제는 풀기 힘들다.


## [1] Time-domain Speech Separation
Microphone (이하 MIC)이 하나인 조건인 single-channel<sup>*</sup>에서 각기 다른 speech source를 분리하는 single-channel speech separation에 대한 문제를 먼저 정의해보자.

* Single-channel : 인간으로 비유를 하자면 한 쪽 귀로만 들어<sup>Monaural</sup> 공간 정보가 없는 조건을 말하며, MIC 개수를 언급할 때에는 channel로 표기함 (e.g., signle-channel, multi-channel etc.)


### [1]-1. Problem Statement

길이가 $T$이고, $C$개의 speech source가 섞여 있는 discrete-time waveform input mixture인 $x(t) \in \mathbb{R}^{1\times T}$ 가 주어졌다고 하자. 이 때, $C$개의 source들은 각 $s_1(t), s_2(t), ...,s_C(t) \in \mathbb{R}^{1\times T}$로 표기하며, 이 source들을 time-domain에서 직접 estimation 해내는 것을 목표로 한다.
    
$$x(t) = \sum^C_{i=1}s_i(t)$$


- 전반적으로, 다음 block diagram을 따라 separation이 진행된다.


<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b6b5c339-f2e4-44eb-8701-212894b7ac98/Screen_Shot_2021-01-12_at_4.26.19_PM.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T032604Z&X-Amz-Expires=86400&X-Amz-Signature=2bc4ac6cca186499f691d192a4020c5c7ff8c57c8f33a48f306596a82176a19b&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Screen_Shot_2021-01-12_at_4.26.19_PM.png%22" height="150px" /><br>
<b>Fig. 2</b>: Time-domain audio separation network block diagram
</center>

> ★ 기본적으로 frame단위의 mixture에 대한 latent represenatation에 각 source에 해당하는 mask들을 씌워 separation한다.


### [1]-2. Input

- $x(t) \in \mathbb{R}^{1\times T}$를 길이가 $L$인 $\hat{T}$개의 overlaaping segment $\mathbf{x}_k \in \mathbb{R}^{1\times L}$로 나누어 준다. (단, $k=1,...,\hat{T}$)
- $\hat{T}$개의 waveform segment $\mathbf{x}_k$ 들을 각각 encoder 단으로 넣어준다.

사실상 $X\in\mathbb{R}^{\hat{T}\times L}$이 한꺼번에 encoder로 들어가는 것이지만, 아래 설명은 각 segment (또는 frame) 별로 다뤄지고 있다. $L$은 frame 개수를 결정하는 아주 중요한 hyperparameter로, 뒤에 설명하겠지만 작을수록 성능이 좋아졌다. 물론 $L$이 작아지면 $\hat{T}$는 커진다.


### [1]-3. Convolutional Autoencoder

Mixture signal에 대한 STFT representation을 convolutional encoder/decoder로 대체하게 된 배경은 speech separation에 optimized된 audio representation을 만들어주기 위한 것

**Encoder**

Encoder는 waveform mixture에 대한 segment $\mathbf{x}_k \in \mathbb{R}^{1\times L}$를 speech separation에 optimal하게 길이 $N$인 latent represenation $\mathbf{w} \in \mathbb{R}^{1 \times N}$로 encoding해준다. 
-   Encoder를 matrix multiplication 형태로 써주면 다음과 같다.
    $$\mathbf{w}=\mathcal{H}(\mathbf{x}\mathbf{U})$$
    
-   $\mathbf{U}\in \mathbb{R}^{L\times N}$ : Encoder basis function 역할을 하는 $N$개의 길이가 $L$인 vector로 구성된 matrix (논문에는 $N\times L$로 잘못 나와있음)

-   $\mathcal{H}(\cdot)$ : Optional nonlinear function
	- 이전 다른 모델들은 nonlinear activation function인 ReLU (Rectified Linear Unit)을 써서 encoded represenation의 non-negativity를 보장해주었다.
    - Conv-TasNet에서는 여러 조건에서의 실험을 통해 linear encoder와 decoder에 non-negative constraint을 주는 것보다 sigmoid activation을 써주는 것이 더 좋은 성능을 낸다는 것을 밝혀냈다.

**Decoder**

Decoder를 거치면 mask가 씌워진 각 estimated source에 대한 latent representation $\mathbf{d}_i\in1\times L$를 길이 $L$인 waveform source $\hat{\mathbf{s}}_i,\ i=1,2,...,C$를 reconstruction하게 된다.

$$\hat{s}_i=\mathbf{d}_i\mathbf{V}$$

- $\mathbf{d}_i\in\mathbb{R}^{1\times L}$ : Separator에서 생성한 mask로 추정된 $i$ 번째 source에 대한 latent representation
- $\mathbf{V}\in \mathbb{R}^{N\times L}$ : Decoder basis function matrix

**Implementation**
- 실제 모델 구현에선, encoder와 decoder에 각각 convolutional layer와 transposed convolutional layer를 쓰는데, 각 segment들을 overlapping 하기 쉬워 빠르게 training할 수 있고, 모델이 더 잘 수렴한다. (PyTorch 1-D transposed convolutional layers)
- Encoder/decoder representation에 대해선 뒤에서 상세하게 다룰 예정.
    

### [1]-4. Separator part

1.  $C$개의 vector (또는 mask) $\mathbf{m}_i \in \mathbb{R}^{1 \times N}, i=1,2,...,C$를 추정해낸다.

	(단, $\sum^{C}_{i=1} \mathbf{m}_i = \mathbf{1}$)
	
	→ Mask를 추정하는 방법은 잠시 후 자세히..
    
2.  Mixture representation $\mathbf{w} \in \mathbb{R}^{1 \times N}$에 각 $\mathbf{m}_i$를 element-wise multiplication을 하게 되면, 각 source의 encoded representation $\mathbf{d}_i \in \mathbb{R}^{1 \times N}$ 이 나온다. 간단히 말해, mixture에 weighting function (mask)를 씌워 source separation을 한다.
    
    $$\mathbf{d}_i = \mathbf{w}\odot\mathbf{m}_i$$
    

## [2] Convolutional Separation Module

### [2]-1. 특징

1.  Mixture $\mathbf{x}$에서 각 Source $s_i$를 separation하기 위한 mask $\mathbf{m}_i$를 추정하는 module
2.  Temporal Convolutional Network (TCN)에서 영감을 받아, 1-D dilated convolutional block을 여러 층 쌓아 convolution으로만 구성되어 있다.
    -   Sequence modeling에 쓰이는 RNN 계열 모델은 Long-term dependency를 보는 데에 유용하게 쓰이지만, recurrent connection 때문에 parallel processing에 제한이 있어 느리다.
    -   따라서, RNN 계열 모델을 대체하여 Long-term dependency를 볼 수 있고, Parallel processing이 가능한 TCN을 사용한 것이다.
3.  Standard convolution 대신에 쓰인 depth-wise convolution은 parameter 수와 compuational cost를 줄여주었다.

### [2]-2. Temporal convolutional Network (TCN)
이 모델에서 쓰인 TCN 구조는 [WaveNet](https://arxiv.org/abs/1609.03499)에서 쓰인 dilated convolution과 residual path, skip-connection path 구조를 가져와 응용한 것이다. Dilation을 주면 큰 temporal context window를 만들어 줄 수 있어 speech signal의 long-range dependency를 잡아내는 데에 좋다.

아래 Figure 3는 WaveNet에서 쓰인 구조인데, $X=4$인 한 layer를 표현한 것이라고 볼 수 있다.

<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a869fd9f-8ace-4261-a356-7c9fa7e52661/Screen_Shot_2021-01-12_at_8.28.36_PM.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T120940Z&X-Amz-Expires=86400&X-Amz-Signature=4f894cb469295c7b26441a705dcc9b0a47ee351a581de6a7a870d49356bdd9aa&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Screen_Shot_2021-01-12_at_8.28.36_PM.png%22" height="150px" /><br>
</center>

이러한 dilated convolution을 포함한 TCN 구조의 Conv-TasNet 전체 block diagram을 보면 다음과 같다.

<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c03c8b0c-e549-4be0-b0b5-a49c0ba08ff6/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T121554Z&X-Amz-Expires=86400&X-Amz-Signature=027f968cc7a650f926a5e9ab5448096fa0ea54102f6e9136fe764e73e088f976&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" height="150px" /><br>
<b>Figure 4.</b> Conv-TasNet Block Diagram<br>
(출처 : SAPL Seminar of Ph.D candidate 변재욱)
</center>

-   각 Layer는 $X$개의 각 convolutional block들로 이루어져 있고, 각 block의 dilation factor는 $1, 2, 4, ..., 2^{X-1}$ 로 증가하는 형태를 띈다. 또한, 이 layer는 $R$ 번 반복된다.

-   TCN의 출력은 Kernel size가 1인 $1\times 1$ convolution (a.k.a Point-wise convolution)을 통과하게 되고, Non-linear activation function인 Sigmoid를 지나 $C$ 개의 Mask vector를 추정한다.


### [2]-3. 1-D convolutional block**

TCN에서 반복적으로 쓰인 1-D convolutional block을 자세히 알아보자.
-   각 block의 입력은 출력과 길이를 같게 해주기 위해, zero-padding 해준다.
-   Residual path : 각 block의 input을 다음 block으로 넘겨준다.
-   Skip-connection path : 각 block에서 나오는 output을 더해 최종 TCN의 출력으로 사용한다.
-   [MobileNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html)에서 쓰였던 depthwise separable convolution $S\text{-}conv(\cdot)$ 테크닉을 가져와 standard convolution을 대체하여 Parameter 수를 줄여주었다. 

<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a353a70f-da72-4c70-a5d6-7ee9ca29e951/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T124645Z&X-Amz-Expires=86400&X-Amz-Signature=76beaead19a9a54126e7db1683f47711e26947d5a906b81ea0738b4cf6e31322&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width="200px"/><br>
<b>Figure 5.</b> 1-D Convolutional Block
</center>

**Depthwise separable convolution**
- $S\text{-}conv(\cdot)$는 Figure 5 처럼 차례로 depthwise convolution $D\text{-}conv(\cdot)$, pointwise convolution $1\times 1\text{-}conv(\cdot)$으로 구성되어 있다. (처음에 보이는  $1\times 1\text{-}conv(\cdot)$는 Bottleneck)

	1) $D\text{-}conv(\mathbf{Y},\mathbf{K})$ 는 입력 $\mathbf{Y}$의 각 row와 상응하는 행렬 $\mathbf{K}$의 row에 대해 Convolution 연산을 한다.
        $$D\text{-}conv(\mathbf{Y},\mathbf{K}) = \text{concat}(\mathbf{y}_j\circledast \mathbf{k}_j),\ j=1,...,N$$
    
	2)  $S\text{-}conv(\mathbf{Y},\mathbf{K},\mathbf{L})$는 $D\text{-}conv(\mathbf{Y},\mathbf{K})$ 와 Convolutional kernel $L$의 Convolution으로, $1\times 1\text{-}conv(\cdot)$를 통해 Linear하게 Feature space로 변환해준다. $L \in \mathbb{R}^{G\times H\times 1}$ : Size 1의 Convolutional kernel
	    
	    $$S\text{-}conv(\mathbf{Y},\mathbf{K},\mathbf{L})=D\text{-}conv(\mathbf{Y},\mathbf{K}) \circledast \mathbf{L}\\ $$
	    
		$\mathbf{Y}\in\mathbb{R}^{G\times M}$: $S\text{-}conv(\cdot)$의 입력
		$\mathbf{K}\in\mathbb{R}^{G\times P}$ : Size $P$의 Convolutional kernel
		$\mathbf{y}_j\in\mathbb{R}^{1\times M}$ : 행렬 $\mathbf{Y}$의 $j$ 번째 row 
		$\mathbf{k}_j\in\mathbb{R}^{1\times P}$ : 행렬 $\mathbf{K}$의 $j$ 번째 row

<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d53467e1-9b71-4c98-a264-b5c1e49c64f1/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T125617Z&X-Amz-Expires=86400&X-Amz-Signature=8c303fc57ad2fab8303c64fa99d41fb9bc5c7dee3e2c766ec2bdf86485246c84&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width="300px"/><br>
<b>Figure 6.</b> Depthwise Separable Convolution<br>
(출처 : <a href="https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec">Medium blog</a> )
</center>

-   Kernel size $\mathbf{\hat{K}} \in \mathbb{R}^{G\times H \times P}$의 standard convolution과 비교하여, depthwise separable convolution은 $G\times P+G\times H$개의 parameter로 모델 사이즈를 대략 $P$만큼 줄였다.


### Bottleneck layer

- Figure 4를 보면, separation module의 앞 부분에는 linear $1\times 1\text{-}conv(\cdot)$ block 하나가 bottleneck layer로 존재한다. 이는 Input channel의 수와 convolutional block들의 residual path channel의 수를 뜻하는 $B$를 결정하는 역할을 한다.

-   Figure 5를 보면, 1-D conv block의 앞 부분에도 bottleneck layer가 존재하는데, 이 $1\times1\text{-}conv(\cdot)$에 의해 feature dimension 즉, input과 residual path의 channel 개수를 결정해준다.


- 예를 들어, 1-D conv block의 Input channel이 $B$라고 하면, $1\times1\text{-}conv(\cdot)$에 의해 $H$로 확장해준다.
-   Depthwise separable convolution의 뒷부분에 있는 $1\times1\text{-}conv(\cdot)$에 의해 Skip connection 및 Output의 Channel을 다시 $B$ 및 $Sc$로 변환해준다. 가장 높은 성능을 보이는 Hyperparameter 설정은 $B = Sc$이기 때문에 같은 Channel로 바꿔주는 것을 볼 수 있다.

???
