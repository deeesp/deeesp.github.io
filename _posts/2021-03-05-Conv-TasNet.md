---
title:  "[Speech] Conv-TasNet 톺아보기"
excerpt: "Time-domain single-channel speech separation, Conv-TasNet 분석, 정리"
categories:
  - Speech
  
tags:
  - Conv-TasNet
  - Speech
  - Speech separation
  - Source separation
  - 컨브테스넷
  - 음성신호처리
  - 음성분리
  - 음원분리
  - Speech Processing
  - Deep Learning
  - 딥러닝

last_modified_at: 2021-03-06-14:00:00

toc: true
toc_sticky: true

---

## 서론

Speech separation 분야의 역사에 한 획을 그은 [Conv-TasNet](https://ieeexplore.ieee.org/abstract/document/8707065)은 2019년도 IEEE/ACM TASLP (Transactions on Audio, speech, and language processing) 저널에 출판된 논문으로, 여전히 separation과 enhancement 분야에서 base-line이 되고 있습니다. Conv-TasNet을 읽고, 분석 및 정리해 보았습니다.

<br><br>

## TL;DR

<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c7ae61e5-958f-403b-85e1-84b16c282861/speech_separation_on_wsj0-2mix.jpeg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T031636Z&X-Amz-Expires=86400&X-Amz-Signature=e6d0a19fff2e5255bcfda31f10033c63cee0cf33631f611dd62045b7ebb2c957&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22speech_separation_on_wsj0-2mix.jpeg%22" height="100px" /><br>
<b>Fig. 1</b>: Speech separation SOTA performance on wsj0-2mix <br> (출처 : <a href="https://paperswithcode.com/sota/speech-separation-on-wsj0-2mix">Papers with code</a>)
</center>
 <br>
  
1.  Conv-TasNet은 각 speech source에 대한 mask를 time-domain에서 직접 estimation하는 speech seaparation<sup>음원 분리</sup> 모델로, 성능적으로 상당한 breakthrough를 이뤄낸 모델이다. 현재까지도 Conv-TasNet을 기반으로 한 다양한 변형 모델이 나오며 성능향상을 보이고 있다. (e.g., [DPRNN](https://ieeexplore.ieee.org/abstract/document/9054266) (ICASSP 2020), [DPTNet](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2205.pdf) (Interspeech 2020), [SepFormer](https://arxiv.org/abs/2010.13154) (ICASSP 2021) 등)
	- wsj0-2mix dataset 기준 SI-SNRi 15.3dB, 기존 state-of-the-art 모델에서 4dB에 가까운 성능 향상
 <br><br>
3. Speech separation task에서는 기존 접근 방법처럼 mixture signal을 time-frequency representation (즉, STFT를 통한 spectrogram representation)에서 처리하면 다음과 같은 이유로 suboptimal하기 때문에, time-domain approach로 다음 문제를 해결하였다.
	- Spectrogram 계산 시 high-resolution frequency를 필요로 하기 때문에, 긴 temporal window를 이용해 STFT를 하게 되고, long latency를 야기한다.
	- 또한, signal의 phase와 magnitude가 decoupling되기 때문에, clean source들의 phase reconstruction 시 정확도에 upper bound를 야기한다.
 <br><br>
4. Speech waveform을 speech separation task에 optimal한 latent represenatation으로 만들어 End-to-End deep learning framework을 구축하였다.
	- Latent representation을 만들어 주는 linear autoencoder는 SI-SNR loss를 최소화 해주도록 separator와 jointly training하였고, 실제 구현 시에는 1-D convolution과 1-D transposed convolution을 이용해 구현하였다.
 <br><br>
5. 물론, 다음과 같은 제약이 상당히 들어가 있어 추가 연구가 필요하다.
	- Microphone이 하나인 single-channel 모델이다.
	- Clean speech 환경이다. (noisy하고 reverbrant한 환경에서는 성능 안좋음)
	- 겹치는 source의 수를 알고 있어야 한다. (Unknown number of speakers)
	- Real-world에서 회의 상황과 같은 continuous speech spearation 문제는 풀기 힘들다.

<br><br>

## [1] Time-domain Speech Separation
<br>
Single-channel<sup>*</sup>에서 각기 다른 speech source를 분리하는 single-channel speech separation에 대한 문제를 먼저 정의해보자.

* **Single-channel** : Microphone (이하 MIC)이 하나인 조건으로, 인간으로 비유를 하자면 한 쪽 귀로만 들어<sup>Monaural</sup> 공간 정보가 없는 조건을 말하며, MIC 개수를 언급할 때에는 channel로 표기함 (e.g., signle-channel, multi-channel etc.)
<br>

### [1]-1. Problem Statement

길이가 $T$이고, $C$개의 speech source가 섞여 있는 discrete-time waveform input mixture인 $x(t) \in \mathbb{R}^{1\times T}$ 가 주어졌다고 하자. 이 때, $C$개의 source들은 각 $s_1(t), s_2(t), ...,s_C(t) \in \mathbb{R}^{1\times T}$로 표기하며, 이 source들을 time-domain에서 직접 estimation 해내는 것을 목표로 한다.
    
$$x(t) = \sum^C_{i=1}s_i(t)$$


- 전반적으로, 다음 block diagram을 따라 separation이 진행된다.


<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b6b5c339-f2e4-44eb-8701-212894b7ac98/Screen_Shot_2021-01-12_at_4.26.19_PM.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T032604Z&X-Amz-Expires=86400&X-Amz-Signature=2bc4ac6cca186499f691d192a4020c5c7ff8c57c8f33a48f306596a82176a19b&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Screen_Shot_2021-01-12_at_4.26.19_PM.png%22" height="150px" /><br>
<b>Fig. 2</b>: Time-domain audio separation network block diagram
</center>

> ★ 기본적으로 frame단위의 mixture에 대한 latent represenatation에 각 source에 해당하는 mask들을 씌워 separation한다.
<br>

### [1]-2. Input

- $x(t) \in \mathbb{R}^{1\times T}$를 길이가 $L$인 $\hat{T}$개의 overlaaping segment $\mathbf{x}_k \in \mathbb{R}^{1\times L}$로 나누어 준다. (단, $k=1,...,\hat{T}$)
- $\hat{T}$개의 waveform segment $\mathbf{x}_k$ 들을 각각 encoder 단으로 넣어준다.

사실상 $X\in\mathbb{R}^{\hat{T}\times L}$이 한꺼번에 encoder로 들어가는 것이지만, 아래 설명은 각 segment (또는 frame) 별로 다뤄지고 있다. $L$은 frame 개수를 결정하는 아주 중요한 hyperparameter로, 뒤에 설명하겠지만 작을수록 성능이 좋아졌다. 물론 $L$이 작아지면 $\hat{T}$는 커진다.

<br>

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
<br>    

### [1]-4. Separator part

1.  $C$개의 vector (또는 mask) $\mathbf{m}_i \in \mathbb{R}^{1 \times N}, i=1,2,...,C$를 추정해낸다.

	(단, $\sum^{C}_{i=1} \mathbf{m}_i = \mathbf{1}$)
	
	→ Mask를 추정하는 방법은 잠시 후 자세히..
    
2.  Mixture representation $\mathbf{w} \in \mathbb{R}^{1 \times N}$에 각 $\mathbf{m}_i$를 element-wise multiplication을 하게 되면, 각 source의 encoded representation $\mathbf{d}_i \in \mathbb{R}^{1 \times N}$ 이 나온다. 간단히 말해, mixture에 weighting function (mask)를 씌워 source separation을 한다.
    
    $$\mathbf{d}_i = \mathbf{w}\odot\mathbf{m}_i$$    
<br><br>

## [2] Convolutional Separation Module
<br>

### [2]-1. 특징

1.  Mixture $\mathbf{x}$에서 각 Source $s_i$를 separation하기 위한 mask $\mathbf{m}_i$를 추정하는 module
2.  Temporal Convolutional Network (TCN)에서 영감을 받아, 1-D dilated convolutional block을 여러 층 쌓아 convolution으로만 구성되어 있다.
    -   Sequence modeling에 쓰이는 RNN 계열 모델은 Long-term dependency를 보는 데에 유용하게 쓰이지만, recurrent connection 때문에 parallel processing에 제한이 있어 느리다.
    -   따라서, RNN 계열 모델을 대체하여 Long-term dependency를 볼 수 있고, Parallel processing이 가능한 TCN을 사용한 것이다.
3.  Standard convolution 대신에 쓰인 depth-wise convolution은 parameter 수와 compuational cost를 줄여주었다.
<br>

### [2]-2. Temporal convolutional Network (TCN)
이 모델에서 쓰인 TCN 구조는 [WaveNet](https://arxiv.org/abs/1609.03499)에서 쓰인 dilated convolution과 residual path, skip-connection path 구조를 가져와 응용한 것이다. Dilation을 주면 큰 temporal context window를 만들어 줄 수 있어 speech signal의 long-range dependency를 잡아내는 데에 좋다.

아래 Figure 3는 WaveNet에서 쓰인 구조인데, $X=4$인 한 layer를 표현한 것이라고 볼 수 있다.

<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a869fd9f-8ace-4261-a356-7c9fa7e52661/Screen_Shot_2021-01-12_at_8.28.36_PM.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T120940Z&X-Amz-Expires=86400&X-Amz-Signature=4f894cb469295c7b26441a705dcc9b0a47ee351a581de6a7a870d49356bdd9aa&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Screen_Shot_2021-01-12_at_8.28.36_PM.png%22" height="150px" /><br>
</center>
<br><br>
이러한 dilated convolution을 포함한 TCN 구조의 Conv-TasNet 전체 block diagram을 보면 다음과 같다.

<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c03c8b0c-e549-4be0-b0b5-a49c0ba08ff6/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T121554Z&X-Amz-Expires=86400&X-Amz-Signature=027f968cc7a650f926a5e9ab5448096fa0ea54102f6e9136fe764e73e088f976&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" height="150px" /><br>
<b>Figure 4.</b> Conv-TasNet Block Diagram<br>
(출처 : SAPL Seminar of Ph.D candidate 변재욱)
</center>
<br><br>
-   각 Layer는 $X$개의 각 convolutional block들로 이루어져 있고, 각 block의 dilation factor는 $1, 2, 4, ..., 2^{X-1}$ 로 증가하는 형태를 띈다. 또한, 이 layer는 $R$ 번 반복된다.

-   TCN의 출력은 Kernel size가 1인 $1\times 1$ convolution (a.k.a Point-wise convolution)을 통과하게 되고, Non-linear activation function인 Sigmoid를 지나 $C$ 개의 Mask vector를 추정한다.
<br>

### [2]-3. 1-D convolutional block

TCN에서 반복적으로 쓰인 1-D convolutional block을 자세히 알아보자.
-   각 block의 입력은 출력과 길이를 같게 해주기 위해, zero-padding 해준다.
-   Residual path : 각 block의 input을 다음 block으로 넘겨준다.
-   Skip-connection path : 각 block에서 나오는 output을 더해 최종 TCN의 출력으로 사용한다.
-   [MobileNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html)에서 쓰였던 depthwise separable convolution $S\text{-}conv(\cdot)$ 테크닉을 가져와 standard convolution을 대체하여 Parameter 수를 줄여주었다. 


<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a353a70f-da72-4c70-a5d6-7ee9ca29e951/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T124645Z&X-Amz-Expires=86400&X-Amz-Signature=76beaead19a9a54126e7db1683f47711e26947d5a906b81ea0738b4cf6e31322&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width="200px"/><br>
<b>Figure 5.</b> 1-D Convolutional Block
</center>
<br>

**Depthwise separable convolution**
- $S\text{-}conv(\cdot)$는 Figure 5 처럼 차례로 depthwise convolution $D\text{-}conv(\cdot)$, pointwise convolution $1\times 1\text{-}conv(\cdot)$으로 구성되어 있다. (처음에 보이는  $1\times 1\text{-}conv(\cdot)$는 Bottleneck)

	1) $D\text{-}conv(\mathbf{Y},\mathbf{K})$ 는 입력 $\mathbf{Y}$의 각 row와 상응하는 행렬 $\mathbf{K}$의 row에 대해 Convolution 연산을 한다.
        $$D\text{-}conv(\mathbf{Y},\mathbf{K}) = \text{concat}(\mathbf{y}_j\circledast \mathbf{k}_j),\ j=1,...,N$$
    
	2)  $S\text{-}conv(\mathbf{Y},\mathbf{K},\mathbf{L})$는 $D\text{-}conv(\mathbf{Y},\mathbf{K})$ 와 Convolutional kernel $L$의 Convolution으로, $1\times 1\text{-}conv(\cdot)$를 통해 Linear하게 Feature space로 변환해준다. $L \in \mathbb{R}^{G\times H\times 1}$ : Size 1의 Convolutional kernel
	    
	    $$S\text{-}conv(\mathbf{Y},\mathbf{K},\mathbf{L})=D\text{-}conv(\mathbf{Y},\mathbf{K}) \circledast \mathbf{L}\\ $$
	    
	- $\mathbf{Y}\in\mathbb{R}^{G\times M}$: $S\text{-}conv(\cdot)$의 입력
	- $\mathbf{K}\in\mathbb{R}^{G\times P}$ : Size $P$의 Convolutional kernel
	- $\mathbf{y}_j\in\mathbb{R}^{1\times M}$ : 행렬 $\mathbf{Y}$의 $j$ 번째 row 
	- $\mathbf{k}_j\in\mathbb{R}^{1\times P}$ : 행렬 $\mathbf{K}$의 $j$ 번째 row


<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d53467e1-9b71-4c98-a264-b5c1e49c64f1/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210305T125617Z&X-Amz-Expires=86400&X-Amz-Signature=8c303fc57ad2fab8303c64fa99d41fb9bc5c7dee3e2c766ec2bdf86485246c84&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22" width="300px"/><br>
<b>Figure 6.</b> Depthwise Separable Convolution<br>
(출처 : <a href="https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec">Medium blog</a> )
</center>

-   Kernel size $\mathbf{\hat{K}} \in \mathbb{R}^{G\times H \times P}$의 standard convolution과 비교하여, depthwise separable convolution은 $G\times P+G\times H$개의 parameter로 모델 사이즈를 대략 $P$만큼 줄였다.
<br>

### [2]-4. Bottleneck layer

- Figure 4를 보면, separation module의 앞 부분에는 linear $1\times 1\text{-}conv(\cdot)$ block 하나가 bottleneck layer로 존재한다. 이는 Input channel의 수와 convolutional block들의 residual path channel의 수를 뜻하는 $B$를 결정하는 역할을 한다.

-   Figure 5를 보면, 1-D conv block의 앞 부분에도 bottleneck layer가 존재하는데, 이 $1\times1\text{-}conv(\cdot)$에 의해 feature dimension 즉, input과 residual path의 channel 개수를 결정해준다.


- 예를 들어, 1-D conv block의 Input channel이 $B$라고 하면, $1\times1\text{-}conv(\cdot)$에 의해 $H$로 확장해준다.
-   Depthwise separable convolution의 뒷부분에 있는 $1\times1\text{-}conv(\cdot)$에 의해 Skip connection 및 Output의 Channel을 다시 $B$ 및 $Sc$로 변환해준다. 가장 높은 성능을 보이는 Hyperparameter 설정은 $B = Sc$이기 때문에 같은 Channel로 바꿔주는 것을 볼 수 있다.

???
<br><br>



## Training Objective

Source separation의 Evaluation metric으로 쓰이는 Scale-Invariant Source-to-Noise Ratio (SI-SNR)을 직접적으로 최대화해주는 것을 Training Objective 로 한다. Training 과정에서 Utterance-level permutation invariant training (uPIT) 방법이 Source permutation problem을 해결하는 데에 사용된다.

[Measures for Source Separation Evaluation](https://www.notion.so/Measures-for-Source-Separation-Evaluation-a657796ce36a4cbc978b8d121ca49168)

## 실험

### Dataset

-   WSJ0-2,3mix datasets
-   Speech mixture들은 WSJ0 dataset에서 임의 추출한 utterance들을 ±5dB 사이의 random한 SNR로 섞어 생성되었다.
-   Evaluation set은 16명의 unseen speaker의 utterance들을 사용해 같은 방법으로 만들어짐.
-   모든 waveform들은 8kHz로 resampling되었다.

### Experiment Configurations

-   100 epochs on 4-sec long segments
-   Init learning rate : $1\text{e}^{-3}$
-   3연속 epoch에서 validation set의 accuracy에 발전에 없으면 learning rate 반으로 줄임
-   Adam optimizer 씀
-   Training할 때 maximum $\text{L2-norm of 5}$이 Gradient clipping으로 적용됨


## Non-negativity of Encoder (이부분 추가적인 이해 필요)

-   ReLU fucntion으로 encoder ouput에 non-negativity를 강제하는 constraint는 다음 가정을 기반으로 한다.
    
    > Unbounded encoder representation은 unbounded mask를 야기할지도 모른다. 따라서, encoder ouput에 non-negativity를 강제하는 것은 encoder output에 대한 masking operation이 mixture와 speaker waveform이 basis function들의 non-negative combination으로 표현될 때에만 의미가 있다.
    
-   그러나, nonlinear function $\mathcal{H}$를 없애면 또 다른 가정이 만들어진다.
    
    > Unbounded 하지만 highly overcomplete한 mixture representation으로도, clean source들을 reconstruction하기 위한 non-negative mask set은 여전히 찾을 수 있다.
    
    [Overcomplete Autoencoder](https://www.notion.so/Overcomplete-Autoencoder-aff74ab6dd4b4e899a8389f950a83eb8)
    
-   여기서 representation의 overcompleteness는 아주 중요한 역할을 한다.
    
-   만약 각 source들 뿐만 아니라 mixture에 대해서도 unique한 weight feature가 하나만 존재한다면, mask의 non-negativity를 보장할 수 없다.
    
-   또한 위의 두 가지 가정에 주의하여, encoder와 decoder의 basis function인 $\mathbf{U}$와 $\mathbf{V}$ 사이의 관계에 constraint을 주지 않는다. 즉, mixture signal을 완벽하게 reconstruction하도록 강제하지 않는다.
    
-   Auto-encoder의 특성을 명확하게 해주기 위한 방법으로는 $\mathbf{V}$를 $\mathbf{U}$의 pseudo-inverse로 선택하는 방법이 있다. (즉, least-square reconstruction)
    
-   Encoder와 decoder 설계는 mask estimation에 영향을 준다.
    
-   Autoencoder의 경우에는 unit summation constraint을 반드시 만족해야 하는 반면, unit summation constraint이 반드시 필요한 것은 아니다. (?)
    

그래서 여러가지 encoder/decoder 설정으로 비교하며 실험해봄
<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/072364d2-95a7-472e-84e1-8753d41ab467/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210306%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210306T043309Z&X-Amz-Expires=86400&X-Amz-Signature=1e02f44a06f78627d2408e80988d5e9e26bef69ebe7ea2083b8f86206982a066&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22"/><br>
<b>Table 1.</b> Encoder/decoder configurations<br>
</center>
<br>

Separation 정확도는 Pseudo-inverse autoencoder가 가장 안좋은 성능을 보였는데, 이 framework에서는 explicit autoencoder configuration이 반드시 좋은 separation 성능 향상을 보여주는 것은 아니라는 것을 보여주었다. 다른 설정은 비슷한 결과를 보여주었는데, Sigmoid를 이용한 linear encoder와 decoder가 조금 더 나은 성능을 보여주었다.

<br><br>

## Properties of the basis functions

-   Mixture signal에 대한 STFT representation을 convolutional encoder로 대체하게 된 배경은 speech separation에 optimized된 audio representation을 만들어주기 위한 것
    
-   Encoder와 decoder의 basis function들을 각각 행렬 $\mathbf{U}$, $\mathbf{V}$의 row들이라고 하면, basis function들을 UPGMA method를 이용해 Euclidean distance의 similarity를 기준으로 정렬하여 그리면 다음과 같다.
    
<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f266077f-7f0a-49f4-857f-a68f2df54dc6/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210306%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210306T043532Z&X-Amz-Expires=86400&X-Amz-Signature=19d68673680e112b012e4dcca27ed7be3d7d663875f6d15fc190ddb8c34b6ce6&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22"/><br>
<b>Figure.</b> Encoder/decoder Basis Functions<br>
</center>
<br>

-   각 시점에서 각 speaker에 상응하는 basis ouput의 power에 따라 encoder의 representation에 빨간색과 파란색으로 색을 칠하였는데, 이는 encoder representation의 sparsity를 설명해주고 있다.
-   또한 각 그림의 오른쪽에는 각 filter들에 대한 FFT의 magnitude를 그린 것이고, Basis function들의 다양한 frequency와 phase tuning을 보여준다. 다수의 필터들이 저역대의 frequency에 튜닝되어 있다.
-   같은 주파수로 튜닝된 필터들이 다양한 phase value로 표현되는 것을 볼 수 있는데, low-frequency basis function들의 circular shift가 관찰되는 것으로 볼 수 있다.
-   이 말은 즉슨, 더 우수한 speech separation 성능을 내기 위해서는 speech에서 phase information의 explicit encoding 뿐만 아니라 pitch와 같은 low-frequency feature들이 중요한 역할을 한다는 것을 말해주고 있다.

<br><br>

## Hyperparameter Tuning

### Conv-TasNet Version 2
<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6247899f-9034-44c4-80d2-7fa8f6a56f7a/KakaoTalk_Photo_2021-03-04-00-25-06.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210306%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210306T043738Z&X-Amz-Expires=86400&X-Amz-Signature=2452c1e1d9b4793061bbd7ebd5faa94e0aad6f26a138085f6dfde69efd2782b3&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22KakaoTalk_Photo_2021-03-04-00-25-06.png%22"/><br>
<b>Table.</b> Conv-TasNet v2 Hypterparameter Table w/o Skip-connections<br>
</center>
<br>



### Conv-TasNet Version 3
<center>
<img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4b2a284e-964e-4aca-be19-55e209699fd3/KakaoTalk_Photo_2021-03-04-00-25-48.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210306%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210306T043740Z&X-Amz-Expires=86400&X-Amz-Signature=b58bd72093c8622ca0d5a4725890ae001a3635b11c9f57f9ea12c2aa1bb98702&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22KakaoTalk_Photo_2021-03-04-00-25-48.png%22"/><br>
<b>Table.</b> Conv-TasNet v3 Hypterparameter Table w/ Skip-connections<br>
</center>
<br>

1.  $N$ : # of filters in autoencoder Encoder와 decoder의 basis signal 개수가 커질수록 basis signal의 overcompleteness가 증가하고, 성능향상을 보였다.
    
2.  $L$ : Length of the filters in samples Segment 길이가 작을수록 좋은 성능을 보였다. $2ms$가 best performance 였는데($L/fs = 16/8000=0.002s$), encoder ouput의 time step이 커지면 같은 크기의 $L$에도 Deep LSTM network를 training시키기가 힘들었다.
    

	- 더 작을 때는?? Pattern 을 파악할 때, 조금씩 보고 파악하는 게 좋은듯? 정보를 놓치는게 적을 것이다. Window length 작아 → Time resolution 좋아 → performance 좋아 → 커널이 확실히 잡힌다. 길면 0으로 깔려버리고.. 흠... Pitch나 formant 를 잘 캐치하길 바라는것??? Receptive field랑 연관...?

	- SI-SNR과 PESQ를 동시에 살리기 위해 다양한 length를 가진 encoder 를 concat해서 써봄 → Spex : 각각의 장점을 살리기 위해서 Multi-resolution Dimension mis-match → zero-padding

3.  $B$ : # of channels in bottleneck & the residual paths' 1x1-conv blocks
    
4.  $Sc$ : # of channels in skip-connection path's 1x1-conv blocks Skip connection block의 channel 수가 증가하면 성능이 좋아졌지만 model size가 크게 증가하여 성능과 model size 사이의 trade-off 관계가 형성되었다. 여기서는 작은 skip-connection 선택!
    
5.  $H$ : # of channels in convolutional blocks MobileNet에서 Convolutional block과 bottleneck의 비율 $H/B$가 5 정도 나올 때 좋은 성능을 보였다.
    
6.  $P$ : Kernel size in convolutional blocks 여기서는 3으로 고정
    
7.  $X$ : # of convolutional blocks in each repeat Receptive field가 같을 때, NN이 깊어질수록 좋은 성능을 보였다. Model capacity가 증가하여 그런 것으로 보인다. 이 또한 독립변수와 종속변수를 제대로 설정하지 않고 비교한 것 같아 표만으로는 알 수 없는듯
    
8.  $R$ : # of repeats
    
9.  Receptive field Receptive field 크기가 커질수록 speech signal의 temporal dependency를 모델링하는데 중요한 역할을 하기 때문에 좋은 성능을 보였다. (표에서는 명확히 비교가 안된거같은데..) 의미가 있는 hyperparameter인가 싶기도 하고../
    

$$\text{Receptive Field} = (2^X\times R\times (P-1)\times L)/ f_s$$
