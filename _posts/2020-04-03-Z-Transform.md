---
title:  "Z-변환과 & 이산시간 LTI 시스템"
excerpt: "Yann LeCun 교수님의 NYU Deep Learning 강의 - CNN 개념잡기"
categories:
  - Signal_Processing
  
tags:
  - 신호처리
  - 신호 및 시스템
  - 푸리에 변환
  - z-변환
  - Signal Processing
  - Fourier Transform
  - Z-transform
  - Signals and Systems
  
last_modified_at: 2020-04-03

toc: true
toc_sticky: true
---

# Z-변환과 & 이산시간 LTI 시스템

## [1] Introduction

- Z-변환(Z-Transform)은 이산시간(Discrete-time)에서의 라플라스 변환(Laplace Transform)이라고 생각하면 된다.
- 즉, 이산시간 신호와 시스템을 분석하는 데에 쓰이는 좋은 도구이다.
- 이산시간 신호나 시퀀스(Sequences) z-domain에 나타낸다.

    $$z \in C$$

- 차분 방정식(Difference equations)를 대수식(Algebraic equations)으로 변환해준다.
- 정확하게 말하자면 LCCDE(Linear Constant-Coefficient Difference Equation)를 풀기 위한 것이다.
- 푸리에 변환(Fourier Transform)은 모든 시퀀스를 수렴시키지 않기 때문에 이를 일반화 하고, 더 표기가 편한 z-변환의 형태로 신호를 분석하기 위해 쓰인다.

## [2] Definition

- 일반적인 Discrete-time signal $x[n]$에 대해서, 다음과 같이 Infinite sum 또는 Infinite Power series라고 불리는 식으로 정의한다.

    $$X(z)= Z\{ x[n]\} = \sum_{n=-\infty }^{\infty}x[n]z^{-n}$$

    - 엄밀히 말하자면, 위 표현은 양방향(Bilateral or Two-sided) z-변환으로, 단방향(Unilateral 0r Single-sided, one-sided) z-변환은 다음과 같다.
    - $n<0$에서 $x[n] = 0$ (즉, Causal sequence)이면, 단방향과 양방향 변환이 같다.

    $$\mathcal{X}(z) = X_I(z)=\sum_{n=0}^{\infty}x[n]z^{-n}$$

    $$x[n] = 0\ \ for\ n<0\ \rightarrow X(z) = X_I(z)$$

- 이 때, $z$는 극좌표(Polar form)에서 복소 값을 가진(Complex-valued) 연속 변수이다.

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e1bbd805-1e98-4bbb-a545-d23f87d2b31f/Screen_Shot_2020-04-03_at_4.31.09_PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e1bbd805-1e98-4bbb-a545-d23f87d2b31f/Screen_Shot_2020-04-03_at_4.31.09_PM.png)

$$z = re^{j\omega} = Re(z) + jIm(z) = r \cos (\omega) + j\sin (\omega)$$

$r$: $z$의 진폭(Magnitude)
$ω$: $z$의 각도(Anlge)

- 즉, z-변환은 복소함수 이므로, 복소 z-평면을 사용하여 표현하고 해석하기 쉽다.
- 이러한 시퀀스와 z-변환 관계는 다음 처럼 표기한다.

    $$x[n] \leftrightarrow X(z)$$

## [3] DTFT와 Z-변환의 관계

### DTFT

$$X(e^{j\omega})=X(z)|_{z=e^{j\omega}} = \sum_{n=-\infty }^{\infty}x[n]e^{-j\omega n}$$

- z-변환은 $x[n]r^{-n}$의 DTFT이다.

    $$X(e^{j\omega})=\sum_{n=-\infty }^{\infty}x[n](re^{j\omega})^{-n}=\sum_{n=-\infty }^{\infty}x[n]r^{-n}e^{-j\omega n}$$

    $$Z\{x[n]\} = DTFT\{x[n]r^{-n}\}$$

- 여기서 $r=1$이면(i.e. $|z|=1$), z-변환은 $x[n]$의 DTFT가 된다.
- 다시 말해, DTFT는 단위 원(Unit circle) $r = 1$ 에서의 z-변환이 된다.

    $$z = re^{j\omega}\ (r=1, \ 0 \le \omega \le 2\pi)$$

### Existence

- 만약에, $x[n]$가 Absolutely summable하지도, Square-summable하지도 않으면, DTFT는 존재하는지 안하는지 모른다. 하지만 z-변환은 존재한다!
- **절대가합(Absolutely summable) - Stable**
- 절대가합(Absolute summability)은 DTFT가 존재하기 위한 충분조건이다.
- DTFT가 존재하기 위해선 $|X(e^{jω})|$가 존재해야 한다.

$$|X(e^{jw})|= \left| \sum_{n=-\infty }^{\infty}x[n]e^{-jwn}\right|\le \sum_{n=-\infty }^{\infty}|x[n]||e^{-jwn}| \le\sum_{n=-\infty }^{\infty}|x[n]| < \infty$$

- **Square-summable - Finite Energy**
- Absolutely summable 하지 않아도, Square-summable 하면 DTFT가 존재한다.

$$\sum_{n=-\infty }^{\infty}|x[n]|^2 < \infty$$

## [4] 수렴범위 (ROC: Region of Convergence)

- DTFT와 비슷하게, z-변환도 모든 시퀀스나 $z$ 값에 대해 수렴 하지는 않는다.
- 시퀀스가 absolutely summable하면, DTFT에서는 $ω$에 대한 연속함수로 수렴하게 된다.
- z-변환도 수렴 범위 내에서만 가능하다.

### Definition

- z-변환이 수렴할 때, 복소 값 $z$의 집합(또는 값들의 범위), $|z|$ 에 의해 결정된다.
- DTFT에서 부터 일반화, z-변환의 정의인 멱급수의 수렴 범위

    $$|X(z)|= \left| \sum_{n=-\infty }^{\infty}x[n]z^{-n}\right|\le \sum_{n=-\infty }^{\infty}|x[n]||z^{-n}| \le\sum_{n=-\infty }^{\infty}|x[n]| < \infty$$

- e.g. $x[n] = u[n]$
- u[n]은 absolutely summable하지 않아 DTFT가 존재하지 않는다.
- 하지만 $r^{-n}u[n]$은 $r>1$이면 absolutely summable 하여 u[n]에 대한 z-변환은 $r = |z| > 1$일 때 존재한다.

## Reference

1. Alan V. Oppenheim, Discrete-time Signal Processing, Pearson Education
2. Hwei Hsu, Schaum's Outline of Signals and Systems, McGraw-Hill
3. GIST 김홍국 교수님의 Digital Signal Processing 강의
