---
title:  "[DSP] Fourier Series"
excerpt: "주기 신호에 대한 푸리에 급수 표현"
categories:
  - Signal_Processing
  
tags:
  - 신호처리
  - 신호 및 시스템
  - 푸리에 급수
  - 푸리에 변환
  - Signal Processing
  - Fourier Transform
  - Fourier Series
  - Signals and Systems
  
last_modified_at: 2020-04-17

toc: true
toc_sticky: true
---

# 푸리에 급수

![Fourier Series](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Periodic_identity_function.gif/400px-Periodic_identity_function.gif)

## [1] Harmonically Related Complex Exponentials의 선형결합

- 어떤 신호가 주기를 가지고 있으면, 양수 $T$에 대해서, 다음과 같이 나타낼 수 있다.
$$x(t) = x(t+T) \ \ \text{for all} t$$
- 0이 아닌 최소 양수 $T$를 $x(t)$의 기본 주기<sup>Fundamental Period</sup>로 하며, 기본 주파수<sup>Fundamental Frequency</sup>로 $\omega_0 = 2\pi /T$를 갖는다.
- 우리는 다음 정현파<sup>Sinusoidal</sup> 신호와 복소지수를 기본 주파수 $\omega_0$와 기본 주기 $T = 2\pi/\omega_0$를 가진 기본적인 주기 신호라고 알고 있다.
$$x(t) = \cos \omega_0 t$$
$$x(t) = e^{j\omega_0 t}$$
- 위와 Harmonically Related Complex Exponentials의 집합은 다음과 같이 나타낸다.
$$\phi_k(t)=e^{jk\omega_0t}=e^{jk(w\pi / T)t},\ \ k= 0, \pm 1, \pm 2, /ldots$$
- 각 신호들은 여러 개의 기본 주파수 $\omega_0$를 가지므로, 주기를 $T$로 하는 Harmonically Related Complex Exponentials의 선형결합 형태로 나타낼 수 있다.
- 어떠한 형태의 주기신호든 푸리에 급수 형태로 나타낼 수 있다.
