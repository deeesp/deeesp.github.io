---
title:  "[DSP 04] Sampling of Continuous-Time Signals"
excerpt: "Aliasing과 A/D 변환의 탐구"
categories:
  - Signal_Processing
  
tags:
  - 신호처리
  - 신호 및 시스템
  - 푸리에 변환
  - 샘플링
  - Signal Processing
  - Fourier Transform
  - Sampling
  - Aliasing
  - Signals and Systems
  
last_modified_at: 2020-04-06

toc: true
toc_sticky: true
---

# 연속시간 신호 샘플링

## [1] Introduction

- 디지털 신호는 다루기 편하고 재사용이 가능하다.
- 따라서 주로 연속시간 신호를 샘플링하여 이산시간 신호의 형태로 나타내고, Discrete-time LTI system에서 처리한다.
- 이러한 A/D 변환과 복구 과정, 그리고 이 과정에서 생기는 Aliasing 현상과 샘플링 이론 등의 개념에 대해 알아본다.

## [2]  Periodic Sampling

