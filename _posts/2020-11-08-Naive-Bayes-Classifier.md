---
title:  "[ML] Naive Bayes Classifier (나이브 베이즈 분류)"
excerpt: "나이브 베이즈 분류의 가정과 이론"
categories:
  - Machine Learning
  
tags:
  - Machine Learning
  - 머신러닝
  - Deep Learning
  - Naive Bayes Classifier
  - 나이브 베이즈 분류
  
last_modified_at: 2020-11-08-22:00:00

toc: true
toc_sticky: true
---


## 1. Goal

- Learning $P(Y \mid X)$ where $X = <X_1, X_2, ... , X_n>$ estimating $P(X \mid Y)$ and $P(Y)$
- With an assumption, The representation $P(X \mid Y)$ is dramatically simplified

## 2. Inductive bias - Assumption

### What is inductive bias?

- A set of assumptions a learner uses to predict results given inputs it has not yet encountered.
- Every machine learning algorithm with any ability to generalize beyond the training data that it sees has some type of inductive bias, which are the assumptions made by the model to learn the target function and to generalize beyond training data. For example, in linear regression, the model assumes that the output or dependent variable is related to independent variable linearly (in the weights). This is an inductive bias of the model.

### Conditional Independence (조건부 독립)

- $X$ is conditionally independent of the value of $Y$ given $Z$

$$(\forall i,j,k)\ P(X=x_i \mid Y=y_j, Z=z_k) = P(X=x_i \mid Z=z_k)$$

- It means $X$ is possible to be dependent on $Y$ without $Z$

### 따름 정리

- $P(X \mid Y)\ = \ P(X_1,X_2 \mid Y) = P(X_1,X_2,Y)/P(Y)$
                    $=\ P(X_1 \mid X_2,Y)P(X_2 \mid Y)= P(X_1,X_2,Y)/P(X_2,Y)*P(X_2,Y)/P(Y)$
                    $=\ P(X_1 \mid Y)P(X_2 \mid Y)$

$$P(X_1...X_n \mid Y) = \prod_{i=1}^nP(X_i \mid Y)$$

- $X_i$ is conditionally independent of each of the other $X_k$s given $Y$
- Dramatic reduction of the number of parameters from $2(2^n-1)$ to $2n$

## 3. Naive Bayes Algorithm

### 유도

- The probability that $Y$ will take on its $k$th possible value, according to Bayes Rule

    $$P(Y=y_k \mid X_1\ldots X_n)
    = \frac
    {P(Y=y_k) P(X_1 \ldots X_n \mid Y=y_k)}
    {\sum_j P(Y=y_j)P(X_1 \ldots X_n \mid Y=y_j)}$$

- Assuming $X_i$ are conditionally independent given $Y$

    $$P(Y=y_k \mid X_1\ldots X_n)
    = \frac
    {P(Y=y_k)\prod _i P(X_i \mid Y=y_k)}
    {\sum_j P(Y=y_j)\prod _i P(X_i \mid Y=y_j)}$$

- Given a new instance $X^{new}=<X_1 \ldots X_n>$
- Calculate the probability that $Y$ will take on any given value
    - the observed attribute values of $X^{new}$
    - the distributions $P(Y)$ and $P(X_i \mid Y)$ estimated from training data

### Naive Bayes Classification Rule

- The most probable value of $Y$

$$Y \leftarrow \text{argmax}_{y_k}  \frac
{P(Y=y_k)\prod _i P(X_i \mid Y=y_k)}
{\sum_j P(Y=y_j)\prod _i P(X_i \mid Y=y_j)}$$

- Simplified version (the denominator does not dependent on $y_k$)

$$Y \leftarrow \text{argmax}_{y_k}  P(Y=y_k)\prod _i P(X_i \mid Y=y_k)$$
