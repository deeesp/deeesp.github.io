---
title:  "[NLP 01] Natural Language Processing Overview "
excerpt: "NLP 자연어처리의 전반적인 경향 파악"

categories:
  - Natural Language Processing
  
tags:
  - Natural Language Processing
  - Artificial Intelligence
  - NLP
  - 자연어처리
  - 인공지능
  
  
last_modified_at: 2020-03-18

toc: true
toc_sticky: true
---


# NLP Overview

## 1. Natural Language Processing


### Natural Language란?

- 한국어, 영어, 프랑스어, 스페인어, 중국어와 같이 사람의 사용하는  수 많은 *Human languages*를 말한다.
- 자연어 처리 in Korean
> 컴퓨터는 '계산기'에 불과하다. 컴퓨터는 사람의 말, 즉 자연어<sup>Natural Language</sup>를 바로 이해할 수 없다.
> 자연어를 100% 이해하는 인공지능이 등장하더라도 그 이해<sup>Understanding</sup>의 본질은 연산이나 처리이다.
> 컴퓨터가 자연어를 계산 가능한형식으로 바꾸어 주어야 한다.

<div style="text-align: right">
  이기창님의 서적 [한국어 임베딩] 중 </div>

### Natural Language Processing(NLP) 이란?

- 말 그대로 Natural Language에 관한 Task를 Process해주는 것으로, 언어의 Properties를 이해하기 위해 Preprocessing(전처리)를 해주는 과정이다.
- 다양한 응용 Task가 있다.
e.g. Understanding(가장 기본), Translation, Question Answering, Paraphrasing etc.
- Understanding: Source를 이해하고 목적에 맞게 처리하는 과정을 주로 NLP라고 부른다.
- Computational Linguistics와 비슷하나 이는 Grammar와 Logic과 같은 Formal language와 Computer Science 분야에 집중하고, NLP는 언어 그 자체에 좀더 집중한다.

## 2. Academic position


![NLP%20Overview/Screen_Shot_2020-03-18_at_2.54.03_PM.png](/_posts/NLP_Overview/Screen_Shot_2020-03-18_at_2.54.03_PM.png)

- NLP는 다양한 분야가 합쳐진 연구분야이다.
- AI는 Computer Science와 Statistics를 기반으로 한다.
- Sound, symbol: before entering NLP, sound analyze needed

## 3. AI methods applied to NLP

![NLP%20Overview/Screen_Shot_2020-03-19_at_8.06.33_PM.png](/_posts/NLP_Overview/Screen_Shot_2020-03-19_at_8.06.33_PM.png)

### Three Major Approaches

1. **Logic:** 과거에는 가장 좋은 Representation이였고,  Interpretable하였다.
2. **Statistics**: Logic 기반의 모델은 최적화에 한계가 있어 2014까지는 통계를 기반으로 한 Markov 모델과  Bayesian networks를 주로 이용하였다.
3. **Neuroscience**: 현재까지는 State-Of-The-Art 모델로 Neural Networks 모델이 주를 이루고 있다. Kernel Machines도 비슷하여 아직 사용 되고 있긴 하다.

### Genetic programming

---

- Search Algorithm, Optimization Algorithm에 유용하다.
- High computational cost 를 요구하므로, 금수저라면 시도해도 좋다.
- Google도 이 방법으로 NN을 찾는데 접근하였다.
- Meta-learning, Auto-ML이 관련있다.
- 대부분 집단에서는 impractical하다고 생각한다.

## 4. A summary of NLP

---

![NLP%20Overview/Untitled.png](/_posts/NLP Overview/Untitled.png)

### Steps

---

For understanding a single sentence

- Lexical Analysis: Word 성질을 이해하기 위한 단계
- Syntactic Analysis: 단어 사이의 관계, dependency를 분석하는 단계
- Semantic Analysis: 문법적인 관계를 이해하는 단계

- Discourse Integration: 위에 분석된 문장으로, 문장과 문장 사이의 관계를 분석
- Pragmatic Analysis: 실용적인 단계로, 전체 문서를 요약하는 단계이다.

### Applications

---

1. Question Answering: 많은 global 회사들이 이 분야를 응용하고 있으며 정부가 투자도 많이 한다.
2. Text Classification
3. Speech Recognition: NLP 분야에서 큰 파트를 차지하지만, NLP와는 별개라고 생각하는 집단도 많지만 넓게 보면 NLP분야라고 할 수 있다. 이 task만의 커뮤니티가 있다.
4. Machine Translation: 복잡하고 큰 task로, 문장의 properties를 이해하여 Source에서 Target으로 번역해야 한다. 이걸 해결하면 대부분의 NLP task를 해결할 수 있다.
5. Caption Generation: Vison 연구와 가까우며, 이미지를 이해하고 caption을 생성해낸다.
6. Language Modeling: 전통적이고 중요하며, 기본이 되는 task이다. grammar와 관련있으며, 이 또한 잘 해놓으면 다른 task에 다양하게 사용 가능하다.
7. Document Summarization: 문서 및 문장을 이해하고 제목을 만들어 준다. 성능이 괜찮아 이미 산업에서 많이 사용되고 있다.

### Input / Output

---

- 크게 Written Text(문어체)와 Speech(Spoken Language - 구어체)로 나뉜다.
- 요즘엔 Vison 분야와 연관시켜 연구하기도 한다.
- 내 생각) 실제로 문어체를 사용하는 pretrained 언어 모델로 구어체를 많이 사용하는 금융 문자를 분석했을 때 Performance도 안 좋았을 뿐만 아니라 Inference time 도 현저히 떨어지는 현상을 볼 수 있다.
[https://dacon.io/competitions/official/235401/overview/](https://dacon.io/competitions/official/235401/overview/)

### Challenges

---

- 이 중 1~3가지를 선정하여 performance를 향상시키는 것이 연구의 목표이다.
- Pragmatics, Phonology, Lexicon, Semantics, Morphology, Syntactic, Discourse Analysis

## 5. Brief History

---

- 전체 다 자세히 볼 필요는 없지만, NLP task에 예부터 어마어마한 노력을 했는지 볼 수 있다.
- 이를 통해 Linguistics 종류와 features를 파악하고 더 나은 모델을 만드는데 집중하는 것이 더 중요하다.
- Machine Translation은 고성능 컴퓨터와 Neural Networks가 등장하기 전 60년간 문제를 풀기 어려웠다.
- 대부분의 모델이 Grammar에 기본을 두고 있다. 지식의 일반화된 representation
- 문법 Rule-based system으로 모델링 하면 obsolete하다고 여긴다. 문법은 이론을 기반으로 하고 있어, 모든 언어의 표현을 grammar로 디자인할 때, 도메인 지식이 기반으로 튜닝하여 성능을 높여야 하기 때문에 우리 분야와 거리가 멀고 unstable 때문이다.
- 따라서 Grammar는 good representation이지만, 도메인 전문지식을 응용해야 하기 때문에 최적화 관점에서 매우 복잡하여 사람들이 통계 모델로 눈을 돌렸다.
- NN모델을 찾기 전까진 probabilistic grammar 모델의 성능이 제일 좋았다  ex, syntax parsing
- 하지만, 여전히 도메인 지식으로 feature, structure 를 사용하는 분야도 있다.

## 6. Why NLP is important?

---

- AI 관점으로 보면, NLP는 one of the largest research areas and practical area to use symbolic, discretized
(continuous for vision / discrete or categorical for nlp / time-series values for acoustics)
- NLP 분야에서 발견된 방법론들은 다른 discretized models를 사용하는 연구분야에도 응용될 수 있다.
→ 예를 들어, MIDI representation과 Transformer 모델을 사용하여 곡을 생성하는 Music Transformer 에서도 NLP분야에서 사용된 연구를 바탕으로 응용되었다.
[https://magenta.tensorflow.org/music-transformer](https://magenta.tensorflow.org/music-transformer)
- 매우 복잡하고, 리소스가 많이 필요하지만, 임펙트가 크다.
- NLP 분야는 인간의 지식을 다루고, 이 지식을 어떻게 표현하고 이 지식으로 inference 하는지 연구하는 main 분야 이기 때문에 중요하다.
- 미래엔 지식을 Black box 로직으로 돌아가는 AI모델로 학습할 것이다.

## 7. Word

---

---

- 자연어 처리의 시작은 word 분석에서 시작한다.
- We should recognize that all the substrings are words which are divided by segmentations.
- Segmentation is a starting point of NLP
- 

> This     is     a    simple    sentence

## 8. Morphological Analysis

---

- **Morphology(형태론, 어형론)**: Linguistics 에서 배우기 때문에 친숙하지 않지만, NLP에서 주로 분석해야 하는 부분이기도 하다.
- recognize the word and category
- Morphing
- understanding root form of its transformation depending on languages
- various words have same meaning → phenotypic representation??
- root form is the genotypic representation

### Morphological Transformation에서 얻을 수 있는 효과는?

- statistical method → critical
- all different phenotypic representations are regarded as different observations
- it reduces the frequency of observing the same outcome of r.v.
- the reliability of probability distribution observed from data can be decreased  ??? 뭐라는거야?
- morphologically different representations 분석하는 이유??
- root form → additional change
- what indicate core contents
- understand category of words

## 9. Part Of speech (품사)

---

- Penn Tree Bank: 예시로 보여 준 표로, syntax, Morphological Anlaysis에 사용되는 유명한 Corpus(말뭉치)이다.
- Gold standard Table 을 만들기 힘들고, 이 때문에 서로 다른 standard table을 기반으로 연구가 진행되어 서로 호환이 되지 않아 문제가 많다.
- 하지만 End-to-End 모델인 NN이 해결해준다.

## 10. Syntax Analysis

---

- Syntactic relation between words
- part of speech analysis → normal syntax analysis
- analyze part of the speech and words together : make the symbol including word information
- sophisticated, accurate, generalized
- understand dependency between words
- can be done by applying grammar

## 11. Semantics

---

- with results of syntax analysis, we know the relation between words
- Logic Representation: syntax를 기반으로 만들어진 semantic representation
- 계층적 구조를 가지고 있다.
- POS를 분석한 후에 그들의 Entity(Property, Aspect)를 결정한다.
- Ontology: Computer Science와 Medical 분야와 같이 쓰이는 언어가 다르기 때문에 Domain이 일반화 될 수 없다.
- 주로 Parsing: what component can be combined with what component?

## 12. Discourse

---

- 앞서 소개한 것들에 비해 다소 뒷단에 위치한 단계
- Anaphora Detection: 누락된 것이 실제로 의미하는 것이 무엇인지
- 문장에서 문단 분석으로 확장하는 데에 중요한 분석이다.

## 13. Why NLP is hard?

---

### 1. Ambiguity

---

Bank : 이거 하나만 보면 뜻이 뭔지 몰라. finance or river

Quantifier scope

Discourse : 뭐가 원인인지 모르넹 ㅎ

correct analysis 를 어떻게 선택하나? 다 맞아 ㅠ

이게 문제다...

### 2. Sparse Data

---

- In AI perspective, sparse data 가 겁나 많다. 근데 Sparsity is strong in NLP research
- Token: Basic Level of Morphological analysis, components smaller than words which make analysis easier sometimes equivalent to words
- word frequency vs rank
- Low-Frequency words에 대한 처리
- 50th rank 이후 대부분의 단어가 집중되어 있다.
- 따라서 frequency와 rank 관계를 log-스케일로 나타낸다.

  

![NLP%20Overview/Screen_Shot_2020-03-19_at_8.55.48_PM.png](/_posts/NLP Overview/Screen_Shot_2020-03-19_at_8.55.48_PM.png)

### 3. Variation, Expressivity, Strong Context Dependency

---

For Variation

- 답이 참 많다.
- Unknown words에 대한 처리도 문제이다.

For Expressivity

- 같은 의미여도 표현할 수 있는 방법이 많다.

For Strong Context Dependency

- 역사적으론 task를 잘게 쪼개서 분석하였다. 하지만 task 사이의 dependency문제가 있다.
- End-to-End NN 모델이 이를 많이 해결해 주고 있다.

## 14. Academic Community

---

1. The biggest society
ACL: The association of computational Linguistics
LDC: Linguistics Data Consortium → 데이터 만들고 업로드하고, 여기서 판다. 데이터와 아이디어가 여기서 많이 공유된다.

2. Related Conferences
- NLP General
→ top3 conference
- NIPS, ICML: Optimization, Model Representation, Generalization
- AAAI: Traditional ML, AI → Statistical or logic based model
- COLING: More related to Linguistics
- Knowledge Mining
- How to extract knowledge


## Reference
---
- 본 포스트는 GIST 김강일 교수님의 Natural Language Processing 강의를 참조하였습니다.
