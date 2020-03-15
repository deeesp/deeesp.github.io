---
title:  "Hash Table"
excerpt: "[Data Structure] Abstract data structure: Hash table"

categories:
  - Algorithm
tags:
  - Algorithm
  - Hash Table
last_modified_at: 2020-03-15

toc: true
toc_sticky: true
---



**해시 테이블 (Hash table)** 이란, **[키(Key)와 값(Value)]**이 하나의 쌍을 이루는 **Abstract data type(추상 자료형)***이다. 이 Key와 Value를 Mapping 시키는 과정을 **해싱(Hashing)**이라고 부른다. 빠른 탐색과 삽입, 삭제 속도를 가져 항목간의 관계를 모형화 하는 데에 유용하게 쓰이는 자료구조 중 하나이다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/785bc0ae-16a1-418d-8f99-569eee5373fd/846E13C1-8206-4B50-9588-F034A4808305.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/785bc0ae-16a1-418d-8f99-569eee5373fd/846E13C1-8206-4B50-9588-F034A4808305.png)

* **Abstract Data Type(ADT, 추상 자료형)**: 인터페이스와 기능 구현 부분을 분리한 자료형. 기능 구현부분을 명시하지 않아(추상화 하여) 사용 시에는 기능이 어떻게 돌아가는 지 몰라도 되는 편리함이 있다. 대표적으로 **스택, 큐, 연결리스트, 딕셔너리**가 있다. *정확한 이해를 위해선 추상화 된 Black box가 어떻게 구현이 되었는지 직접 해보는 것이 좋다.*

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ef280df0-6a94-45b5-b789-be7b16e1b9fd/1280px-Hash_table.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ef280df0-6a94-45b5-b789-be7b16e1b9fd/1280px-Hash_table.png)

Hash Table 예시 (전화번호부)
출처: 위키백과

만약에 이 전화번호부가 **배열**에 알파벳 순서로 되어있다면 이진탐색을 써서 Lisa Smith 씨의 번호를 찾을 수 있을 것이고, O(log n)이라는 시간이 걸린다. 하지만 이름과 전화번호부를 외우고 있는 놈이 있다면 찾을 필요도 없이 이름만 대면 O(1)의 시간으로 바로 찾을 수 있을 것이다. 그놈이 **해시 테이블**이다.

​

**다시 말하면,** 데이터 식별자(Index)가 숫자로 이루어져 데이터를 저장 하는 배열(Array)과 List(리스트)와는 달리 **해시 테이블은** 인터페이스만을 보았을 때에는 **데이터 식별자(Key)로 숫자 뿐만 아니라 문자열, 파일 등을 받아와 Value를 연상해 낸다고 볼 수 있다**.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8d1161ac-56e9-403a-9a9a-4221da00d0a7/CollectionTypes_intro_2x.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8d1161ac-56e9-403a-9a9a-4221da00d0a7/CollectionTypes_intro_2x.png)

## ***1. Same thing, different names***

처음에 이 자료구조를 공부할 때 다양한 이름으로 불려서 다소 혼동이 있었다.

**연관 배열(Associative Array), 맵(Map), 사전(Dicitonary)** 등으로도 불리기도 한다. 물론 Map과 Dictionary 사이에는 아주 미세한 차이가 있다고는 하나 처음에는 무시해도 될 정도이다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/edcd80ca-0481-46a2-9a96-9f64231b508b/associative-array-language-support-1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/edcd80ca-0481-46a2-9a96-9f64231b508b/associative-array-language-support-1.png)

근래에는 이처럼 대부분 언어에서 라이브러리로 이 자료구조를 지원하기 때문에 굳이 직접 구현할 필요는 없다.

​

**Dictionary** (Python)

**Hash** - (Ruby, Perl)

**Map** - (Java, Go, Haskell 등)

**Unordered_map, Map** - (C++)

하지만..! 앞에서 말했듯이 이를 본질적으로 이해하기 위해선 직접 구현할 필요가 있다. 그리 복잡하지는 않으니 꼭 한번 구현해보자. (뒤에서..ㅋㅋ) 구현할 때에는 리스트와 배열로 구현한다. 구현해야 하는 이유는 아래 포프킴씨께서 잘 설명해주신다.

[Hash Table은 프로그래머의 기본기](https://youtu.be/S7vni1hdsZE)

## ***2. 그렇다면 왜 해시 테이블이 요긴하게 사용되나?***

### **평균적으로 탐색, 삽입, 삭제에 O(1)의 시간복잡도를 보인다. ★★★★★**

****- 어떤 항목과 다른 항목의 관계를 모형화하는 데 좋다.

- 중복을 잡아내는 데에 뛰어난 성능을 보인다.

- 웹 서버 등에서 데이터 캐싱(Data Caching)을 하는 데에 사용된다.

- 해싱을 이용해 블록체인 암호화에도 사용된다.

- 그 에도 널리 사용되는 자료구조

## ***3. 단점은?***

### **충돌 (Collision)!!**

-> different keys, same code / different codes, same index

-> 충돌을 줄여주는 **좋.은. 해시 함수(Hash Function)**을 이용하여 효율을 극대화 하여 구현해야 한다. (힘듦ㅠ) **Chaining, Open Addressing 등**을 이용해 collision을 극복하는 방법이 있다.

- 사용률이 0.7보다 커지면 해시 테이블을 **리사이징(Resizing)**해주어야 한다.

- Chaining을 하다보면 한 곳에 집중되는 경향이 있는데, **균형잡힌(Balanced) 해시 함수**를 구현해야 해시 테이블의 진정한 의미를 살릴 수 있다. 그렇지 않으면, O(1)이 아닌 O(n)의 굴레에서 벗어나지 못한다.

- Sorting과 Ordering이 힘들다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c3d2e7f0-7973-4e3f-a4ef-a0128082af53/750px-Hash_table_5_0_1_1_1_1_0_LL.svg.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c3d2e7f0-7973-4e3f-a4ef-a0128082af53/750px-Hash_table_5_0_1_1_1_1_0_LL.svg.png)

Hashtable의 충돌 (Collision)
※ 출처: 위키백과 

## ***4. 좋은 Hash function 이란?***

​

그렇다면 좋은 Hash Function의 조건은 어떠할까?

**다음을 만족해야 한다.**

**(1) Use only the data being hashed**

**(2) Use all of the data being hashed**

**(3) Be determinstic**

**(4) Uniformly distribute data**

**(5) Generate very different hash codes for very similar data**

**- 출처 : CS50, Havard University**

**[※ C++ program for hashing with chaining](https://www.geeksforgeeks.org/c-program-hashing-chaining/)**

![https://postfiles.pstatic.net/MjAxOTAyMDdfMjI2/MDAxNTQ5NTI2ODcyMzQx.v6W8WS9sLNfqKavxeVMajzgPCD6F5GmobFO0nOcXLbAg.dc3SDmwsx63okXqoOerutwxO3VZykaHqw_EHhA95fUog.PNG.jwo0816/hashChaining1.png?type=w773](https://postfiles.pstatic.net/MjAxOTAyMDdfMjI2/MDAxNTQ5NTI2ODcyMzQx.v6W8WS9sLNfqKavxeVMajzgPCD6F5GmobFO0nOcXLbAg.dc3SDmwsx63okXqoOerutwxO3VZykaHqw_EHhA95fUog.PNG.jwo0816/hashChaining1.png?type=w773)

**Advantages:**

1) Simple to implement.

2) Hash table never fills up, we can always add more elements to chain.

3) Less sensitive to the hash function or load factors.

4) It is mostly used when it is unknown how many and how frequently keys may be inserted or deleted.

​

**Disadvantages:**

1) Cache performance of chaining is not good as keys are stored using linked list. Open addressing provides better cache performance as everything is stored in same table.

2) Wastage of Space (Some Parts of hash table are never used)

3) If the chain becomes long, then search time can become O(n) in worst case.

4) Uses extra space for links.

​

※ 출처 : [https://www.geeksforgeeks.org/hashing-set-2-separate-chaining/](https://www.geeksforgeeks.org/hashing-set-2-separate-chaining/)

​

​

여기까지 정리를 하기 위해 아래 영상을 시청해보자. 초보자 입장에서 간결하게 설명을 잘 해준다. (5분) (YouTube에 사실 영어로된 영상도 어마무시하게 많다.)

[해시-해시테이블-해싱 5분만에 이해하기 - Gunny](https://youtu.be/xls6jEZNA7Y)

이분도 정말 기가 막히게 설명해주시는데 목소리가 너무 아름다우시다. 목소리 감상하다 시간갔던 ㅋㅋ (6분 30초)

[[자료구조 알고리즘] 해쉬테이블(Hash Table)에 대해 알아보고 구현하기](https://youtu.be/Vi0hauJemxA)
