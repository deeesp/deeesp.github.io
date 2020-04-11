---
title:  "[Dev 02] Git2 - CLI 기반 버전관리"
excerpt: "CLI를 이용한 git 설치부터 버전관리까지"
categories:
  - git
  
tags:
  - 버전 관리
  - 깃
  - 깃허브
  - git
  - github
  - CLI
  
last_modified_at: 2020-04-11-17:00:00

toc: true
toc_sticky: true
---

## [1] Why CLI for Git rather than other GUI programs

- Git을 제어하는 데에 있어서, Source Tree, Github Desktop, TortoiseGit, CLI 등 다양한 방법이 있다. 이러한 방법 중에, CLI(Command line Interface, 명령어)을 이용하여 버전관리를 해보자.
- 사실, CLI를 제외한 GUI기반의 방법들도 내부적으로는 오리지널 git, CLI를 기반으로 한다.
- CLI 방법은 익숙해지기만 하면 동시다발적인 작업이 한번에 가능하고, 간편하게 git을 다룰 수 있다.
- 또한, GUI를 지원하지 않는 환경에서도 CLI를 통해 git을 제어할 수 있다.

## [2] Mac 설치 방법 (초간단)

1. `cmd + space` 스포트라이트 검색을 통해 `terminal`을 켠 후에 `git` 명령어를 쳤을 때 아래 이미지와 같이 git에 대한 usage 설명이 나오면 이미 git이 설치 되어 있는 상태이다.
    ![git-cli](/images/git-cli.png)
2. 하지만 그렇지 않다면, [https://git-scm.com/](https://git-scm.com/) 에 접속하여 latest source release download를 해준다.
3. 그리고 설치가 되었다면! 다시 1번 과정을 반복하여 설치되었는지 확인한다.

## [3] git에게 특정 디렉토리 관리시키기


## Reference
- [생활코딩 지옥에서 온 GIT 2](https://opentutorials.org/module/3762)
