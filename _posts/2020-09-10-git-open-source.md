---
title:  "[Git] 오픈소스 (Open-source) 프로젝트 기여하기"
excerpt: "Repo 최신화, Branch, Pull Request"
categories:
  - git
  
tags:
  - 버전 관리
  - 깃
  - 깃허브
  - git
  - github
  - CLI
  - 협업
  
last_modified_at: 2020-09-10-22:00:00

toc: true
toc_sticky: true
---

## git으로 오픈소스 프로젝트 협업하기
뉴비, 취업 전 실전 경력이 없는 컴돌이들은 오픈소스를 통해 기여하는 것도 방법이다. 하지만, git 초보들은 오픈소트 프로젝트를 하면서 PR<sup>Pull Request</sup>를 날리는게 처음엔 상당히 헷갈린다. 이번에 코드를 통한 기여는 아니지만 오픈소스 번역 프로젝트에 참여하면서 알아간 내용을 써보았다.

### [1] 포크<sup>Fork</sup> 따온 저장소<sup>Repository</sup> 최신화
포크 떠오자마자 라면 다른 기여자의 코드와 버전 충돌할 리는 없지만, 내 저장소로 포크를 떠온 뒤 시간이 흐르고 나서 기여를 한다면 다른 사람이 수정한 코드와 충돌이 일어날 수 가 있다. 따라서 저장소를 최신화 해준 상태에서 진행해준다. (cherrypick 방법이 있다고는 하는데.. 아직 모름..)

<center>
<img src="/images/git-conflict.png" height="400px" /><br>
<b>그림. 1</b>: edit conflict
</center>

1. `remote`
- github과 같은 원격저장소와 연결하여 관리한다.
- origin(내가 포크 따온 저장소)과 upsteram (원본 소스코드가 있는 곳) 연결하자.

    ```bash
    git remote -v
    git remote add origin https://github.com/deeesp/pytorch-Deep-Learning.git
    git remote add upstream https://github.com/Atcold/pytorch-Deep-Learning.git
    ```

2. `fetch` & `merge` 또는 `pull`
- upstream으로부터 가져와서 실제 최신화 하는 과정

    ```bash
    git fetch upstream
    git merge upstream/master

    git pull upstream master
    ```

3. `push`
- 내 저장소로 push해준다.

    ```bash
    git push origin master
    ```

### [2] 브랜치 만들어서 수정

1. `checkout`

    ```bash
    git checkout -b kr-11-1
    ```

2. 원하는 파일 `add` 및 `commit`

    ```bash
    git add *
    git commit -m "[KR] Translation of 11-1.md"
    ```

3. 내 저장소에 `push`

    ```bash
    git push origin kr-11-1
    ```

4. github에서 `pull request`
