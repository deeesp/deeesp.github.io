---
title:  "[Git 02] CLI 기반 버전관리"
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

# CLI(Command line Interface, 명령어)를 이용한 git 버전관리
- 앞서 [GIT 1 Overview](https://deeesp.github.io/git/GIT-1-Overview/)에서 git에 대해 전체적인 감을 잡았다. 이제 실제 git을 이용해 버전관리를 해보자!!
- Git을 제어하는 데에 있어서, Source Tree, Github Desktop, TortoiseGit, CLI 등 다양한 방법이 있다.
- 이러한 방법 중에, 명령어를 기반으로 한 CLI를 이용하여 버전관리를 해보자.

## [1] Why CLI for Git rather than other GUI programs

1. 사실상, CLI를 제외한 GUI기반의 방법들도 내부적으로는 오리지널 git, CLI를 기반으로 한다.
2. CLI 방법은 복잡하지만 익숙해지기만 하면 간편하게 git을 다룰 수 있고, 특히, GUI 없는 환경(e.g. 서버)에서도 제어가 가능하다.
3. 또한, 명령어의 특성상 동시다발적으로 처리해야 할 작업을 명령어를 통해 한번에 처리가 가능하여 개발자들에게 인기가 많다.


## [2] Mac 설치 방법 (초간단)

1. git이 이미 설치되었는지 확인해보자.
    - `cmd + space` 스포트라이트 검색을 통해 `terminal`을 켠 후에 `git` 명령어를 쳤을 때 아래 이미지와 같이 git에 대한 usage 설명이 나오면 이미 git이 설치 되어 있는 상태이다.  
    
    ![git-cli](/images/git-cli.png){: width="70%" height="70%"}  
    
2. 하지만 그렇지 않다면, [https://git-scm.com/](https://git-scm.com/) 에 접속하여 latest source release download를 해준다.
3. 그리고 설치가 되었다면! 다시 1번 과정을 반복하여 설치되었는지 확인한다.

## [3] git에게 특정 디렉토리 관리시키기  
  
![git-dir-managing](/images/git-cli-dir.png){: width="70%" height="70%"}  
  
1. `/desktop` 디렉토리에 앞으로 작업할 `hello-git-cli`라는 디렉토리를 만들어 보았다. 그리고 `cd hello-git-cli` 명령어로 그 디렉토리로 들어가보자. 그럼 현재 디렉토리 상태는 `/Users/mac/desktop/hello-git-cli`이고, 내부에 아무것도 없는 것을 볼 수 있다.
2. 이제, git에게 `git init .` 명령어를 통해 이 디렉토리 버전관리를 하라고 명령을 해보자.
    - 1) `git`은 git 명령어라는 것을 말하고 `init`은 초기화<sup>Initialization</sup>의 줄임말이다. 뒤에 있는 ` .`은 현재 디렉토리 버전관리를 시키는 것이다.
    - 2) `ls -al`을 통해 내부 디렉토리를 보면 `.git`이 생긴 것을 볼 수 있다. 이 내부를 보면, 부모 디렉토리이자 우리가 작업할 디렉토리인 `hello-git-cli`에서 생성되는 변화들이 버전으로 만들어져 이 `.git`에 저장된다.
    - 3) 즉, `.git`에는 앞으로 `hello-git-cli`의 역사가 기록이 되는 것이다.

## [4] 버전 만들기

### ★ 기억해야 할 것 ★
![git-flow](/images/git-flow.png)
1. **Working Tree**: 파일을 만들고 수정하는, 버전으로 만들어지기 전 단계
2. **Staging Area**: 몇 개의 파일을 하나의 버전으로 만들지, 버전으로 만들고자 하는 파일들을 Staging Area에 올린다.
3. **Repository**: 버전이 저장되어 있는 곳. Staging Area에 있는 파일들을 하나의 버전으로 만들어 `.git`에 저장한다.

### 실습 해보기

1. `hello-git-cli`에 새로운 파일을 만들어보자.  
  
    ![new-text](/images/git-new-txt.png)
    - `nano hello1.txt`로 텍스트 파일을 만들어주자. 
        - `nano` 명령어는 CLI 환경에서 간단하게 텍스트 파일이나 프로그램 파일을 작성하고 편집하는 명령어이다.
    - GNU nano 창이 뜨면, 아무 내용이나 써주고 `ctrl+x`<sup>exit</sup>과 `y`<sup>yes</sup>, `enter`<sup>file name 확인</sup>를 차례대로 눌러주자.
    - 파일이 생성이 되었으면, `cat hello1.txt`로 내용을 출력해보자.
        ![cat](/images/git-cat.png)
        - `cat`은 concatenate의 줄임말로, 여러 개 파일의 받아서 파일의 내용을 출력하는 명령어이다. 간단하게 파일의 내용을 보기 위한 용도.

2. `git status` 명령어로 현재 상태를 보자.  
  
    ![status](/images/git-status.png)
    - `git status`: working tree status - git의 상태를 보기 위해 사용하는 명령어로 자주 사용하게 된다.
    - **No commits yet**: `commit` == 버전, 아직 버전이 없는 상태이다.
    - **Untracked files**: `hello1.txt`가 추적되지 않고 있음을 보여주고 있다. git은 특정 파일을 버전관리 하겠다고 명시적으로 한번 지정해주지 않으면 없는 셈 치기 때문에 `hello1.txt`가 버전관리 되고 있지 않다는 것이다.

3. `git add xx.xx` 명령어로 **Working Tree**에 있는 수정사항을 **Staging Area**에 올려 버전으로 만들어주자.
    - `git add`: add to staging area
    - `hello1.txt`를 버전으로 만들 것이니 `git add hello1.txt`명령어로 **Working Tree**에서 **Staging Area**로 올리자.
    - 다시 `git status`로 확인해보면, **Untracked files**에서 **Changes to be commited**로 바뀌어 버전이 될 파일을 보여준다.
    
4. `git commit`으로 버전을 만들어보자.
    - `git commit`: create version - 새로운 버전을 만들어 준다.
    - `git commit`만을 치면 아래와 같은 에디터가 나와 commit message를 작성해줄 수 있다.
        ![git-commit](/images/git-commit.png)
    - 하지만 이 과정이 귀찮으므로, `-m`을 통해 한꺼번에 메시지를 작성해줄 수 있다. `git commit -m "message"`
        ![git-commit](/images/git-commit-m.png)
    - `git status`로 다시 현재 상태를 관찰해보자.
        - **nothing to commit**: 버전으로 만들 것이 없다.
        - **working tree clean**: 버전이 되지 않은 수정사항이 없다.
5. `git log`를 통해 버전이 잘 만들어 졌는지 확인해 보자.
    - `git log`: show version - 일종의 버전의 역사를 볼 수 있는 명령어이다.
    - 나갈 때는 `q`를 누르면 된다.

6. 위의 과정을 다시 한번 반복해서 버전이 바뀌는지 해보자.
    ![commit2](/images/git-commit-2.png)
    
    
## [5] 여러 개 파일을 하나의 버전으로 만들기
- 보통 하나의 작업에는 앞서 봤던 하나의 버전에 하나의 파일의 수정이 있는 것보단, 여러 개의 파일의 수정사항을 담고 있는 경우가 많다.
- [4]를 응용하여 여러 개의 파일의 수정사항을 하나의 버전으로 만들어 보자.

### 실습해보기
1. `nano hello1.txt`에 세 번째 내용을 추가한다.
2. `nano hello2.txt`로 새로운 파일을 추가한다.
3. `git status`를 해보면, 두 파일 모두 **Staging Area**에 있지 않고, **Working Tree**에 있을 것이다.  
    ![multiple-file](/images/git-multiple-status.png)
    - 하지만, `hello1.txt`는 이미 1회 이상 버전관리를 해왔기 때문에 git은 이를 기억하고 관리하고 있다.
    - `hello2.txt`은 버전관리를 한 적이 없기 때문에 git은 이를 없는 셈 친다.
    - 이 상태로 백업을 한다면, `hello2.txt`는 백업과 협업을 할 수 가 없다. 자동으로 tracking해주지 않기 때문이다.
    
4. `git add hello1.txt`, `git add hello2.txt`로 **Staging Area**에 올려준다.  
    ![multiple-add](/images/git-multiple-add.png)
    - 또는 **Working Tree**에 있는 모든 파일을 올리고 싶으면 `git add *`명령어를 사용할 수 있다.
    - 하지만 `git add *` 명령어는 **.gitignore** 문제가 있어 `git add .`을 사용해 주자.
    
5. `git commit -m`으로 버전을 생성해준다.  
    ![multiple-add](/images/git-multiple-commit.png)
    
6. `git log`로 생성된 버전들을 살펴보자.
    - 어떤 버전이 생성되어 있는 지는 나와있지만, 세부적으로 어떤 파일들이 연루되어 있는지에 대한 내용이 없다. 아래 [Stackoverflow](https://stackoverflow.com/questions/1230084/how-to-have-git-log-show-filenames-like-svn-log-v)에서 참조한 `git log --stat`명령어로 확인해보자.  
    ![log-detail](/images/git-log-detail.png)
    
### Optional
- `add`와 `commit`을 동시에 해주고 싶을 때는?
    - `git commit -am` 명령어를 사용해 준다.  
    ![add-commit](/images/git-add-commit.png)

## Reference
- [생활코딩 지옥에서 온 GIT 2](https://opentutorials.org/module/3762)
