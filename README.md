# 유튜브 영상 요약 및 댓글 생성기

이 프로젝트는 유튜브 영상의 내용을 자동으로 요약하고, 관련된 긍정적인 댓글을 생성하는 파이썬 스크립트입니다.

## 주요 기능

- 유튜브 영상 URL을 입력받아 자동으로 내용 요약
- 자막이 없는 경우 음성을 텍스트로 변환
- 토큰 사용량 및 비용 계산
- 한국어(80%)와 영어(20%) 비율로 100개의 긍정적인 댓글 자동 생성

## 설치 방법

1. Python 3.x가 설치되어 있는지 확인합니다.

2. 프로젝트를 클론합니다:
```bash
git clone [repository-url]
cd youtube-reply
```

3. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다:
```bash
python main.py
```

2. 프롬프트가 표시되면 유튜브 URL을 입력합니다.

3. 스크립트가 자동으로 다음 작업을 수행합니다:
   - 영상 정보 추출
   - 자막/음성 텍스트 추출
   - 내용 요약
   - 댓글 생성

## 출력 결과

스크립트는 다음 정보��� 출력합니다:
- 영상 제목
- 영상 길이
- 사용된 토큰 수
- 예상 비용 (원화)
- 요약된 내용
- 생성된 100개의 댓글

## 주의사항

- 인터넷 연결이 필요합니다.
- 자막이 없는 영상의 경우 처리 시간이 더 오래 걸릴 수 있습니다.
- 비용 계산은 예시이며, 실제 API 사용료와 다를 수 있습니다. 