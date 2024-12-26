import os
from dotenv import load_dotenv
import yt_dlp
import whisper
import requests
from tqdm import tqdm
import re
import nltk
import random
import time
import json
import warnings

# Whisper FP16 경고 메시지 필터링
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# 환경 변수 로드
load_dotenv()

# Gemini API 설정
GEMINI_API_KEY = "AIzaSyCGqESejboX1ek0OEsXfvOAiWpql9CLwV4"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

class YouTubeSummarizer:
    def __init__(self):
        # Whisper 모델 초기화 시 FP32 명시적 지정
        self.whisper_model = whisper.load_model("base", device="cpu")
        self.usd_to_krw = 1450
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['ko', 'en'],
            'quiet': True,
            'no_warnings': True
        }
        
    def get_video_info(self, url):
        """유튜브 영상 정보 추출"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # 자막 정보 확인
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                # 한국어 또는 영어 자막 찾기
                captions = None
                for lang in ['ko', 'en']:
                    if lang in subtitles:
                        captions = subtitles[lang]
                        break
                    elif lang in automatic_captions:
                        captions = automatic_captions[lang]
                        break
                
                print(f"제목: {info['title']}")
                print(f"길이: {info['duration']}초")
                
                return {
                    'title': info['title'],
                    'length': info['duration'],
                    'captions': captions,
                    'url': url,
                    'description': info.get('description', '')
                }
        except Exception as e:
            print(f"영상 정보 추출 중 오류 발생: {str(e)}")
            return None

    def extract_text(self, video_info):
        """자막 또는 음성에서 텍스트 ��출"""
        try:
            # 자막이 있는 경우 자막 사용
            if video_info.get('captions'):
                print("자막을 사용하여 텍스트 추출 중...")
                captions = video_info['captions']
                
                # 자막 URL에서 텍스트 추출
                if isinstance(captions, list) and len(captions) > 0:
                    caption_url = None
                    # vtt 또는 srv3 형식의 자막 찾기
                    for fmt in captions:
                        if fmt.get('ext') in ['vtt', 'srv3']:
                            caption_url = fmt.get('url')
                            break
                    
                    if caption_url:
                        response = requests.get(caption_url)
                        if response.status_code == 200:
                            # VTT 형식 자막 파싱
                            lines = response.text.split('\n')
                            text_lines = []
                            for line in lines:
                                # 타임스탬프와 숫자 라인 제외
                                if not re.match(r'^\d|^\[|^WEBVTT|^NOTE|^Language:|^-->|^$', line):
                                    text_lines.append(line.strip())
                            return ' '.join(text_lines)
            
            # 자막이 없는 경우 음성 인식 사용
            print("자막이 없어 음성 인식을 사용합니다...")
            # 오디오 다운로드 옵션 설정
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': 'temp_audio',
                'quiet': True,
                'no_warnings': True
            }
            
            # 오디오 다운로드
            print("오디오 다운로드 중...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_info['url']])
            
            # Whisper를 사용하여 음성을 텍스트로 변환
            print("음성을 텍스트로 변환 중...")
            result = self.whisper_model.transcribe("temp_audio.wav")
            
            # 임시 파일 제거
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            
            return result['text']
            
        except Exception as e:
            print(f"텍스트 추출 중 오류 발생: {str(e)}")
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            return "영상의 내용을 이해하기 어려웠습니다."

    def _call_gemini_api(self, prompt):
        """Gemini API 호출"""
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "contents": [{
                "parts":[{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 2048
            }
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
        return None

    def summarize_text(self, text):
        """Gemini를 사용한 텍스트 요약"""
        if not text:
            return {'summary': '영상의 내용을 이해하기 어려웠습니다.', 'tokens': 0}
            
        try:
            prompt = f"""
다음 텍스트를 300자 이내로 요약해주세요. 핵심적인 내용만 간단명료하게 작성해주세요:

{text}

요약:"""
            
            summary = self._call_gemini_api(prompt)
            if not summary:
                return {'summary': text[:500], 'tokens': len(text.split())}
            
            # 토큰 수 추정 (실제 Gemini API는 토큰 수를 반환하지 않음)
            tokens = len(text.split()) // 3
            
            return {
                'summary': summary,
                'tokens': tokens
            }
        except Exception as e:
            print(f"요약 중 오류 발생: {str(e)}")
            return {'summary': text[:500], 'tokens': len(text.split())}

    def generate_comments(self, summary, video_info):
        """Gemini를 사용한 댓글 생성"""
        if not summary:
            print("요약문이 없어 댓글을 생성할 수 없습니다.")
            return ["좋은 영상입니다"] * 100

        try:
            # 통합된 댓글 생성 프롬프트
            prompt = f"""
다음 유튜브 영상에 대한 댓글 100개를 생성해주세요.

영상 제목: {video_info['title']}
영상 내용 요약: {summary}

조건:
1. 댓글 작성 비율:
   - 한국인 댓글 (90개):
     * 10대 학생 (20개)
     * 20대 초반 대학생 (20개)
     * 20대 후반 직장인 (20개)
     * 30-40대 부모 (20개)
     * 50대 이상 (10개)
   - 외국인 댓글 (10개):
     * 영어 댓글 (6개):
       - 미국인 (3개, 캐주얼한 톤)
       - 영국인 (2개, 격식있는 톤)
       - 동남아시아인 (1개, 친근한 톤)
     * 한국어 댓글 (4개):
       - 일본인 (2개, 어색한 한국어)
       - 중국인 (2개, 어색한 한국어)
   
2. 문체를 다양하게 활용하세요:
   - 한국어 댓글:
     * 반말과 존댓말을 섞어서 사용
     * 마침표 사용을 랜덤하게 (마침표 있음/없음, ㅋㅋ/ㅎㅎ로 끝남)
     * 이모티콘과 한글 이모티콘 자연스럽게 사용
     * 감탄사나 구어체 표현 포함
   - 영어 댓글:
     * 각 국가별 특징적인 표현 사용 (미국식/영국식)
     * 이모지 자연스럽게 활용
     * 구어체와 축약어 활용
   - 외국인의 한국어 댓글:
     * 약간의 문법적 오류 포함
     * 해당 국가 특유의 말투나 표현 사용
     * 모국어 단어 1-2개 자연럽게 섞기
   
3. 내용 구성:
   - 영상의 구체적인 내용을 언급
   - 개인적인 경험이나 감상 추가
   - 다른 시청자와의 소통하는 듯한 멘트 포함
   - 크리에이터를 향한 응원이나 피드백
   
4. 형식:
   - 각 댓글은 새로운 줄로 구분
   - 댓글 앞에 번호 붙이지 않기
   - 자연스러운 길이로 작성 (너무 길거나 짧지 않게)

댓글:"""

            # 댓글 생성
            comments = self._call_gemini_api(prompt)
            
            if not comments:
                print("Gemini API가 응답하지 않았습니다.")
                return ["좋은 영상입니다"] * 100
            
            # 댓글 분리 및 정리
            comments = [c.strip() for c in comments.split('\n') if c.strip()]
            
            if len(comments) < 10:
                print(f"생성된 댓글이 너무 적습니다: {len(comments)}개")
                return ["좋은 영상입니다"] * 100
            
            # 댓글 선택 및 섞기
            random.shuffle(comments)
            
            return comments[:100]
            
        except Exception as e:
            print(f"댓글 생성 중 상세 오류 발생: {str(e)}")
            if hasattr(e, 'response'):
                print(f"API 응답: {e.response.text if hasattr(e.response, 'text') else '없음'}")
            return ["좋은 영상입니다"] * 100

    def calculate_cost(self, tokens):
        """비용 계산 (원화)"""
        # 예시 비용: 1000 토큰당 0.1 USD
        usd_cost = (tokens / 1000) * 0.1
        krw_cost = usd_cost * self.usd_to_krw  # 고정 환율 사용
        return round(krw_cost, 2)

    def _split_text(self, text, max_length=500):
        """텍스트를 청크로 분할"""
        if not text:
            return []
        
        # 문장 단위로 분할
        sentences = re.split('[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _extract_keywords(self, text):
        """텍스트에서 키워드 추출"""
        if not text:
            return []
            
        # 기본적인 불용어 설정
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', '이', '그', '저', '것', '수', '등', '및'])
        
        # 텍스트를 단어로 분리
        words = text.split()
        
        # 불용어 제거 및 길이 제한
        keywords = [word for word in words if len(word) > 1 and word.lower() not in stopwords]
        
        # 중복 제거 및 상위 20개 선택
        unique_keywords = list(set(keywords))
        return unique_keywords[:20] if unique_keywords else ["좋은 내용"]

def format_length(seconds):
    """영상 길이를 분:초 형식으로 변환"""
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02d}"

def main():
    url = input("YouTube URL을 입력하세요: ")
    
    # 전체 실행 시간 측정 시작
    total_start_time = time.time()
    
    summarizer = YouTubeSummarizer()
    
    print("\n1. 영상 정보 가져오는 중...")
    step1_start = time.time()
    video_info = summarizer.get_video_info(url)
    step1_time = time.time() - step1_start
    
    if not video_info:
        print("영상 정보 가져올 수 없습니다.")
        return
    
    print("\n2. 텍스트 추출 중...")
    step2_start = time.time()
    text = summarizer.extract_text(video_info)
    step2_time = time.time() - step2_start
    
    print("\n3. 텍스트 요약 중...")
    step3_start = time.time()
    summary_info = summarizer.summarize_text(text)
    step3_time = time.time() - step3_start
    
    print("\n4. 댓글 생성 중...")
    step4_start = time.time()
    comments = summarizer.generate_comments(summary_info['summary'], video_info)
    step4_time = time.time() - step4_start
    
    # 전체 실행 시간 계산
    total_time = time.time() - total_start_time
    
    cost = summarizer.calculate_cost(summary_info['tokens'])
    
    # 결과 출력
    print("\n=== 결과 ===")
    print(f"제목: {video_info['title']}")
    print(f"길이: {format_length(video_info['length'])}")
    print(f"토큰 수: {summary_info['tokens']}")
    print(f"비용: ₩{cost}")
    print("\n실행 시간:")
    print(f"1. 영상 정보 추출: {step1_time:.1f}초")
    print(f"2. 텍스트 추출: {step2_time:.1f}초")
    print(f"3. 텍스트 요약: {step3_time:.1f}초")
    print(f"4. 댓글 생성: {step4_time:.1f}초")
    print(f"총 실행 시간: {total_time:.1f}초")
    print("\n요약:")
    print(summary_info['summary'])
    print("\n생성된 댓글 (총 100개):")
    for i, comment in enumerate(comments, 1):
        print(f"{i}. {comment}")

if __name__ == "__main__":
    main() 