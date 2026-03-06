"""
mbn-analytics v2 설정 파일
"""
import os
import sys
from dotenv import load_dotenv

# 상위 디렉토리의 .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-4o-mini"
GPT_TEMPERATURE = 0.1
GPT_MAX_TOKENS = 800
GPT_TIMEOUT = 30
RATE_LIMIT_DELAY = 0.5

# ============================================================
# 디렉토리 구조
# ============================================================
# v2/
# ├── data/                    # CSV 데이터 파일 (여기에 새 파일 덮어쓰기)
# │   ├── chatbot.csv
# │   ├── chatsession.csv
# │   └── chatdialog.csv
# ├── output/                  # 분석 결과
# │   ├── dashboard_data.json
# │   └── last_sync.json
# └── ...

V2_DIR = os.path.dirname(os.path.abspath(__file__))

# 입력 데이터 경로 (v2/data/)
DATA_DIR = os.path.join(V2_DIR, 'data')
CHATBOT_CSV = os.path.join(DATA_DIR, 'chatbot.csv')
CHATSESSION_CSV = os.path.join(DATA_DIR, 'chatsession.csv')
CHATDIALOG_CSV = os.path.join(DATA_DIR, 'chatdialog.csv')
USER_SURVEY_CSV = os.path.join(DATA_DIR, 'anonymous_user_survey.csv')  # 실제 성별/연령대 데이터
ANONYMOUS_USER_CSV = os.path.join(DATA_DIR, 'anonymous_user.csv')  # 전체 등록 유저

# 출력 데이터 경로 (v2/output/)
OUTPUT_DIR = os.path.join(V2_DIR, 'output')
DASHBOARD_DATA_JSON = os.path.join(OUTPUT_DIR, 'dashboard_data.json')
LAST_SYNC_JSON = os.path.join(OUTPUT_DIR, 'last_sync.json')
ANALYZED_DIALOGS_JSON = os.path.join(OUTPUT_DIR, 'analyzed_dialogs.json')  # 분석된 대화 캐시

# 데이터 필터링
DATE_FILTER = '2026-01-13'  # 이 날짜 이후 데이터만 사용

# 베이커 설정
TOTAL_BAKERS = 72

# 방문 횟수 추정 설정
DEFAULT_CUTOFF_SECONDS = 1800  # 30분 (데이터 부족 시 기본값)
MIN_GAPS_FOR_PERSONAL_CUTOFF = 5  # 개인별 cutoff 계산에 필요한 최소 간격 수
CUTOFF_PERCENTILE = 95  # 백분위수 기준

# 사용자 유형 분류 기준
USER_TYPE_THRESHOLDS = {
    'fan': {'min_favorability': 4.0, 'max_risk_rate': 0.10},
    'troll': {'max_favorability': 2.5, 'min_risk_rate': 0.30}
}

# 위험도 유형
RISK_TYPES = {
    'family': '가족정보 노출',
    'personal': '개인사 노출',
    'spoiler': '방송 스포일러'
}

# 질문 유형
QUESTION_TYPES = {
    'intro': '자기소개 요청',
    'recipe': '빵/레시피 질문',
    'personal': '개인적 질문',
    'broadcast': '방송/대회 질문',
    'shop': '가게/구매 질문',
    'other': '기타'
}

# 감정 분류
SENTIMENTS = ['positive', 'neutral', 'negative']

# 목업 데이터 설정 (성별/연령대)
GENDER_RATIO = {'male': 0.45, 'female': 0.55}
AGE_WEIGHTS = {
    '10대': 0.05,
    '20대': 0.25,
    '30대': 0.30,
    '40대': 0.25,
    '50대': 0.10,
    '60대': 0.04,
    '70대이상': 0.01
}

# 베이커 4분면 분류 기준
QUADRANT_THRESHOLDS = {
    'dialog_median': None,  # 런타임에 계산
    'concentration_median': None  # 런타임에 계산
}
