#!/usr/bin/env python3
"""
mbn-analytics v2 데이터 처리 스크립트

주요 기능:
1. CSV 파일 로드 및 날짜 필터링
2. 성별/연령대 목업 데이터 생성
3. 실시간(오늘)/누적 통계 분리
4. GPT 하이브리드 분석 실행
5. 베이커 중심 호감도 분석 (4분면 분류)
6. 위험도 대응 분석
7. 사용자 분포 분석
8. 고객 여정 분석
9. dashboard_data.json 생성

사용법:
    python data_processor.py full                     # 전체 재분석
    python data_processor.py incremental              # 증분 분석 (새 데이터만)
    python data_processor.py full 2026-01-13 2026-01-19  # 날짜 범위 전체 분석
"""
import json
import csv
import re
import sys
import os
import random
import hashlib
from collections import defaultdict
from datetime import datetime, date, timezone, timedelta
import numpy as np

KST = timezone(timedelta(hours=9))
from hybrid_analyzer import HybridAnalyzer
from config import (
    DATE_FILTER, TOTAL_BAKERS, DEFAULT_CUTOFF_SECONDS,
    MIN_GAPS_FOR_PERSONAL_CUTOFF, CUTOFF_PERCENTILE,
    USER_TYPE_THRESHOLDS, RISK_TYPES, QUESTION_TYPES, SENTIMENTS,
    CHATBOT_CSV, CHATSESSION_CSV, CHATDIALOG_CSV, USER_SURVEY_CSV, ANONYMOUS_USER_CSV,
    GENDER_RATIO, AGE_WEIGHTS, OUTPUT_DIR, DASHBOARD_DATA_JSON
)
from sync_manager import sync_manager

# 진행률 파일 경로
PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress.json')


def update_progress(current, total, stage="GPT 분석"):
    """진행률을 파일에 저장"""
    progress = {
        'current': current,
        'total': total,
        'stage': stage,
        'percent': round(current / total * 100, 1) if total > 0 else 0
    }
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False)
    except:
        pass


def clear_progress():
    """진행률 파일 삭제"""
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
    except:
        pass


# ============================================
# 1. 사용자 인구통계 데이터 (실제 설문 + 목업)
# ============================================

# 설문 데이터 연령대 매핑
AGE_MAPPING = {
    'TEENS': '10대',
    'TWENTIES': '20대',
    'THIRTIES': '30대',
    'FORTIES': '40대',
    'FIFTIES': '50대',
    'SIXTIES': '60대',
    'SEVENTIES': '70대이상'
}

# 설문 데이터 성별 매핑 (한글)
GENDER_MAPPING = {
    'MALE': '남자',
    'FEMALE': '여자'
}


def load_user_survey() -> dict:
    """
    anonymous_user_survey.csv에서 실제 성별/연령대 데이터 로드
    
    CSV 구조 (헤더 없음):
    0: survey_id
    1: user_id
    2: type (INITIAL)
    3: gender (MALE, FEMALE)
    4: age (TEENS, TWENTIES, THIRTIES, FORTIES, FIFTIES 등)
    5: extra_data
    6: created_at
    7: updated_at
    """
    survey_data = {}
    
    if not os.path.exists(USER_SURVEY_CSV):
        print(f"   [경고] 설문 데이터 파일 없음: {USER_SURVEY_CSV}")
        return survey_data
    
    with open(USER_SURVEY_CSV, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 5:
                user_id = row[1]
                gender_raw = row[3]
                age_raw = row[4]
                
                gender = GENDER_MAPPING.get(gender_raw, '모름')
                age = AGE_MAPPING.get(age_raw, '모름')
                
                survey_data[user_id] = {
                    'gender': gender,
                    'age': age
                }
    
    return survey_data


def generate_user_demographics(user_ids: list) -> dict:
    """
    사용자별 성별/연령대 데이터 생성
    - 실제 설문 데이터가 있으면 사용
    - 없으면 '모름'으로 처리 (목업 데이터 사용하지 않음)
    """
    # 실제 설문 데이터 로드
    survey_data = load_user_survey()
    survey_count = 0
    unknown_count = 0
    
    demographics = {}
    
    for user_id in user_ids:
        # 실제 설문 데이터가 있으면 사용
        if user_id in survey_data:
            demographics[user_id] = survey_data[user_id]
            survey_count += 1
        else:
            # 설문 데이터 없으면 '모름'으로 처리
            demographics[user_id] = {
                'gender': '모름',
                'age': '모름'
            }
            unknown_count += 1
    
    print(f"   실제 설문 데이터: {survey_count}명, 모름: {unknown_count}명")
    
    return demographics


# ============================================
# 2. CSV 파일 로드
# ============================================
from datetime import timedelta

def utc_to_kst(dt_str):
    """UTC 시간 문자열을 KST(+9시간)로 변환"""
    if not dt_str or len(dt_str) < 19:
        return dt_str
    try:
        dt = datetime.strptime(dt_str[:19], '%Y-%m-%d %H:%M:%S')
        kst = dt + timedelta(hours=9)
        return kst.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return dt_str


def load_anonymous_users():
    """anonymous_user.csv에서 전체 등록 유저 로드
    Returns: dict of {uid: {'created_at': str, ...}}
    """
    users = {}
    if not os.path.exists(ANONYMOUS_USER_CSV):
        return users
    with open(ANONYMOUS_USER_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and len(row) >= 3:
                uid = row[0].strip('"')
                created_at = row[2].strip('"') if len(row) > 2 else ''
                users[uid] = {'created_at': utc_to_kst(created_at) if created_at else ''}
    return users


def load_chatbots():
    """chatbot.csv 로드"""
    chatbots = {}
    with open(CHATBOT_CSV, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                chatbots[row[0]] = row[1]
    return chatbots


def load_sessions():
    """chatsession.csv 로드"""
    sessions = []
    with open(CHATSESSION_CSV, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 5 and row[0].startswith('S'):
                sessions.append({
                    'session_id': row[0],
                    'user_id': row[1],
                    'chatbot_id': row[2],
                    'title': row[3],
                    'created_at': utc_to_kst(row[4])  # UTC → KST 변환
                })
    return sessions


def load_dialogs():
    """chatdialog.csv 로드"""
    dialogs = []
    date_pattern = re.compile(r'(202[56]-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    
    with open(CHATDIALOG_CSV, 'r', encoding='utf-8') as f:
        for line in f:
            row = list(csv.reader([line]))[0]
            if len(row) >= 6:
                dates = date_pattern.findall(line)
                created_at = utc_to_kst(dates[0]) if dates else ''  # UTC → KST 변환
                
                dialogs.append({
                    'dialog_id': row[0],
                    'session_id': row[1],
                    'user_id': row[2],
                    'chatbot_id': row[3],
                    'question': row[4],
                    'answer': row[5],
                    'created_at': created_at
                })
    return dialogs


# ============================================
# 3. 방문 횟수 계산
# ============================================
def parse_datetime(dt_str):
    """날짜 문자열을 datetime 객체로 변환"""
    try:
        return datetime.strptime(dt_str[:19], '%Y-%m-%d %H:%M:%S')
    except:
        return None


def calculate_user_visits(user_dialogs):
    """사용자별 방문 횟수와 개인별 cutoff를 계산"""
    timestamps = []
    for d in user_dialogs:
        dt = parse_datetime(d.get('created_at', ''))
        if dt:
            timestamps.append(dt)
    
    timestamps.sort()
    
    if len(timestamps) < 2:
        return {'visit_count': 1, 'cutoff_seconds': DEFAULT_CUTOFF_SECONDS, 'gaps': []}
    
    gaps = []
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i-1]).total_seconds()
        gaps.append(gap)
    
    if len(gaps) >= MIN_GAPS_FOR_PERSONAL_CUTOFF:
        cutoff = np.percentile(gaps, CUTOFF_PERCENTILE)
    else:
        cutoff = DEFAULT_CUTOFF_SECONDS
    
    visit_count = 1 + sum(1 for g in gaps if g > cutoff)
    
    return {
        'visit_count': visit_count,
        'cutoff_seconds': cutoff,
        'gaps': gaps
    }


# ============================================
# 4. 사용자 유형 분류
# ============================================
def classify_user_type(avg_favorability, risk_rate):
    """사용자 유형 분류 (팬/중립/악플러)"""
    fan_thresh = USER_TYPE_THRESHOLDS['fan']
    troll_thresh = USER_TYPE_THRESHOLDS['troll']
    
    if avg_favorability >= fan_thresh['min_favorability'] and risk_rate < fan_thresh['max_risk_rate']:
        return 'fan'
    elif avg_favorability < troll_thresh['max_favorability'] or risk_rate >= troll_thresh['min_risk_rate']:
        return 'troll'
    else:
        return 'neutral'


# ============================================
# 5. 베이커 4분면 분류
# ============================================
def classify_baker_quadrant(dialog_count, unique_users, median_dialogs, median_users):
    """
    베이커 4분면 분류 (X축: 사용자수, Y축: 대화수 기준)
    - 대중인기형: 사용자 多, 대화 多 (오른쪽 상단) - 많은 사람이 많이 대화
    - 소수광팬형: 사용자 少, 대화 多 (왼쪽 상단) - 소수가 집중적으로 대화
    - 넓고얕은형: 사용자 多, 대화 少 (오른쪽 하단) - 많은 사람이 조금씩 대화
    - 무관심형: 사용자 少, 대화 少 (왼쪽 하단) - 관심 적음
    """
    high_dialog = dialog_count >= median_dialogs
    high_users = unique_users >= median_users
    
    if high_users and high_dialog:
        return 'popular'  # 대중인기형 (오른쪽 상단)
    elif not high_users and high_dialog:
        return 'minority_fan'  # 소수광팬형 (왼쪽 상단)
    elif high_users and not high_dialog:
        return 'shallow'  # 넓고얕은형 (오른쪽 하단)
    else:
        return 'ignored'  # 무관심형 (왼쪽 하단)


# ============================================
# 6. 필터링 함수
# ============================================
def get_date_range_from_data(dialogs):
    """데이터에서 날짜 범위 추출"""
    dates = []
    for d in dialogs:
        created_at = d.get('created_at', '')
        if created_at:
            dates.append(created_at[:10])
    
    if dates:
        return min(dates), max(dates)
    return DATE_FILTER, DATE_FILTER


def filter_dialogs_by_date(dialogs, start_date, end_date):
    """날짜 범위로 대화 필터링"""
    filtered = []
    for d in dialogs:
        created_at = d.get('created_at', '')
        if created_at:
            date_str = created_at[:10]
            if start_date <= date_str <= end_date:
                filtered.append(d)
    return filtered


def filter_dialogs_by_today(dialogs):
    """오늘 날짜 대화만 필터링"""
    today = date.today().isoformat()
    return [d for d in dialogs if d.get('created_at', '').startswith(today)]


def filter_existing_data(start_date, end_date):
    """
    기존 분석 데이터를 날짜 범위로 필터링하여 새로운 dashboard_data 생성
    GPT 재분석 없이 집계만 다시 계산
    CSV 파일 없이 analyzed_dialogs.json만으로도 동작
    """
    from config import ANALYZED_DIALOGS_JSON
    
    # DATE_FILTER 이전 날짜는 강제로 DATE_FILTER로 변경
    if start_date < DATE_FILTER:
        start_date = DATE_FILTER
    
    # 유효성 검사
    if start_date > end_date:
        return None
    
    # 기존 분석 데이터 로드
    if not os.path.exists(ANALYZED_DIALOGS_JSON):
        return None
    
    with open(ANALYZED_DIALOGS_JSON, 'r', encoding='utf-8') as f:
        analyzed_dialogs_raw = json.load(f)
    
    # analyzed_dialogs.json에서 직접 대화 데이터 추출 (CSV 불필요)
    # 날짜 필터링 적용
    filtered_dialogs = []
    analyzed_dialogs = {}
    
    for d in analyzed_dialogs_raw:
        dialog_id = d.get('dialog_id', '')
        session_id = d.get('session_id', '')
        
        # 유효한 데이터만 처리 (dialog_id가 D로 시작하고 session_id가 있어야 함)
        if not dialog_id or not dialog_id.startswith('D') or not session_id:
            continue
        
        # analyzed_dialogs.json에 저장된 시간은 이미 KST 변환됨 (load_dialogs에서 변환 후 저장)
        # 중복 변환 방지를 위해 그대로 사용
        created_at = d.get('created_at', '')
        date_str = created_at[:10] if created_at else ''
        
        # DATE_FILTER 이전 데이터 제외
        if date_str < DATE_FILTER:
            continue
        
        # 사용자 지정 날짜 범위로 필터링
        if start_date <= date_str <= end_date:
            filtered_dialogs.append(d)
            analyzed_dialogs[dialog_id] = d
    
    if not filtered_dialogs:
        return None
    
    # chatbot 이름 매핑 (CSV 없으면 analyzed_dialogs에서 추출)
    try:
        chatbot_name_map = load_chatbots()
    except:
        # CSV 없으면 빈 맵 사용 (chatbot_id를 이름으로 사용)
        chatbot_name_map = {}
    
    # 분석 데이터도 필터링
    filtered_analyzed = analyzed_dialogs
    
    # 인덱싱 (analyzed_dialogs.json에 user_id, chatbot_id가 이미 있음)
    dialogs_by_session = defaultdict(list)
    dialogs_by_user = defaultdict(list)
    dialogs_by_chatbot = defaultdict(list)
    session_user_map = {}
    session_chatbot_map = {}
    
    for d in filtered_dialogs:
        session_id = d.get('session_id', '')
        user_id = d.get('user_id', '')
        chatbot_id = d.get('chatbot_id', '')
        
        dialogs_by_session[session_id].append(d)
        
        if user_id:
            dialogs_by_user[user_id].append(d)
            session_user_map[session_id] = user_id
        if chatbot_id:
            dialogs_by_chatbot[chatbot_id].append(d)
            session_chatbot_map[session_id] = chatbot_id
    
    user_demographics = {}
    
    # 데이터의 마지막 날짜/시각 찾기
    if filtered_dialogs:
        last_dialog = max(filtered_dialogs, key=lambda x: x.get('created_at', ''))
        last_datetime = last_dialog.get('created_at', '')
        last_date = last_datetime[:10] if last_datetime else ''
        last_time = last_datetime[11:16] if len(last_datetime) >= 16 else ''  # HH:MM
    else:
        last_date = ''
        last_time = ''
        last_datetime = ''
    
    # 마지막 날짜의 대화만 실시간 현황으로 사용
    realtime_dialogs = [d for d in filtered_dialogs if d.get('created_at', '')[:10] == last_date]
    realtime_session_ids = set(d['session_id'] for d in realtime_dialogs)
    realtime_user_ids = set(session_user_map.get(sid) for sid in realtime_session_ids if session_user_map.get(sid))
    
    # 전체 사용자
    all_user_ids = set(dialogs_by_user.keys())
    
    # user_id를 dialog에 추가
    for d in filtered_dialogs:
        d['user_id'] = session_user_map.get(d['session_id'], '')
    for d in realtime_dialogs:
        d['user_id'] = session_user_map.get(d['session_id'], '')
    
    # 실시간/누적 통계 계산
    realtime_stats = calculate_basic_stats(realtime_dialogs, user_demographics)
    realtime_stats['hourly_stats'] = calculate_hourly_stats(realtime_dialogs)
    realtime_stats['hourly_users_stats'] = calculate_hourly_users_stats(realtime_dialogs)
    realtime_stats['last_date'] = last_date  # YYYY-MM-DD
    realtime_stats['last_time'] = last_time  # HH:MM

    cumulative_stats = calculate_basic_stats(filtered_dialogs, user_demographics)
    cumulative_stats['daily_stats'] = calculate_daily_stats(filtered_dialogs)
    cumulative_stats['daily_users_stats'] = calculate_daily_users_stats(filtered_dialogs)

    # 전체 현황 overview 추가 데이터 계산
    experience_user_ids = set(d.get('user_id', '') for d in filtered_dialogs if d.get('user_id'))

    # 등록 유저: anonymous_user.csv (전체 등록) > chatsession (세션 생성) 순으로 우선
    anonymous_users = load_anonymous_users()
    if anonymous_users:
        # 날짜 범위 내 등록 유저
        all_registered_user_ids = set(
            uid for uid, info in anonymous_users.items()
            if info.get('created_at', '')[:10] <= end_date
        )
    else:
        # anonymous_user.csv 없으면 chatsession 기반
        try:
            sessions = load_sessions()
            filtered_sessions = [s for s in sessions if start_date <= s['created_at'][:10] <= end_date]
            all_registered_user_ids = set(s['user_id'] for s in filtered_sessions if s.get('user_id'))
        except:
            all_registered_user_ids = experience_user_ids

    # 세션 생성 유저 (대화 없는 유저 포함, chatsession 기반)
    try:
        if not anonymous_users:
            sessions = load_sessions()
        else:
            sessions = load_sessions()
        filtered_sessions = [s for s in sessions if start_date <= s['created_at'][:10] <= end_date]
        all_session_user_ids = set(s['user_id'] for s in filtered_sessions if s.get('user_id'))
    except:
        all_session_user_ids = experience_user_ids

    # 전날 대비 변화량 계산
    prev_day_stats = calculate_prev_day_comparison(filtered_dialogs, last_date, all_registered_user_ids, session_user_map)
    cumulative_stats['registered_users'] = len(all_registered_user_ids)
    cumulative_stats['experience_users'] = len(experience_user_ids)
    cumulative_stats['prev_day'] = prev_day_stats

    # 대화 빈도 분포 (실시간 + 누적)
    realtime_stats['conversation_freq'] = calculate_conversation_frequency(
        realtime_dialogs, realtime_user_ids if realtime_user_ids else None)
    cumulative_stats['conversation_freq'] = calculate_conversation_frequency(
        filtered_dialogs, all_registered_user_ids)

    # 베이커별 분석 (필터링된 분석 데이터 사용)
    baker_stats = defaultdict(lambda: {
        'dialog_count': 0,
        'unique_users': set(),
        'scores': [],
        'sentiment_counts': {'positive': 0, 'neutral': 0, 'negative': 0}
    })
    
    for dialog in filtered_dialogs:
        session_id = dialog['session_id']
        chatbot_id = session_chatbot_map.get(session_id)
        user_id = session_user_map.get(session_id)
        dialog_id = dialog['dialog_id']
        
        if not chatbot_id:
            continue
        
        baker_stats[chatbot_id]['dialog_count'] += 1
        if user_id:
            baker_stats[chatbot_id]['unique_users'].add(user_id)
        
        # 분석 결과 적용
        analysis = filtered_analyzed.get(dialog_id, {})
        fav = analysis.get('favorability', {})
        score = fav.get('score', 3)
        sentiment = fav.get('sentiment', 'neutral')
        
        baker_stats[chatbot_id]['scores'].append(score)
        if sentiment in baker_stats[chatbot_id]['sentiment_counts']:
            baker_stats[chatbot_id]['sentiment_counts'][sentiment] += 1
    
    # 베이커 데이터 정리
    baker_list = []
    for chatbot_id, stats in baker_stats.items():
        if stats['dialog_count'] == 0:
            continue
        
        avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 3.0
        unique_users = len(stats['unique_users'])
        
        baker_list.append({
            'chatbot_id': chatbot_id,
            'name': chatbot_name_map.get(chatbot_id, f'베이커{chatbot_id}'),
            'dialog_count': stats['dialog_count'],
            'unique_users': unique_users,
            'avg_score': avg_score,
            'sentiment_counts': stats['sentiment_counts']
        })
    
    # 정규화 및 호감지수 계산
    if baker_list:
        max_dialogs = max(b['dialog_count'] for b in baker_list)
        max_users = max(b['unique_users'] for b in baker_list)
        
        for b in baker_list:
            norm_dialog = b['dialog_count'] / max_dialogs if max_dialogs > 0 else 0
            norm_users = b['unique_users'] / max_users if max_users > 0 else 0
            norm_score = (b['avg_score'] - 1) / 4
            b['popularity_score'] = round((norm_dialog + norm_users + norm_score) / 3 * 100, 1)
    
    baker_list.sort(key=lambda x: x['popularity_score'], reverse=True)
    
    # 4분면 분류
    median_dialogs = 0
    median_users = 0
    quadrants = {'popular': [], 'minority_fan': [], 'shallow': [], 'ignored': []}
    
    if len(baker_list) >= 2:
        dialog_counts = [b['dialog_count'] for b in baker_list]
        user_counts = [b['unique_users'] for b in baker_list]
        median_dialogs = sorted(dialog_counts)[len(dialog_counts)//2]
        median_users = sorted(user_counts)[len(user_counts)//2]
        
        for b in baker_list:
            high_dialogs = b['dialog_count'] >= median_dialogs
            high_users = b['unique_users'] >= median_users
            
            if high_dialogs and high_users:
                b['quadrant'] = 'popular'
            elif high_dialogs and not high_users:
                b['quadrant'] = 'minority_fan'
            elif not high_dialogs and high_users:
                b['quadrant'] = 'shallow'
            else:
                b['quadrant'] = 'ignored'
            
            quadrants[b['quadrant']].append({'name': b['name'], 'chatbot_id': b['chatbot_id']})
    
    # 감정별 TOP
    for b in baker_list:
        total = sum(b['sentiment_counts'].values())
        if total > 0:
            b['positive_ratio'] = b['sentiment_counts']['positive'] / total
            b['negative_ratio'] = b['sentiment_counts']['negative'] / total
            b['neutral_ratio'] = b['sentiment_counts']['neutral'] / total
        else:
            b['positive_ratio'] = b['negative_ratio'] = b['neutral_ratio'] = 0
    
    top_praised = sorted(baker_list, key=lambda x: x['positive_ratio'], reverse=True)[:5]
    top_criticized = sorted(baker_list, key=lambda x: x['negative_ratio'], reverse=True)[:5]
    top_neutral = sorted(baker_list, key=lambda x: x['neutral_ratio'], reverse=True)[:5]
    
    # 위험도 분석
    risk_count = 0
    severity_counts = {'high': 0, 'medium': 0, 'safe': 0}
    type_counts = {'family': 0, 'personal': 0, 'spoiler': 0}
    type_severity = {t: {'high': 0, 'medium': 0, 'safe': 0} for t in type_counts}
    baker_risk = defaultdict(lambda: {'risk_count': 0, 'safe_count': 0})
    
    # 일별 위험 통계 (추이 차트용)
    daily_risk_stats = defaultdict(lambda: {'total': 0, 'risk': 0, 'safe': 0})
    
    # 먼저 모든 대화의 일별 총 대화수 계산
    for d in filtered_dialogs:
        date_str = d.get('created_at', '')[:10]
        if date_str:
            daily_risk_stats[date_str]['total'] += 1
    
    for dialog_id, analysis in filtered_analyzed.items():
        risk = analysis.get('risk', {})
        # has_risk 또는 detected 둘 다 지원
        if risk.get('has_risk') or risk.get('detected'):
            risk_count += 1
            # risks 배열에서 첫 번째 위험 정보 추출 (새 형식), 없으면 직접 필드 사용 (구 형식)
            risks_list = risk.get('risks', [])
            if risks_list:
                first_risk = risks_list[0]
                severity = first_risk.get('severity', 'safe')
                risk_type = first_risk.get('type', 'personal')
            else:
                severity = risk.get('severity', 'safe')
                risk_type = risk.get('type', 'personal')
            
            if severity in severity_counts:
                severity_counts[severity] += 1
            if risk_type in type_counts:
                type_counts[risk_type] += 1
                if severity in type_severity[risk_type]:
                    type_severity[risk_type][severity] += 1
            
            # 베이커별 위험 집계 및 일별 통계
            dialog = next((d for d in filtered_dialogs if d['dialog_id'] == dialog_id), None)
            if dialog:
                chatbot_id = session_chatbot_map.get(dialog['session_id'])
                if chatbot_id:
                    baker_risk[chatbot_id]['risk_count'] += 1
                    if severity == 'safe':
                        baker_risk[chatbot_id]['safe_count'] += 1
                
                # 일별 위험 통계
                date_str = dialog.get('created_at', '')[:10]
                if date_str:
                    daily_risk_stats[date_str]['risk'] += 1
                    if severity == 'safe':
                        daily_risk_stats[date_str]['safe'] += 1
    
    # 일별 추이 데이터 생성 (날짜순 정렬)
    daily_trend = []
    for date_str in sorted(daily_risk_stats.keys()):
        stats = daily_risk_stats[date_str]
        total = stats['total']
        risk = stats['risk']
        safe = stats['safe']
        
        risk_ratio = (risk / total * 100) if total > 0 else 0
        safe_ratio = (safe / risk * 100) if risk > 0 else 100
        
        daily_trend.append({
            'date': date_str,
            'total_dialogs': total,
            'risk_count': risk,
            'safe_count': safe,
            'risk_ratio': round(risk_ratio, 2),
            'safe_ratio': round(safe_ratio, 2)
        })
    
    total_risk = severity_counts['high'] + severity_counts['medium'] + severity_counts['safe']
    safe_response_ratio = severity_counts['safe'] / total_risk if total_risk > 0 else 1.0
    
    # 베이커별 위험 정리
    baker_risk_list = []
    for chatbot_id, stats in baker_risk.items():
        if stats['risk_count'] > 0:
            safe_rate = stats['safe_count'] / stats['risk_count']
            baker_risk_list.append({
                'chatbot_id': chatbot_id,
                'name': chatbot_name_map.get(chatbot_id, f'베이커{chatbot_id}'),
                'risk_count': stats['risk_count'],
                'safe_rate': safe_rate
            })
    
    top_risk_bakers = sorted(baker_risk_list, key=lambda x: x['risk_count'], reverse=True)[:5]
    vulnerable_bakers = sorted(baker_risk_list, key=lambda x: x['safe_rate'])[:5]
    
    # 위험 대화 상세 목록 (all_items)
    risk_items = []
    for dialog_id, analysis in filtered_analyzed.items():
        risk = analysis.get('risk', {})
        if risk.get('has_risk') or risk.get('detected'):
            dialog = next((d for d in filtered_dialogs if d['dialog_id'] == dialog_id), None)
            if dialog:
                chatbot_id = session_chatbot_map.get(dialog['session_id'])
                # risks 배열에서 정보 추출 (새 형식), 없으면 직접 필드 사용 (구 형식)
                risks_list = risk.get('risks', [])
                if risks_list:
                    first_risk = risks_list[0]
                    risk_type = first_risk.get('type', 'personal')
                    severity = first_risk.get('severity', 'safe')
                    reason = first_risk.get('reason', '')
                else:
                    risk_type = risk.get('type', 'personal')
                    severity = risk.get('severity', 'safe')
                    reason = risk.get('reason', '')
                
                risk_items.append({
                    'dialog_id': dialog_id,
                    'user_id': dialog.get('user_id', ''),
                    'chatbot_id': chatbot_id,
                    'chatbot_name': chatbot_name_map.get(chatbot_id, f'베이커{chatbot_id}') if chatbot_id else '알수없음',
                    'question': dialog.get('question', ''),
                    'answer': dialog.get('answer', ''),
                    'created_at': dialog.get('created_at', ''),
                    'risk_type': risk_type,
                    'severity': severity,
                    'reason': reason
                })
    
    # 사용자 분석
    user_stats = []
    for user_id, user_dialogs in dialogs_by_user.items():
        scores = []
        risk_dialogs = 0
        baker_dialogs = defaultdict(int)  # 베이커별 대화 수 (편애율 계산용)
        
        for d in user_dialogs:
            analysis = filtered_analyzed.get(d['dialog_id'], {})
            fav = analysis.get('favorability', {})
            scores.append(fav.get('score', 3))
            
            risk_info = analysis.get('risk', {})
            if risk_info.get('has_risk') or risk_info.get('detected'):
                risk_dialogs += 1
            
            chatbot_id = session_chatbot_map.get(d['session_id'])
            if chatbot_id:
                baker_dialogs[chatbot_id] += 1
        
        avg_fav = sum(scores) / len(scores) if scores else 3.0
        risk_rate = risk_dialogs / len(user_dialogs) if user_dialogs else 0
        
        # 편애율 계산 (가장 많이 대화한 베이커의 비율)
        total_dialogs = len(user_dialogs)
        if baker_dialogs:
            top_baker_id = max(baker_dialogs, key=baker_dialogs.get)
            top_baker_count = baker_dialogs[top_baker_id]
            bias_rate = top_baker_count / total_dialogs if total_dialogs > 0 else 0
            biased_baker_name = chatbot_name_map.get(top_baker_id, f'베이커{top_baker_id}')
        else:
            bias_rate = 0
            biased_baker_name = '-'
        
        # 유형 분류
        if avg_fav >= 4.0 and risk_rate < 0.10:
            user_type = 'fan'
        elif avg_fav < 2.5 or risk_rate >= 0.30:
            user_type = 'troll'
        else:
            user_type = 'neutral'
        
        user_stats.append({
            'user_id': user_id,
            'dialog_count': total_dialogs,
            'baker_count': len(baker_dialogs),
            'avg_favorability': round(avg_fav, 2),
            'risk_rate': round(risk_rate, 3),
            'user_type': user_type,
            'bias_rate': round(bias_rate, 3),
            'biased_baker_name': biased_baker_name
        })
    
    type_counts_user = {'fan': 0, 'neutral': 0, 'troll': 0}
    for u in user_stats:
        type_counts_user[u['user_type']] += 1
    
    # 대화수별 분포
    dialog_dist = {'1-5': 0, '6-10': 0, '11-20': 0, '21-50': 0, '51+': 0}
    for u in user_stats:
        dc = u['dialog_count']
        if dc <= 5:
            dialog_dist['1-5'] += 1
        elif dc <= 10:
            dialog_dist['6-10'] += 1
        elif dc <= 20:
            dialog_dist['11-20'] += 1
        elif dc <= 50:
            dialog_dist['21-50'] += 1
        else:
            dialog_dist['51+'] += 1
    
    top_users = sorted(user_stats, key=lambda x: x['dialog_count'], reverse=True)[:10]
    
    # 고객 여정 분석 - 방문 횟수 분포
    visit_distribution = {'1회': 0, '2회': 0, '3-5회': 0, '6-10회': 0, '11회+': 0}
    for user_id, user_dlgs in dialogs_by_user.items():
        visit_info = calculate_user_visits(user_dlgs)
        vc = visit_info['visit_count']
        if vc == 1:
            visit_distribution['1회'] += 1
        elif vc == 2:
            visit_distribution['2회'] += 1
        elif vc <= 5:
            visit_distribution['3-5회'] += 1
        elif vc <= 10:
            visit_distribution['6-10회'] += 1
        else:
            visit_distribution['11회+'] += 1
    
    baker_count_dist = {'1': 0, '2-5': 0, '6-10': 0, '11-30': 0, '31-72': 0}
    for u in user_stats:
        bc = u['baker_count']
        if bc == 1:
            baker_count_dist['1'] += 1
        elif bc <= 5:
            baker_count_dist['2-5'] += 1
        elif bc <= 10:
            baker_count_dist['6-10'] += 1
        elif bc <= 30:
            baker_count_dist['11-30'] += 1
        else:
            baker_count_dist['31-72'] += 1
    
    # 대화 간격 분포
    gap_distribution = {
        '0-1분': 0, '1-5분': 0, '5-10분': 0, '10-30분': 0,
        '30분-1시간': 0, '1-2시간': 0, '2-6시간': 0,
        '6-12시간': 0, '12-24시간': 0, '24시간+': 0
    }
    
    for user_id, user_dialogs in dialogs_by_user.items():
        sorted_dialogs = sorted(user_dialogs, key=lambda x: x.get('created_at', ''))
        for i in range(1, len(sorted_dialogs)):
            prev_time = parse_datetime(sorted_dialogs[i-1].get('created_at'))
            curr_time = parse_datetime(sorted_dialogs[i].get('created_at'))
            if prev_time and curr_time:
                gap_seconds = (curr_time - prev_time).total_seconds()
                if gap_seconds < 60:
                    gap_distribution['0-1분'] += 1
                elif gap_seconds < 300:
                    gap_distribution['1-5분'] += 1
                elif gap_seconds < 600:
                    gap_distribution['5-10분'] += 1
                elif gap_seconds < 1800:
                    gap_distribution['10-30분'] += 1
                elif gap_seconds < 3600:
                    gap_distribution['30분-1시간'] += 1
                elif gap_seconds < 7200:
                    gap_distribution['1-2시간'] += 1
                elif gap_seconds < 21600:
                    gap_distribution['2-6시간'] += 1
                elif gap_seconds < 43200:
                    gap_distribution['6-12시간'] += 1
                elif gap_seconds < 86400:
                    gap_distribution['12-24시간'] += 1
                else:
                    gap_distribution['24시간+'] += 1
    
    # 결과 조립
    result = {
        'realtime': realtime_stats,
        'cumulative': cumulative_stats,
        'favorability': {
            'popular_bakers': baker_list[:10],
            'top_by_dialogs': [
                {'chatbot_id': b['chatbot_id'], 'name': b['name'], 'value': b['dialog_count']}
                for b in sorted(baker_list, key=lambda x: x['dialog_count'], reverse=True)[:10]
            ],
            'top_by_users': [
                {'chatbot_id': b['chatbot_id'], 'name': b['name'], 'value': b['unique_users']}
                for b in sorted(baker_list, key=lambda x: x['unique_users'], reverse=True)[:10]
            ],
            'top_by_favorability': [
                {'chatbot_id': b['chatbot_id'], 'name': b['name'], 'value': round(b['avg_score'], 2)}
                for b in sorted(baker_list, key=lambda x: x['avg_score'], reverse=True)[:10]
            ],
            'baker_ranking': baker_list[:20],
            'all_bakers_quadrant': baker_list,
            'quadrant_info': {
                'median_dialogs': median_dialogs,
                'median_users': median_users,
                'quadrants': quadrants
            },
            'top_praised': top_praised,
            'top_criticized': top_criticized,
            'top_neutral': top_neutral
        },
        'risk': {
            'total_dialogs': len(filtered_dialogs),
            'risk_count': risk_count,
            'risk_ratio': risk_count / len(filtered_dialogs) if filtered_dialogs else 0,
            'safe_response_ratio': safe_response_ratio,
            'type_counts': type_counts,
            'type_severity': type_severity,
            'severity_counts': severity_counts,
            'top_risk_bakers': top_risk_bakers,
            'vulnerable_bakers': vulnerable_bakers,
            'all_items': sorted(risk_items, key=lambda x: x['created_at'], reverse=True)[:200],
            'daily_trend': daily_trend
        },
        'users': {
            'total_count': len(user_stats),
            'avg_dialog_count': sum(u['dialog_count'] for u in user_stats) / len(user_stats) if user_stats else 0,
            'dialog_distribution': dialog_dist,
            'baker_distribution': baker_count_dist,
            'type_counts': type_counts_user,
            'top_users': top_users
        },
        'journey': {
            'visit_distribution': visit_distribution,
            'baker_count_distribution': baker_count_dist,
            'gap_distribution': gap_distribution,
            'conversation_freq': cumulative_stats.get('conversation_freq', {})
        },
        'meta': {
            'analysis_method': 'hybrid',
            'analysis_type': 'filtered',
            'date_filter': f'{start_date} ~ {end_date}',
            'analysis_range': {
                'start_date': start_date,
                'end_date': end_date
            },
            'data_range': {
                'min_date': min(d.get('created_at', '')[:10] for d in filtered_dialogs) if filtered_dialogs else '',
                'max_date': max(d.get('created_at', '')[:10] for d in filtered_dialogs) if filtered_dialogs else ''
            },
            'analyzed_at': datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S'),
            'total_analyzed_dialogs': len(filtered_dialogs)
        }
    }
    
    return result


# ============================================
# 7. 통계 계산 함수들
# ============================================
def calculate_basic_stats(dialogs, demographics):
    """기본 통계 계산 (대화수, 사용자수)"""
    user_ids = set(d['user_id'] for d in dialogs if d.get('user_id'))

    return {
        'total_dialogs': len(dialogs),
        'total_users': len(user_ids)
    }


def calculate_conversation_frequency(dialogs, all_session_user_ids=None):
    """사용자별 대화 빈도 분포 계산

    Args:
        dialogs: 대화 데이터 리스트
        all_session_user_ids: 세션을 생성한 전체 유저 ID set (대화 없는 유저 포함)

    Returns:
        dict: {'대화없음': n, '1회': n, '2회': n, '3-5회': n, '6-10회': n, '11-20회': n, '21회+': n}
    """
    # 대화가 있는 유저별 대화 수 계산
    user_dialog_count = defaultdict(int)
    for d in dialogs:
        user_id = d.get('user_id', '')
        if user_id:
            user_dialog_count[user_id] += 1

    freq = {'대화없음': 0, '1회': 0, '2회': 0, '3-5회': 0, '6-10회': 0, '11-20회': 0, '21회+': 0}

    # 대화 없는 유저 수 계산
    if all_session_user_ids:
        dialog_user_ids = set(user_dialog_count.keys())
        no_dialog_users = all_session_user_ids - dialog_user_ids
        freq['대화없음'] = len(no_dialog_users)

    # 대화가 있는 유저 분류
    for user_id, count in user_dialog_count.items():
        if count == 1:
            freq['1회'] += 1
        elif count == 2:
            freq['2회'] += 1
        elif count <= 5:
            freq['3-5회'] += 1
        elif count <= 10:
            freq['6-10회'] += 1
        elif count <= 20:
            freq['11-20회'] += 1
        else:
            freq['21회+'] += 1

    return freq


def calculate_hourly_stats(dialogs):
    """시간대별 대화량"""
    hourly = {f"{h:02d}": 0 for h in range(24)}
    for d in dialogs:
        created_at = d.get('created_at', '')
        if len(created_at) >= 13:
            hour = created_at[11:13]
            if hour in hourly:
                hourly[hour] += 1
    return hourly


def calculate_hourly_users_stats(dialogs):
    """시간대별 고유 사용자 수"""
    hourly_users = {f"{h:02d}": set() for h in range(24)}
    for d in dialogs:
        created_at = d.get('created_at', '')
        user_id = d.get('user_id', '')
        if len(created_at) >= 13 and user_id:
            hour = created_at[11:13]
            if hour in hourly_users:
                hourly_users[hour].add(user_id)
    return {h: len(users) for h, users in hourly_users.items()}


def calculate_daily_stats(dialogs):
    """일별 대화량"""
    daily = defaultdict(int)
    for d in dialogs:
        created_at = d.get('created_at', '')
        if len(created_at) >= 10:
            date_str = created_at[:10]
            daily[date_str] += 1
    return dict(sorted(daily.items()))


def calculate_daily_users_stats(dialogs):
    """일별 고유 사용자 수"""
    daily_users = defaultdict(set)
    for d in dialogs:
        created_at = d.get('created_at', '')
        user_id = d.get('user_id', '')
        if len(created_at) >= 10 and user_id:
            date_str = created_at[:10]
            daily_users[date_str].add(user_id)
    return dict(sorted({date: len(users) for date, users in daily_users.items()}.items()))


def calculate_prev_day_comparison(dialogs, last_date, all_session_user_ids, session_user_map):
    """전날 대비 변화량 계산"""
    from datetime import timedelta

    if not last_date or not dialogs:
        return {'new_users': 0, 'new_users_rate': 0, 'new_dialogs': 0, 'new_dialogs_rate': 0,
                'new_experience_users': 0, 'new_experience_users_rate': 0}

    try:
        last_dt = datetime.strptime(last_date, '%Y-%m-%d')
        prev_date = (last_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    except:
        return {'new_users': 0, 'new_users_rate': 0, 'new_dialogs': 0, 'new_dialogs_rate': 0,
                'new_experience_users': 0, 'new_experience_users_rate': 0}

    # 전날까지의 데이터 vs 마지막날 포함 데이터 비교
    prev_dialogs = [d for d in dialogs if d.get('created_at', '')[:10] <= prev_date]
    last_day_dialogs = [d for d in dialogs if d.get('created_at', '')[:10] == last_date]

    prev_dialog_count = len(prev_dialogs)
    new_dialog_count = len(last_day_dialogs)

    prev_experience_users = set(d.get('user_id', '') for d in prev_dialogs if d.get('user_id'))
    last_day_users = set(d.get('user_id', '') for d in last_day_dialogs if d.get('user_id'))
    new_experience_users = len(last_day_users - prev_experience_users)

    # 신규 등록 유저 (세션 기반)
    prev_session_users = set()
    for d in prev_dialogs:
        sid = d.get('session_id', '')
        uid = session_user_map.get(sid)
        if uid:
            prev_session_users.add(uid)
    new_registered = len(all_session_user_ids) - len(prev_session_users) if prev_session_users else 0

    return {
        'new_users': new_registered,
        'new_users_rate': round(new_registered / max(len(prev_session_users), 1) * 100, 2),
        'new_dialogs': new_dialog_count,
        'new_dialogs_rate': round(new_dialog_count / max(prev_dialog_count, 1) * 100, 2),
        'new_experience_users': new_experience_users,
        'new_experience_users_rate': round(new_experience_users / max(len(prev_experience_users), 1) * 100, 2)
    }


# ============================================
# 8. 메인 처리 로직
# ============================================
def main(start_date=None, end_date=None, analysis_type='full', pre_analyzed=None):
    """
    메인 데이터 처리 함수
    
    Args:
        start_date: 분석 시작 날짜
        end_date: 분석 종료 날짜
        analysis_type: 'full' (전체 재분석) 또는 'incremental' (증분 분석)
        pre_analyzed: 이미 분석된 대화 데이터 (증분 분석 시 사용)
    """
    print("=" * 60)
    print(f"mbn-analytics v2 데이터 처리 ({'전체 재분석' if analysis_type == 'full' else '증분 분석'})")
    print("=" * 60)
    
    # 데이터 로드
    print("\n[1/8] 데이터 로드...")
    chatbots = load_chatbots()
    sessions = load_sessions()
    dialogs = load_dialogs()
    
    # 날짜 범위 결정
    data_min_date, data_max_date = get_date_range_from_data(dialogs)
    
    if start_date is None and end_date is None:
        start_date = data_min_date  # 전체 분석은 처음부터
        end_date = data_max_date
        print(f"   [전체 모드] 전체 기간 분석: {start_date} ~ {end_date}")
    else:
        if start_date is None:
            start_date = DATE_FILTER
        if end_date is None:
            end_date = data_max_date
        print(f"   [날짜 범위 모드] {start_date} ~ {end_date}")
    
    # 날짜 범위로 필터링
    dialogs = filter_dialogs_by_date(dialogs, start_date, end_date)
    # 세션도 날짜 범위로 필터링 (대화 없는 유저도 포함하기 위해 독립적으로 필터링)
    all_sessions_in_range = [s for s in sessions if start_date <= s['created_at'][:10] <= end_date]
    session_ids_with_dialogs = set(d['session_id'] for d in dialogs)
    sessions_with_dialogs = [s for s in all_sessions_in_range if s['session_id'] in session_ids_with_dialogs]

    print(f"   베이커: {len(chatbots)}명")
    print(f"   세션(전체): {len(all_sessions_in_range)}개, 세션(대화있음): {len(sessions_with_dialogs)}개")
    print(f"   대화: {len(dialogs)}개")

    # 인덱싱
    dialogs_by_session = defaultdict(list)
    dialogs_by_user = defaultdict(list)
    dialogs_by_chatbot = defaultdict(list)

    for d in dialogs:
        dialogs_by_session[d['session_id']].append(d)
        dialogs_by_user[d['user_id']].append(d)
        dialogs_by_chatbot[d['chatbot_id']].append(d)

    # 전체 세션 기반 유저 집합 (대화 없는 유저 포함)
    all_session_user_ids_full = set(s['user_id'] for s in all_sessions_in_range if s.get('user_id'))
    demographics = {}
    
    # GPT 분석 (pre_analyzed가 있으면 건너뜀)
    if pre_analyzed is not None:
        print("\n[3/8] 분석된 데이터 사용 (증분 분석 결과)...")
        analyzed_dialogs = pre_analyzed
        print(f"   분석된 대화: {len(analyzed_dialogs)}개")
    else:
        print("\n[3/8] 하이브리드 분석 실행 (키워드 1차 + GPT 2차)...")
        analyzer = HybridAnalyzer()
        
        def progress_callback(current, total):
            update_progress(current, total, "GPT 분석")
            if current % 10 == 0 or current == total:
                print(f"   진행: {current}/{total} ({current/total*100:.1f}%)")
        
        analyzed_dialogs = analyzer.analyze_batch(dialogs, progress_callback=progress_callback)
        print(f"   분석 완료: {len(analyzed_dialogs)}개")
    
    clear_progress()
    
    # 실시간/누적 통계
    print("\n[4/8] 실시간/누적 통계 계산...")
    today_dialogs = filter_dialogs_by_today(analyzed_dialogs)
    
    # 오늘 데이터가 없으면 마지막 날짜 데이터 사용
    if not today_dialogs and analyzed_dialogs:
        last_dialog = max(analyzed_dialogs, key=lambda x: x.get('created_at', ''))
        last_date = last_dialog.get('created_at', '')[:10] if last_dialog.get('created_at') else ''
        last_time = last_dialog.get('created_at', '')[11:16] if len(last_dialog.get('created_at', '')) >= 16 else ''
        today_dialogs = [d for d in analyzed_dialogs if d.get('created_at', '')[:10] == last_date]
    else:
        last_dialog = max(today_dialogs, key=lambda x: x.get('created_at', '')) if today_dialogs else None
        last_date = last_dialog.get('created_at', '')[:10] if last_dialog else ''
        last_time = last_dialog.get('created_at', '')[11:16] if last_dialog and len(last_dialog.get('created_at', '')) >= 16 else ''
    
    realtime_stats = calculate_basic_stats(today_dialogs, demographics)
    realtime_stats['hourly_stats'] = calculate_hourly_stats(today_dialogs)
    realtime_stats['hourly_users_stats'] = calculate_hourly_users_stats(today_dialogs)
    realtime_stats['last_date'] = last_date
    realtime_stats['last_time'] = last_time

    cumulative_stats = calculate_basic_stats(analyzed_dialogs, demographics)
    cumulative_stats['daily_stats'] = calculate_daily_stats(analyzed_dialogs)
    cumulative_stats['daily_users_stats'] = calculate_daily_users_stats(analyzed_dialogs)

    # 전체 현황 overview 추가 데이터 계산
    experience_user_ids = set(d.get('user_id', '') for d in analyzed_dialogs if d.get('user_id'))
    session_user_map_full = {s['session_id']: s['user_id'] for s in all_sessions_in_range if s.get('user_id')}

    # 등록 유저: anonymous_user.csv (전체 등록) > chatsession 기반
    anonymous_users = load_anonymous_users()
    if anonymous_users:
        all_registered_user_ids = set(
            uid for uid, info in anonymous_users.items()
            if info.get('created_at', '')[:10] <= end_date
        )
        print(f"   전체 등록 유저 (anonymous_user): {len(all_registered_user_ids)}명")
    else:
        all_registered_user_ids = all_session_user_ids_full
        print(f"   세션 기반 유저: {len(all_registered_user_ids)}명")

    prev_day_stats = calculate_prev_day_comparison(analyzed_dialogs, last_date, all_registered_user_ids, session_user_map_full)
    cumulative_stats['registered_users'] = len(all_registered_user_ids)
    cumulative_stats['experience_users'] = len(experience_user_ids)
    cumulative_stats['prev_day'] = prev_day_stats

    # 대화 빈도 분포
    realtime_stats['conversation_freq'] = calculate_conversation_frequency(today_dialogs)
    cumulative_stats['conversation_freq'] = calculate_conversation_frequency(
        analyzed_dialogs, all_registered_user_ids)
    
    print(f"   실시간({last_date}): {realtime_stats['total_dialogs']}개 대화, {realtime_stats['total_users']}명")
    print(f"   누적: {cumulative_stats['total_dialogs']}개 대화, {cumulative_stats['total_users']}명")
    
    # 호감도 분석 (베이커 중심)
    print("\n[5/8] 호감도 분석 (베이커 중심)...")
    
    baker_stats = {}
    for chatbot_id, chatbot_dialogs in dialogs_by_chatbot.items():
        if not chatbot_id.startswith('C-CHJB'):
            continue
        
        # 해당 베이커의 분석된 대화 찾기
        baker_analyzed = [ad for ad in analyzed_dialogs if ad['chatbot_id'] == chatbot_id]
        
        if not baker_analyzed:
            continue
        
        # 감정 분포
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        scores = []
        for ad in baker_analyzed:
            fav = ad.get('favorability', {})
            sentiment = fav.get('sentiment', 'neutral')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            scores.append(fav.get('score', 3))
        
        # 고유 사용자 수
        unique_users = len(set(d['user_id'] for d in chatbot_dialogs))
        
        baker_name = chatbots.get(chatbot_id, chatbot_id)
        baker_stats[chatbot_id] = {
            'name': baker_name,
            'dialog_count': len(baker_analyzed),
            'unique_users': unique_users,
            'concentration': len(baker_analyzed) / unique_users if unique_users > 0 else 0,
            'avg_score': round(sum(scores) / len(scores), 2) if scores else 3.0,
            'sentiment_counts': sentiment_counts,
            'positive_ratio': sentiment_counts['positive'] / len(baker_analyzed) if baker_analyzed else 0,
            'negative_ratio': sentiment_counts['negative'] / len(baker_analyzed) if baker_analyzed else 0
        }
    
    # 4분면 분류 (X축: 사용자수, Y축: 대화수 기준)
    if baker_stats:
        dialog_counts = [b['dialog_count'] for b in baker_stats.values()]
        user_counts = [b['unique_users'] for b in baker_stats.values()]
        median_dialogs = np.median(dialog_counts)
        median_users = np.median(user_counts)
        
        for chatbot_id, stats in baker_stats.items():
            stats['quadrant'] = classify_baker_quadrant(
                stats['dialog_count'],
                stats['unique_users'],
                median_dialogs,
                median_users
            )
    
    # 베이커 랭킹 (각 지표별 Top10)
    top_by_dialogs = sorted(baker_stats.items(), key=lambda x: -x[1]['dialog_count'])[:10]
    top_by_users = sorted(baker_stats.items(), key=lambda x: -x[1]['unique_users'])[:10]
    top_by_favorability = sorted(baker_stats.items(), key=lambda x: -x[1]['avg_score'])[:10]
    
    # 인기 베이커 종합 점수 계산 (대화량 + 사용자수 + 호감도 정규화 합산)
    if baker_stats:
        max_dialogs = max(b['dialog_count'] for b in baker_stats.values())
        max_users = max(b['unique_users'] for b in baker_stats.values())
        max_score = 5.0  # 호감도 최대값
        min_score = 1.0  # 호감도 최소값
        
        for chatbot_id, stats in baker_stats.items():
            # 각 지표를 0~1로 정규화
            dialog_norm = stats['dialog_count'] / max_dialogs if max_dialogs > 0 else 0
            user_norm = stats['unique_users'] / max_users if max_users > 0 else 0
            score_norm = (stats['avg_score'] - min_score) / (max_score - min_score)
            
            # 종합 인기 점수 (각 지표 동일 가중치)
            stats['popularity_score'] = round((dialog_norm + user_norm + score_norm) / 3 * 100, 1)
    
    # 인기 베이커 Top10
    popular_bakers = sorted(baker_stats.items(), key=lambda x: -x[1]['popularity_score'])[:10]
    
    # 기존 랭킹 (대화량 기준)
    baker_ranking = sorted(baker_stats.items(), key=lambda x: -x[1]['dialog_count'])[:20]
    top_praised = sorted(baker_stats.items(), key=lambda x: -x[1]['positive_ratio'])[:5]
    top_criticized = sorted(baker_stats.items(), key=lambda x: -x[1]['negative_ratio'])[:5]
    top_neutral = sorted(baker_stats.items(), key=lambda x: -((1 - x[1]['positive_ratio'] - x[1]['negative_ratio'])))[:5]
    
    # 4분면별 베이커 목록
    quadrant_bakers = defaultdict(list)
    for chatbot_id, stats in baker_stats.items():
        quadrant_bakers[stats['quadrant']].append({
            'chatbot_id': chatbot_id,
            'name': stats['name'],
            'dialog_count': stats['dialog_count'],
            'unique_users': stats['unique_users'],
            'concentration': round(stats['concentration'], 2)
        })
    
    print(f"   베이커 {len(baker_stats)}명 분석 완료")
    
    # 위험도 분석
    print("\n[6/8] 위험도 분석...")
    
    total_dialogs = len(analyzed_dialogs)
    risk_dialogs = [d for d in analyzed_dialogs if d.get('risk', {}).get('has_risk')]
    risk_count = len(risk_dialogs)
    
    # 위험 유형별 카운트 및 severity별 카운트
    risk_type_counts = {'family': 0, 'personal': 0, 'spoiler': 0}
    risk_type_severity = {
        'family': {'high': 0, 'medium': 0, 'safe': 0},
        'personal': {'high': 0, 'medium': 0, 'safe': 0},
        'spoiler': {'high': 0, 'medium': 0, 'safe': 0}
    }
    severity_counts = {'high': 0, 'medium': 0, 'safe': 0}
    risk_items = []
    
    for d in analyzed_dialogs:
        risk = d.get('risk', {})
        if risk.get('has_risk'):
            for r in risk.get('risks', []):
                risk_type = r.get('type')
                severity = r.get('severity')
                
                if risk_type in risk_type_counts:
                    risk_type_counts[risk_type] += 1
                    if severity in risk_type_severity[risk_type]:
                        risk_type_severity[risk_type][severity] += 1
                if severity in severity_counts:
                    severity_counts[severity] += 1
                
                risk_items.append({
                    'dialog_id': d['dialog_id'],
                    'user_id': d['user_id'],
                    'chatbot_id': d['chatbot_id'],
                    'chatbot_name': chatbots.get(d['chatbot_id'], d['chatbot_id']),
                    'question': d['question'][:100],
                    'answer': d['answer'][:200],
                    'created_at': d['created_at'],
                    'risk_type': risk_type,
                    'severity': severity,
                    'reason': r.get('reason', '')
                })
    
    # 위험유형별 안전대응률 계산
    risk_type_safe_rates = {}
    for rt in risk_type_counts:
        total = risk_type_counts[rt]
        safe = risk_type_severity[rt]['safe']
        risk_type_safe_rates[rt] = round(safe / total, 4) if total > 0 else 1.0
    
    # 위험 비율 계산
    risk_ratio = risk_count / total_dialogs if total_dialogs > 0 else 0
    
    # 안전대응률 = 위험질문 건수 대비 안전대응 건수
    # 총 위험 건수 = high + medium + safe (severity별 합계)
    # 안전대응 건수 = safe 건수
    total_risk_items = severity_counts['high'] + severity_counts['medium'] + severity_counts['safe']
    safe_response_count = severity_counts['safe']
    safe_response_ratio = safe_response_count / total_risk_items if total_risk_items > 0 else 1.0
    
    # 베이커별 위험 대응률 (위험질문 건수 대비 안전대응 건수)
    baker_risk_stats = {}
    for chatbot_id, stats in baker_stats.items():
        baker_risk_dialogs = [d for d in analyzed_dialogs 
                            if d['chatbot_id'] == chatbot_id and d.get('risk', {}).get('has_risk')]
        
        # 위험 건수와 안전대응 건수 계산 (개별 risk 항목 기준)
        baker_total_risks = 0
        baker_safe_count = 0
        for d in baker_risk_dialogs:
            for r in d.get('risk', {}).get('risks', []):
                baker_total_risks += 1
                if r.get('severity') == 'safe':
                    baker_safe_count += 1
        
        baker_risk_stats[chatbot_id] = {
            'name': stats['name'],
            'risk_count': baker_total_risks,  # 위험 건수
            'safe_count': baker_safe_count,   # 안전 대응 건수
            'safe_rate': baker_safe_count / baker_total_risks if baker_total_risks > 0 else 1.0
        }
    
    # 취약 베이커 (안전대응률 낮은 순, 위험질문 1개 이상)
    vulnerable_bakers = sorted(
        [(k, v) for k, v in baker_risk_stats.items() if v['risk_count'] > 0],
        key=lambda x: x[1]['safe_rate']
    )[:5]
    
    # 위험질문 많이 받는 베이커 (위험질문 수 기준)
    top_risk_bakers = sorted(
        [(k, v) for k, v in baker_risk_stats.items() if v['risk_count'] > 0],
        key=lambda x: -x[1]['risk_count']
    )[:5]
    
    print(f"   위험 질문 비율: {risk_ratio*100:.1f}%")
    print(f"   안전 대응 비율: {safe_response_ratio*100:.1f}%")
    
    # 사용자 분석
    print("\n[7/8] 사용자 분석...")
    
    user_stats = {}
    user_type_counts = {'fan': 0, 'neutral': 0, 'troll': 0}
    
    for user_id, user_dialogs in dialogs_by_user.items():
        user_analyzed = [ad for ad in analyzed_dialogs if ad['user_id'] == user_id]
        
        scores = [ad.get('favorability', {}).get('score', 3) for ad in user_analyzed]
        avg_fav = sum(scores) / len(scores) if scores else 3.0
        
        risk_count = sum(1 for ad in user_analyzed if ad.get('risk', {}).get('has_risk'))
        risk_rate = risk_count / len(user_analyzed) if user_analyzed else 0
        
        baker_ids = set(d['chatbot_id'] for d in user_dialogs if d['chatbot_id'].startswith('C-CHJB'))
        
        user_type = classify_user_type(avg_fav, risk_rate)
        user_type_counts[user_type] += 1
        
        # 방문 횟수 계산
        visit_info = calculate_user_visits(user_dialogs)
        
        # 편애 분석 (가장 많이 대화한 베이커 비율 계산)
        baker_dialog_counts = defaultdict(int)
        for d in user_dialogs:
            if d['chatbot_id'].startswith('C-CHJB'):
                baker_dialog_counts[d['chatbot_id']] += 1
        
        bias_rate = 0
        biased_baker = None
        if baker_dialog_counts:
            max_baker = max(baker_dialog_counts.items(), key=lambda x: x[1])
            bias_rate = max_baker[1] / len(user_dialogs) if len(user_dialogs) > 0 else 0
            biased_baker = max_baker[0]
        
        user_stats[user_id] = {
            'dialog_count': len(user_dialogs),
            'baker_count': len(baker_ids),
            'avg_favorability': round(avg_fav, 2),
            'risk_rate': round(risk_rate, 3),
            'risk_count': risk_count,
            'user_type': user_type,
            'visit_count': visit_info['visit_count'],
            'bias_rate': round(bias_rate, 3),  # 편애율 추가
            'biased_baker': biased_baker,
            'biased_baker_name': chatbots.get(biased_baker, biased_baker) if biased_baker else None,
            'gender': demographics.get(user_id, {}).get('gender', 'unknown'),
            'age': demographics.get(user_id, {}).get('age', 'unknown')
        }
    
    # 대화수 분포
    dialog_distribution = {
        '1-5': 0, '6-10': 0, '11-20': 0, '21-50': 0, '51+': 0
    }
    for stats in user_stats.values():
        dc = stats['dialog_count']
        if dc <= 5:
            dialog_distribution['1-5'] += 1
        elif dc <= 10:
            dialog_distribution['6-10'] += 1
        elif dc <= 20:
            dialog_distribution['11-20'] += 1
        elif dc <= 50:
            dialog_distribution['21-50'] += 1
        else:
            dialog_distribution['51+'] += 1
    
    # 베이커수 분포
    baker_distribution = {
        '1': 0, '2-5': 0, '6-10': 0, '11-30': 0, '31-72': 0
    }
    for stats in user_stats.values():
        bc = stats['baker_count']
        if bc == 1:
            baker_distribution['1'] += 1
        elif bc <= 5:
            baker_distribution['2-5'] += 1
        elif bc <= 10:
            baker_distribution['6-10'] += 1
        elif bc <= 30:
            baker_distribution['11-30'] += 1
        else:
            baker_distribution['31-72'] += 1
    
    # TOP 사용자
    top_users = sorted(user_stats.items(), key=lambda x: -x[1]['dialog_count'])[:10]
    risk_users = sorted(user_stats.items(), key=lambda x: -x[1]['risk_count'])[:10]
    
    avg_dialog_count = sum(s['dialog_count'] for s in user_stats.values()) / len(user_stats) if user_stats else 0
    
    print(f"   사용자 {len(user_stats)}명 분석 완료")
    print(f"   유형: 팬 {user_type_counts['fan']} / 중립 {user_type_counts['neutral']} / 악플러 {user_type_counts['troll']}")
    
    # 고객 여정 분석
    print("\n[8/8] 고객 여정 분석...")
    
    # 방문 횟수 분포
    visit_distribution = {
        '1회': 0, '2회': 0, '3-5회': 0, '6-10회': 0, '11회+': 0
    }
    for stats in user_stats.values():
        vc = stats['visit_count']
        if vc == 1:
            visit_distribution['1회'] += 1
        elif vc == 2:
            visit_distribution['2회'] += 1
        elif vc <= 5:
            visit_distribution['3-5회'] += 1
        elif vc <= 10:
            visit_distribution['6-10회'] += 1
        else:
            visit_distribution['11회+'] += 1
    
    # 대화 간격 분포
    all_gaps = []
    for user_id in dialogs_by_user.keys():
        visit_info = calculate_user_visits(dialogs_by_user[user_id])
        all_gaps.extend(visit_info['gaps'])
    
    gap_distribution = {}
    if all_gaps:
        gaps_minutes = [g / 60 for g in all_gaps]
        bins = [0, 1, 5, 10, 30, 60, 120, 360, 720, 1440, float('inf')]
        labels = ['0-1분', '1-5분', '5-10분', '10-30분', '30분-1시간', '1-2시간', '2-6시간', '6-12시간', '12-24시간', '24시간+']
        
        for i in range(len(bins) - 1):
            count = sum(1 for g in gaps_minutes if bins[i] <= g < bins[i+1])
            gap_distribution[labels[i]] = count
    
    print(f"   방문 분포 계산 완료")
    
    # JSON 저장
    print("\n" + "=" * 60)
    print("JSON 저장 중...")
    
    data = {
        # 실시간 현황 (오늘)
        'realtime': realtime_stats,
        
        # 누적 현황
        'cumulative': cumulative_stats,
        
        # 호감도 분석
        'favorability': {
            # 인기 베이커 Top10 (종합 점수)
            'popular_bakers': [
                {
                    'chatbot_id': b[0],
                    'name': b[1]['name'],
                    'dialog_count': b[1]['dialog_count'],
                    'unique_users': b[1]['unique_users'],
                    'avg_score': b[1]['avg_score'],
                    'popularity_score': b[1]['popularity_score'],
                    'sentiment_counts': b[1]['sentiment_counts'],
                    'quadrant': b[1]['quadrant']
                }
                for b in popular_bakers
            ],
            # 각 지표별 Top10
            'top_by_dialogs': [
                {'chatbot_id': b[0], 'name': b[1]['name'], 'value': b[1]['dialog_count']}
                for b in top_by_dialogs
            ],
            'top_by_users': [
                {'chatbot_id': b[0], 'name': b[1]['name'], 'value': b[1]['unique_users']}
                for b in top_by_users
            ],
            'top_by_favorability': [
                {'chatbot_id': b[0], 'name': b[1]['name'], 'value': b[1]['avg_score']}
                for b in top_by_favorability
            ],
            'baker_ranking': [
                {
                    'chatbot_id': b[0],
                    'name': b[1]['name'],
                    'dialog_count': b[1]['dialog_count'],
                    'unique_users': b[1]['unique_users'],
                    'avg_score': b[1]['avg_score'],
                    'sentiment_counts': b[1]['sentiment_counts'],
                    'quadrant': b[1]['quadrant']
                }
                for b in baker_ranking
            ],
            # 4분면 차트용 전체 베이커 데이터
            'all_bakers_quadrant': [
                {
                    'chatbot_id': chatbot_id,
                    'name': stats['name'],
                    'dialog_count': stats['dialog_count'],
                    'unique_users': stats['unique_users'],
                    'quadrant': stats['quadrant']
                }
                for chatbot_id, stats in baker_stats.items()
            ],
            'quadrant_info': {
                'median_dialogs': float(median_dialogs) if baker_stats else 0,
                'median_users': float(median_users) if baker_stats else 0,
                'quadrant_bakers': {k: v for k, v in quadrant_bakers.items()}
            },
            'top_praised': [{'chatbot_id': b[0], 'name': b[1]['name'], 'positive_ratio': round(b[1]['positive_ratio'], 3)} for b in top_praised],
            'top_criticized': [{'chatbot_id': b[0], 'name': b[1]['name'], 'negative_ratio': round(b[1]['negative_ratio'], 3)} for b in top_criticized],
            'top_neutral': [{'chatbot_id': b[0], 'name': b[1]['name']} for b in top_neutral]
        },
        
        # 위험도 분석
        'risk': {
            'total_dialogs': total_dialogs,
            'risk_count': risk_count,
            'risk_ratio': round(risk_ratio, 4),
            'safe_response_ratio': round(safe_response_ratio, 4),
            'type_counts': risk_type_counts,
            'type_severity': risk_type_severity,  # 위험유형별 severity 카운트
            'type_safe_rates': risk_type_safe_rates,  # 위험유형별 안전대응률
            'severity_counts': severity_counts,
            'baker_response_rate': {
                k: {
                    'name': v['name'],
                    'risk_count': v['risk_count'],
                    'safe_rate': round(v['safe_rate'], 3)
                }
                for k, v in baker_risk_stats.items() if v['risk_count'] > 0
            },
            'top_risk_bakers': [  # 위험질문 많이 받는 베이커
                {'chatbot_id': b[0], 'name': b[1]['name'], 'risk_count': b[1]['risk_count'], 'safe_rate': round(b[1]['safe_rate'], 3)}
                for b in top_risk_bakers
            ],
            'vulnerable_bakers': [
                {'chatbot_id': b[0], 'name': b[1]['name'], 'risk_count': b[1]['risk_count'], 'safe_rate': round(b[1]['safe_rate'], 3)}
                for b in vulnerable_bakers
            ],
            'all_items': sorted(  # 전체 위험 대화 (클릭 시 필터링용)
                risk_items,
                key=lambda x: x['created_at'], reverse=True
            )[:200]
        },
        
        # 사용자 분석
        'users': {
            'total_count': len(user_stats),
            'avg_dialog_count': round(avg_dialog_count, 1),
            'dialog_distribution': dialog_distribution,
            'baker_distribution': baker_distribution,
            'type_counts': user_type_counts,
            'top_users': [
                {
                    'user_id': u[0],
                    'dialog_count': u[1]['dialog_count'],
                    'baker_count': u[1]['baker_count'],
                    'avg_favorability': u[1]['avg_favorability'],
                    'risk_rate': u[1]['risk_rate'],
                    'user_type': u[1]['user_type'],
                    'bias_rate': u[1]['bias_rate'],
                    'biased_baker_name': u[1]['biased_baker_name']
                }
                for u in top_users
            ],
            'risk_users': [
                {'user_id': u[0], 'risk_count': u[1]['risk_count'], 'dialog_count': u[1]['dialog_count']}
                for u in risk_users if u[1]['risk_count'] > 0
            ]
        },
        
        # 고객 여정 분석
        'journey': {
            'visit_distribution': visit_distribution,
            'baker_count_distribution': baker_distribution,
            'gap_distribution': gap_distribution,
            'conversation_freq': cumulative_stats.get('conversation_freq', {})
        },

        # 메타 정보
        'meta': {
            'analysis_method': 'Hybrid (키워드 1차 + GPT 2차)',
            'analysis_type': analysis_type,  # 'full' or 'incremental'
            'date_filter': DATE_FILTER,
            'analysis_range': {
                'start_date': start_date,
                'end_date': end_date
            },
            'data_range': {
                'min_date': data_min_date,
                'max_date': data_max_date
            },
            'analyzed_at': datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S'),
            'total_analyzed_dialogs': len(analyzed_dialogs)
        }
    }
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # JSON 저장
    with open(DASHBOARD_DATA_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 동기화 정보 업데이트
    if analyzed_dialogs:
        last_dialog = max(analyzed_dialogs, key=lambda x: x.get('created_at', ''))
        sync_manager.update_sync_info(
            last_dialog_id=last_dialog['dialog_id'],
            last_created_at=last_dialog['created_at'],
            total_analyzed=len(analyzed_dialogs),
            analysis_type=analysis_type
        )
        
        # 분석된 대화 캐시 저장 (증분 분석용)
        sync_manager.save_analyzed_dialogs(analyzed_dialogs)
    
    print(f"완료! dashboard_data.json 생성됨")
    print("=" * 60)
    
    # 요약 출력
    print(f"\n[요약]")
    print(f"  분석 유형: {'전체 재분석' if analysis_type == 'full' else '증분 분석'}")
    print(f"  실시간(오늘): {realtime_stats['total_dialogs']}개 대화")
    print(f"  누적: {cumulative_stats['total_dialogs']}개 대화, {cumulative_stats['total_users']}명 사용자")
    print(f"  위험 비율: {risk_ratio*100:.1f}%, 안전 대응: {safe_response_ratio*100:.1f}%")
    print(f"  사용자 유형: 팬 {user_type_counts['fan']} / 중립 {user_type_counts['neutral']} / 악플러 {user_type_counts['troll']}")


def main_incremental(start_date=None, end_date=None):
    """증분 분석: 새로 추가된 대화만 분석하여 기존 결과에 병합"""
    
    print("=" * 60)
    print("증분 분석 시작")
    print("=" * 60)
    
    # 기존 분석 데이터 로드
    existing_analyzed = sync_manager.load_analyzed_dialogs()
    sync_info = sync_manager.get_sync_info()
    
    if not existing_analyzed:
        print("기존 분석 데이터가 없습니다. 전체 분석을 먼저 실행하세요.")
        return
    
    print(f"기존 분석 데이터: {len(existing_analyzed)}개")
    print(f"마지막 분석: {sync_info.get('analyzed_at', '없음')}")
    
    # CSV 로드 (load_dialogs 함수 사용 - 헤더 없는 CSV 처리)
    print("\n[1/4] CSV 파일 로드 중...")
    all_dialogs = load_dialogs()
    
    # 새 대화 필터링
    new_dialogs = sync_manager.filter_new_dialogs(all_dialogs)
    print(f"새로 추가된 대화: {len(new_dialogs)}개")
    
    if not new_dialogs:
        print("새로 추가된 대화가 없습니다.")
        return
    
    # 날짜 필터 적용
    if start_date and end_date:
        new_dialogs = [d for d in new_dialogs 
                      if start_date <= d.get('created_at', '')[:10] <= end_date]
        print(f"날짜 필터 적용 후: {len(new_dialogs)}개")
    
    # 새 대화 분석
    print("\n[2/4] 새 대화 분석 중...")
    analyzer = HybridAnalyzer()
    
    # 분석할 대화 준비
    dialogs_to_analyze = []
    for dialog in new_dialogs:
        dialogs_to_analyze.append({
            'dialog_id': dialog['dialog_id'],
            'session_id': dialog.get('session_id', ''),  # session_id 추가
            'user_id': dialog.get('user_id', ''),
            'chatbot_id': dialog.get('chatbot_id', ''),
            'question': dialog.get('question', ''),
            'answer': dialog.get('answer', ''),
            'created_at': dialog.get('created_at', '')
        })
    
    # 분석 실행
    new_analyzed = analyzer.analyze_batch(
        dialogs_to_analyze,
        progress_callback=lambda c, t: update_progress(c, t, "증분 GPT 분석")
    )
    
    print(f"\n[3/4] 결과 병합 중...")
    # 기존 결과와 병합
    merged_analyzed = sync_manager.merge_analyzed_dialogs(existing_analyzed, new_analyzed)
    print(f"병합 결과: {len(merged_analyzed)}개 (기존 {len(existing_analyzed)} + 새 {len(new_analyzed)})")
    
    # 전체 통계 재계산 및 저장
    print("\n[4/4] 통계 재계산 중...")
    # main 함수의 통계 계산 로직 재사용을 위해 전체 분석 함수 호출
    # 단, 분석은 건너뛰고 이미 분석된 데이터로 통계만 계산
    
    # 분석된 대화 캐시 저장
    sync_manager.save_analyzed_dialogs(merged_analyzed)
    
    # 통계 재계산을 위해 main 함수 호출 (분석 건너뛰기 모드)
    main(start_date, end_date, analysis_type='incremental', pre_analyzed=merged_analyzed)


def main_reanalyze_period(start_date, end_date):
    """특정 기간만 재분석: 해당 기간의 대화를 다시 분석하여 기존 결과에 병합"""
    
    print("=" * 60)
    print(f"기간 재분석 시작: {start_date} ~ {end_date}")
    print("=" * 60)
    
    # UTC → KST 변환을 위해 start_date를 하루 전으로 조정 (UTC 15:00 = KST 다음날 00:00)
    # 실제 CSV 데이터는 UTC이므로, KST 기준으로 필터링하려면 조정 필요
    
    # 기존 분석 데이터 로드
    existing_analyzed = sync_manager.load_analyzed_dialogs()
    
    if not existing_analyzed:
        print("기존 분석 데이터가 없습니다. 전체 분석을 먼저 실행하세요.")
        return
    
    print(f"기존 분석 데이터: {len(existing_analyzed)}개")
    
    # 해당 기간의 대화를 기존 분석에서 제외 (KST 변환 후 비교)
    period_dialogs_ids = set()
    remaining_analyzed = []
    
    for d in existing_analyzed:
        # analyzed_dialogs.json에 저장된 시간은 이미 KST (중복 변환 방지)
        created_at = d.get('created_at', '')
        date_str = created_at[:10] if created_at else ''
        
        if start_date <= date_str <= end_date:
            period_dialogs_ids.add(d.get('dialog_id'))
        else:
            remaining_analyzed.append(d)
    
    print(f"재분석 대상 기간 대화: {len(period_dialogs_ids)}개")
    print(f"유지될 기존 분석: {len(remaining_analyzed)}개")
    
    if not period_dialogs_ids:
        print("해당 기간에 분석된 대화가 없습니다.")
        return
    
    # CSV에서 해당 기간 대화 로드
    print("\n[1/4] CSV 파일에서 기간 대화 로드 중...")
    all_dialogs = load_dialogs()
    
    # 해당 기간 대화 필터링 (KST 기준)
    period_dialogs = []
    for d in all_dialogs:
        date_str = d.get('created_at', '')[:10]  # 이미 KST 변환됨 (load_dialogs에서)
        if start_date <= date_str <= end_date:
            period_dialogs.append(d)
    
    print(f"CSV에서 로드된 기간 대화: {len(period_dialogs)}개")
    
    if not period_dialogs:
        print("해당 기간에 대화가 없습니다.")
        return
    
    # 대화 재분석
    print("\n[2/4] 기간 대화 재분석 중...")
    analyzer = HybridAnalyzer()
    
    dialogs_to_analyze = []
    for dialog in period_dialogs:
        dialogs_to_analyze.append({
            'dialog_id': dialog['dialog_id'],
            'session_id': dialog.get('session_id', ''),
            'user_id': dialog.get('user_id', ''),
            'chatbot_id': dialog.get('chatbot_id', ''),
            'question': dialog.get('question', ''),
            'answer': dialog.get('answer', ''),
            'created_at': dialog.get('created_at', '')
        })
    
    # 분석 실행
    reanalyzed = analyzer.analyze_batch(
        dialogs_to_analyze,
        progress_callback=lambda c, t: update_progress(c, t, "기간 재분석")
    )
    
    print(f"\n[3/4] 결과 병합 중...")
    # 기존 결과(해당 기간 제외)와 재분석 결과 병합
    merged_analyzed = sync_manager.merge_analyzed_dialogs(remaining_analyzed, reanalyzed)
    print(f"병합 결과: {len(merged_analyzed)}개 (기존 {len(remaining_analyzed)} + 재분석 {len(reanalyzed)})")
    
    # 저장
    sync_manager.save_analyzed_dialogs(merged_analyzed)
    
    # 통계 재계산
    print("\n[4/4] 통계 재계산 중...")
    main(analysis_type='reanalyze', pre_analyzed=merged_analyzed)
    
    print("\n" + "=" * 60)
    print(f"기간 재분석 완료: {start_date} ~ {end_date}")
    print(f"재분석된 대화: {len(reanalyzed)}개")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python data_processor.py full                     # 전체 재분석")
        print("  python data_processor.py incremental              # 증분 분석")
        print("  python data_processor.py full 2026-01-13 2026-01-19  # 날짜 범위 전체 분석")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == 'full':
        # 전체 재분석
        sync_manager.reset()  # 동기화 정보 초기화
        if len(sys.argv) == 2:
            main(analysis_type='full')
        elif len(sys.argv) == 3:
            main(sys.argv[2], sys.argv[2], analysis_type='full')
        elif len(sys.argv) >= 4:
            main(sys.argv[2], sys.argv[3], analysis_type='full')
    
    elif mode == 'incremental':
        # 증분 분석
        if len(sys.argv) == 2:
            main_incremental()
        elif len(sys.argv) == 3:
            main_incremental(sys.argv[2], sys.argv[2])
        elif len(sys.argv) >= 4:
            main_incremental(sys.argv[2], sys.argv[3])
    
    elif mode == 'reanalyze':
        # 기간 재분석
        if len(sys.argv) >= 4:
            main_reanalyze_period(sys.argv[2], sys.argv[3])
        else:
            print("사용법: python data_processor.py reanalyze 2026-02-01 2026-02-02")
            sys.exit(1)
    
    else:
        print(f"알 수 없는 모드: {mode}")
        print("'full', 'incremental', 또는 'reanalyze'를 사용하세요.")
        sys.exit(1)
