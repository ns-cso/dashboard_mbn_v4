"""
리포트 생성 모듈
대시보드 데이터를 기반으로 GPT를 활용한 분석 리포트 생성
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from openai import OpenAI

from config import (
    OPENAI_API_KEY, GPT_MODEL, GPT_TEMPERATURE, GPT_MAX_TOKENS, GPT_TIMEOUT,
    RATE_LIMIT_DELAY, DASHBOARD_DATA_JSON, OUTPUT_DIR
)

# 리포트 데이터 출력 경로
REPORT_DATA_JSON = os.path.join(OUTPUT_DIR, 'report_data.json')


class ReportGenerator:
    """리포트 생성기 클래스"""
    
    # 26개 차트 ID 목록
    CHART_IDS = [
        # 전체 현황 (3개)
        'overview-summary', 'rt-hourly', 'cum-daily',
        # 호감도 분석 (7개)
        'fav-top10', 'fav-dialogs', 'fav-users', 'fav-score', 'fav-table', 'fav-quadrant', 'fav-emotion',
        # 위험도 분석 (6개)
        'risk-ratio', 'risk-safe', 'risk-type', 'risk-severity', 'risk-baker-high', 'risk-baker-low',
        # 사용자 분석 (4개)
        'user-summary', 'user-type', 'user-dialog', 'user-top10',
        # 고객 여정 (3개)
        'journey-visit', 'journey-baker', 'journey-gap'
    ]
    
    def __init__(self):
        """초기화"""
        self.client = None
        if OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.last_api_call = 0
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None
    
    def _wait_for_rate_limit(self):
        """API 호출 간격 조절"""
        elapsed = time.time() - self.last_api_call
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_api_call = time.time()
    
    def _call_gpt(self, system_prompt: str, user_message: str) -> str:
        """GPT API 호출"""
        if not self.client:
            return "GPT API 키가 설정되지 않았습니다."
        
        self._wait_for_rate_limit()
        
        try:
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=GPT_TEMPERATURE,
                max_tokens=GPT_MAX_TOKENS,
                timeout=GPT_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"GPT API 호출 실패: {e}")
            return f"분석 생성 실패: {str(e)}"
    
    def _call_gpt_json(self, system_prompt: str, user_message: str) -> dict:
        """GPT API 호출 (JSON 응답)"""
        if not self.client:
            return {}
        
        self._wait_for_rate_limit()
        
        try:
            response = self.client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=GPT_TEMPERATURE,
                max_tokens=1500,
                timeout=GPT_TIMEOUT,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"GPT API 호출 실패: {e}")
            return {}
    
    def _get_chart_analysis_prompt(self, chart_id: str) -> str:
        """차트별 분석 프롬프트 반환"""
        prompts = {
            # 전체 현황
            'overview-summary': """서비스 핵심 지표 데이터를 분석하세요.
- 누적 유저 수, 대화 체험 유저 수, 누적 대화수 등 핵심 지표 요약
- 체험 전환율(대화 체험 유저 / 누적 유저)의 의미
- 전날 대비 증감 트렌드 분석
- 3문장 이내로 작성""",

            'rt-hourly': """오늘 시간대별 대화량 데이터를 분석하세요.
- 피크 시간대와 특징
- 운영 및 서버 리소스 관리에 대한 시사점
- 2문장 이내로 간결하게 작성""",

            'cum-daily': """일별 대화량 추이 데이터를 분석하세요.
- 전반적인 트렌드 (성장/하락/안정)
- 특이점이 있는 날짜와 추정 원인
- 2문장 이내로 간결하게 작성""",
            
            # 호감도 분석
            'fav-top10': """호감지수 Top 10 베이커 데이터를 분석하세요.
- 상위 베이커들의 공통적 특징
- 인기 요인 분석
- 3문장 이내로 작성""",
            
            'fav-dialogs': """대화량 Top 10 베이커 데이터를 분석하세요.
- 대화량 상위 베이커의 특징
- 대화량과 인기도의 상관관계
- 2문장 이내로 간결하게 작성""",
            
            'fav-users': """고유 사용자수 Top 10 베이커 데이터를 분석하세요.
- 사용자 도달 범위가 넓은 베이커의 특징
- 대중성 관점에서의 시사점
- 2문장 이내로 간결하게 작성""",
            
            'fav-score': """긍부정지수 Top 10 베이커 데이터를 분석하세요.
- 호감도가 높은 베이커의 특징
- 사용자 만족도 관점에서의 시사점
- 2문장 이내로 간결하게 작성""",
            
            'fav-table': """호감지수 상세 데이터 테이블을 분석하세요.
- 종합적인 베이커 성과 평가
- 주목할 만한 베이커와 개선이 필요한 베이커
- 3문장 이내로 작성""",
            
            'fav-quadrant': """베이커 4분면 분석 결과를 해석하세요.
- 각 유형별(대중인기형, 소수광팬형, 넓고얕은형, 무관심형) 베이커 특징
- 유형별 관리 전략 제안
- 3문장 이내로 작성""",
            
            'fav-emotion': """감정별 TOP 베이커 데이터를 분석하세요.
- 칭찬/비난/중립 각 카테고리의 특징
- 감정 반응에 대한 시사점과 개선 방향
- 3문장 이내로 작성""",
            
            # 위험도 분석
            'risk-ratio': """위험 질문 비율 데이터를 평가하세요.
- 현재 위험 수준 평가 (양호/주의/경고)
- 개선 방향 제시
- 2문장 이내로 간결하게 작성""",
            
            'risk-safe': """전체 안전 대응률 데이터를 평가하세요.
- 현재 안전 대응 수준 평가
- 목표 대응률 대비 현황
- 2문장 이내로 간결하게 작성""",
            
            'risk-type': """위험 유형별 분포 데이터를 분석하세요.
- 가장 빈번한 위험 유형과 특징
- 유형별 대응 우선순위 제안
- 2문장 이내로 간결하게 작성""",
            
            'risk-severity': """위험 유형별 심각도 분포 데이터를 분석하세요.
- High/Medium/Safe 비율 분석
- 가장 취약한 영역과 개선 방향
- 2문장 이내로 간결하게 작성""",
            
            'risk-baker-high': """위험 질문 다빈도 베이커 데이터를 분석하세요.
- 위험 질문을 많이 받는 베이커의 특징
- 위험 질문 감소를 위한 제안
- 2문장 이내로 간결하게 작성""",
            
            'risk-baker-low': """안전 대응 취약 베이커 데이터를 분석하세요.
- 안전 대응률이 낮은 베이커 파악
- 시급한 개선이 필요한 베이커와 조치 방안
- 2문장 이내로 간결하게 작성""",
            
            # 사용자 분석
            'user-summary': """사용자 요약 데이터(전체 사용자, 평균 대화수, 팬/악플러 비율)를 분석하세요.
- 전반적인 사용자 건강도 평가
- 커뮤니티 상태에 대한 시사점
- 2문장 이내로 간결하게 작성""",
            
            'user-type': """사용자 유형 분포(팬/중립/악플러) 데이터를 분석하세요.
- 각 유형별 비율과 의미
- 건강한 커뮤니티 유지를 위한 제안
- 2문장 이내로 간결하게 작성""",
            
            'user-dialog': """대화수별 사용자 분포 데이터를 분석하세요.
- 사용자 참여도 분포 특징
- 이탈 방지 및 참여 유도 전략
- 2문장 이내로 간결하게 작성""",
            
            'user-top10': """대화량 TOP 10 사용자 데이터를 분석하세요.
- 헤비유저의 특성과 행동 패턴
- VIP 사용자 관리에 대한 시사점
- 2문장 이내로 간결하게 작성""",
            
            # 고객 여정
            'journey-visit': """방문 횟수 분포 데이터를 분석하세요.
- 재방문율과 충성도 특징
- 재방문 유도 전략 제안
- 2문장 이내로 간결하게 작성""",
            
            'journey-baker': """대화 베이커 수 분포 데이터를 분석하세요.
- 사용자의 베이커 탐색 패턴
- 다양한 베이커 이용 유도 전략
- 2문장 이내로 간결하게 작성""",
            
            'journey-gap': """대화 간격 분포 데이터를 분석하세요.
- 사용자 세션 패턴 특징
- 서비스 체류 시간 증대 전략
- 2문장 이내로 간결하게 작성"""
        }
        
        return prompts.get(chart_id, "해당 데이터를 간결하게 분석하세요.")
    
    def _get_chart_data(self, chart_id: str, dashboard_data: dict) -> dict:
        """차트 ID에 해당하는 데이터 추출"""
        rt = dashboard_data.get('realtime', {})
        cum = dashboard_data.get('cumulative', {})
        fav = dashboard_data.get('favorability', {})
        risk = dashboard_data.get('risk', {})
        users = dashboard_data.get('users', {})
        journey = dashboard_data.get('journey', {})
        
        data_map = {
            # 전체 현황
            'overview-summary': {
                'registered_users': cum.get('registered_users', cum.get('total_users', 0)),
                'experience_users': cum.get('experience_users', cum.get('total_users', 0)),
                'total_dialogs': cum.get('total_dialogs', 0),
                'today_dialogs': rt.get('total_dialogs', 0),
                'today_users': rt.get('total_users', 0),
                'prev_day': cum.get('prev_day', {})
            },
            'rt-hourly': rt.get('hourly_stats', {}),
            'cum-daily': {
                'total_days': len(cum.get('daily_stats', {})),
                'total_dialogs': cum.get('total_dialogs', 0),
                'date_range': list(cum.get('daily_stats', {}).keys())[:5] + ['...'] + list(cum.get('daily_stats', {}).keys())[-5:] if cum.get('daily_stats') else []
            },
            
            # 호감도 분석
            'fav-top10': [{'name': b['name'], 'score': b.get('popularity_score', 0), 'dialogs': b.get('dialog_count', 0)} for b in fav.get('popular_bakers', [])[:10]],
            'fav-dialogs': [{'name': b['name'], 'count': b.get('dialog_count', 0)} for b in fav.get('top_by_dialogs', [])[:10]],
            'fav-users': [{'name': b['name'], 'count': b.get('unique_users', 0)} for b in fav.get('top_by_users', [])[:10]],
            'fav-score': [{'name': b['name'], 'score': b.get('avg_score', 0)} for b in fav.get('top_by_favorability', [])[:10]],
            'fav-table': [{'name': b['name'], 'score': b.get('popularity_score', 0), 'dialogs': b.get('dialog_count', 0), 'users': b.get('unique_users', 0)} for b in fav.get('baker_ranking', [])[:20]],
            'fav-quadrant': fav.get('quadrant_info', {}),
            'fav-emotion': {
                'praised': [{'name': b['name'], 'ratio': b.get('positive_ratio', 0)} for b in fav.get('top_praised', [])[:5]],
                'criticized': [{'name': b['name'], 'ratio': b.get('negative_ratio', 0)} for b in fav.get('top_criticized', [])[:5]],
                'neutral': [{'name': b['name'], 'ratio': b.get('neutral_ratio', 0)} for b in fav.get('top_neutral', [])[:5]]
            },
            
            # 위험도 분석
            'risk-ratio': {'risk_ratio': risk.get('risk_ratio', 0), 'risk_count': risk.get('risk_count', 0), 'total': risk.get('total_dialogs', 0)},
            'risk-safe': {'safe_ratio': risk.get('safe_response_ratio', 0), 'severity': risk.get('severity_counts', {})},
            'risk-type': risk.get('type_counts', {}),
            'risk-severity': risk.get('type_severity', {}),
            'risk-baker-high': [{'name': b['name'], 'risk_count': b.get('risk_count', 0), 'safe_rate': b.get('safe_rate', 0)} for b in risk.get('top_risk_bakers', [])[:5]],
            'risk-baker-low': [{'name': b['name'], 'risk_count': b.get('risk_count', 0), 'safe_rate': b.get('safe_rate', 0)} for b in risk.get('vulnerable_bakers', [])[:5]],
            
            # 사용자 분석
            'user-summary': {
                'total': users.get('total_count', 0),
                'avg_dialogs': users.get('avg_dialog_count', 0),
                'type_counts': users.get('type_counts', {})
            },
            'user-type': users.get('type_counts', {}),
            'user-dialog': users.get('dialog_distribution', {}),
            'user-top10': [{'id': u.get('user_id', '')[:8], 'dialogs': u.get('dialog_count', 0), 'bakers': u.get('baker_count', 0), 'type': u.get('type', '')} for u in users.get('top_users', [])[:10]],
            
            # 고객 여정
            'journey-visit': journey.get('visit_distribution', {}),
            'journey-baker': journey.get('baker_count_distribution', {}),
            'journey-gap': journey.get('gap_distribution', {})
        }
        
        return data_map.get(chart_id, {})
    
    def analyze_chart(self, chart_id: str, dashboard_data: dict) -> str:
        """개별 차트 분석 생성"""
        chart_data = self._get_chart_data(chart_id, dashboard_data)
        prompt = self._get_chart_analysis_prompt(chart_id)
        
        system_prompt = """당신은 천하제빵 AI Baker 서비스 데이터 분석 전문가입니다.
주어진 데이터를 기반으로 비즈니스 인사이트를 도출하세요.
- 객관적인 수치를 언급하며 분석하세요.
- 실행 가능한 제안을 포함하세요.
- 지시된 문장 수 이내로 간결하게 작성하세요.
- 한국어로 작성하세요."""
        
        user_message = f"""분석 요청: {prompt}

데이터:
{json.dumps(chart_data, ensure_ascii=False, indent=2)}"""
        
        return self._call_gpt(system_prompt, user_message)
    
    def generate_executive_summary(self, dashboard_data: dict) -> dict:
        """경영진 요약 생성"""
        system_prompt = """당신은 천하제빵 AI Baker 서비스의 데이터 분석 임원 보고서 작성자입니다.
주어진 대시보드 데이터를 분석하여 경영진 요약을 JSON 형식으로 작성하세요.

응답 형식:
{
    "key_findings": ["발견점1", "발견점2", "발견점3"],
    "warnings": ["주의사항1", "주의사항2"],
    "recommendations": ["권고사항1", "권고사항2", "권고사항3"]
}

- key_findings: 3-5개의 주요 발견점 (긍정적/중립적 내용)
- warnings: 2-3개의 주의가 필요한 사항 (위험 요소, 개선 필요 영역)
- recommendations: 2-4개의 실행 가능한 권고사항
- 각 항목은 간결하게 1문장으로 작성
- 한국어로 작성"""
        
        # 요약 데이터 준비
        cum = dashboard_data.get('cumulative', {})
        risk = dashboard_data.get('risk', {})
        users = dashboard_data.get('users', {})
        fav = dashboard_data.get('favorability', {})
        
        summary_data = {
            'total_dialogs': cum.get('total_dialogs', 0),
            'total_users': cum.get('total_users', 0),
            'risk_ratio': risk.get('risk_ratio', 0),
            'safe_response_ratio': risk.get('safe_response_ratio', 0),
            'user_types': users.get('type_counts', {}),
            'top_baker': fav.get('popular_bakers', [{}])[0].get('name', '-') if fav.get('popular_bakers') else '-',
            'vulnerable_bakers_count': len(risk.get('vulnerable_bakers', []))
        }
        
        user_message = f"""다음 대시보드 핵심 지표를 분석하여 경영진 요약을 작성하세요:

{json.dumps(summary_data, ensure_ascii=False, indent=2)}"""
        
        result = self._call_gpt_json(system_prompt, user_message)
        
        return {
            'key_findings': result.get('key_findings', []),
            'warnings': result.get('warnings', []),
            'recommendations': result.get('recommendations', [])
        }
    
    def generate_insights(self, dashboard_data: dict) -> dict:
        """종합 인사이트 생성"""
        system_prompt = """당신은 천하제빵 AI Baker 서비스의 데이터 기반 전략 컨설턴트입니다.
주어진 대시보드 데이터를 분석하여 종합 인사이트를 JSON 형식으로 작성하세요.

응답 형식:
{
    "achievements": ["성과1", "성과2", "성과3"],
    "improvements": ["개선점1", "개선점2", "개선점3"],
    "action_items": [
        {"priority": "urgent", "action": "액션1", "owner": "담당팀", "expected_effect": "기대효과"},
        {"priority": "important", "action": "액션2", "owner": "담당팀", "expected_effect": "기대효과"},
        {"priority": "normal", "action": "액션3", "owner": "담당팀", "expected_effect": "기대효과"}
    ]
}

- achievements: 3-4개의 핵심 성과 (잘 되고 있는 것)
- improvements: 3-4개의 개선 필요 영역
- action_items: 3-5개의 구체적 액션 아이템
  - priority: "urgent" (긴급), "important" (중요), "normal" (일반)
  - owner: 개발팀, 기획팀, 콘텐츠팀, 운영팀 중 하나
- 한국어로 작성"""
        
        # 분석용 데이터 준비
        cum = dashboard_data.get('cumulative', {})
        risk = dashboard_data.get('risk', {})
        users = dashboard_data.get('users', {})
        fav = dashboard_data.get('favorability', {})
        journey = dashboard_data.get('journey', {})
        
        analysis_data = {
            'service_metrics': {
                'total_dialogs': cum.get('total_dialogs', 0),
                'total_users': cum.get('total_users', 0)
            },
            'risk_metrics': {
                'risk_ratio': risk.get('risk_ratio', 0),
                'safe_response_ratio': risk.get('safe_response_ratio', 0),
                'vulnerable_bakers': [b['name'] for b in risk.get('vulnerable_bakers', [])[:3]]
            },
            'user_health': {
                'fan_ratio': users.get('type_counts', {}).get('fan', 0) / max(users.get('total_count', 1), 1),
                'troll_ratio': users.get('type_counts', {}).get('troll', 0) / max(users.get('total_count', 1), 1)
            },
            'engagement': {
                'single_visit_ratio': journey.get('visit_distribution', {}).get('1회', 0) / max(users.get('total_count', 1), 1)
            },
            'top_performers': [b['name'] for b in fav.get('popular_bakers', [])[:3]]
        }
        
        user_message = f"""다음 데이터를 분석하여 종합 인사이트를 작성하세요:

{json.dumps(analysis_data, ensure_ascii=False, indent=2)}"""
        
        result = self._call_gpt_json(system_prompt, user_message)
        
        return {
            'achievements': result.get('achievements', []),
            'improvements': result.get('improvements', []),
            'action_items': result.get('action_items', [])
        }
    
    def generate_report(self, dashboard_data: dict, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> dict:
        """전체 리포트 생성"""
        self.progress_callback = progress_callback
        
        total_steps = len(self.CHART_IDS) + 2  # 차트 26개 + 경영진요약 + 인사이트
        current_step = 0
        
        def update_progress(message: str):
            nonlocal current_step
            current_step += 1
            if self.progress_callback:
                self.progress_callback(current_step, total_steps, message)
        
        report_data = {
            'meta': {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_period': f"{dashboard_data.get('meta', {}).get('analysis_range', {}).get('start_date', '-')} ~ {dashboard_data.get('meta', {}).get('analysis_range', {}).get('end_date', '-')}"
            },
            'executive_summary': {},
            'charts': {},
            'insights': {}
        }
        
        # 1. 경영진 요약 생성
        update_progress("경영진 요약 생성 중...")
        report_data['executive_summary'] = self.generate_executive_summary(dashboard_data)
        
        # 2. 각 차트 분석 생성
        for chart_id in self.CHART_IDS:
            update_progress(f"차트 분석 중: {chart_id}")
            chart_data = self._get_chart_data(chart_id, dashboard_data)
            analysis = self.analyze_chart(chart_id, dashboard_data)
            report_data['charts'][chart_id] = {
                'data': chart_data,
                'analysis': analysis
            }
        
        # 3. 종합 인사이트 생성
        update_progress("종합 인사이트 생성 중...")
        report_data['insights'] = self.generate_insights(dashboard_data)
        
        return report_data
    
    def load_dashboard_data(self) -> Optional[dict]:
        """대시보드 데이터 로드"""
        if os.path.exists(DASHBOARD_DATA_JSON):
            with open(DASHBOARD_DATA_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save_report_data(self, report_data: dict):
        """리포트 데이터 저장"""
        with open(REPORT_DATA_JSON, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    def get_frame_data(self) -> dict:
        """프레임 데이터 반환 (분석 없이 기본 데이터만)"""
        dashboard_data = self.load_dashboard_data()
        report_data = None
        
        # 기존 리포트 데이터가 있으면 로드
        if os.path.exists(REPORT_DATA_JSON):
            with open(REPORT_DATA_JSON, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
        
        return {
            'dashboard_data': dashboard_data,
            'report_data': report_data
        }


# 리포트 생성 상태 (전역)
report_status = {
    'running': False,
    'completed': False,
    'progress': 0,
    'total': 0,
    'message': '',
    'error': None
}


def generate_report_async(start_date=None, end_date=None):
    """비동기 리포트 생성 (스레드에서 실행)"""
    global report_status

    try:
        report_status['running'] = True
        report_status['completed'] = False
        report_status['error'] = None

        generator = ReportGenerator()

        # 날짜 범위가 지정된 경우 필터링된 데이터 사용
        if start_date and end_date:
            try:
                from data_processor import filter_existing_data
                dashboard_data = filter_existing_data(start_date, end_date)
            except Exception as e:
                print(f"필터링 실패, 기존 데이터 사용: {e}")
                dashboard_data = generator.load_dashboard_data()
        else:
            dashboard_data = generator.load_dashboard_data()

        if not dashboard_data:
            report_status['error'] = '대시보드 데이터가 없습니다.'
            report_status['running'] = False
            return
        
        def progress_callback(current: int, total: int, message: str):
            report_status['progress'] = (current / total) * 100
            report_status['total'] = total
            report_status['message'] = message
        
        report_data = generator.generate_report(dashboard_data, progress_callback)
        generator.save_report_data(report_data)
        
        report_status['completed'] = True
        report_status['progress'] = 100
        report_status['message'] = '리포트 생성 완료'
        
    except Exception as e:
        report_status['error'] = str(e)
    finally:
        report_status['running'] = False


if __name__ == '__main__':
    # 테스트 실행
    generator = ReportGenerator()
    dashboard_data = generator.load_dashboard_data()
    
    if dashboard_data:
        print("대시보드 데이터 로드 완료")
        
        def print_progress(current, total, message):
            print(f"[{current}/{total}] {message}")
        
        report_data = generator.generate_report(dashboard_data, print_progress)
        generator.save_report_data(report_data)
        print(f"\n리포트 저장 완료: {REPORT_DATA_JSON}")
    else:
        print("대시보드 데이터가 없습니다. 먼저 분석을 실행하세요.")
