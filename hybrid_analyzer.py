"""
하이브리드 분석 모듈 (키워드 1차 필터링 + GPT 2차 분석)

기존 gpt_analyzer.py의 최적화 버전:
- 1차: 키워드 기반 빠른 필터링으로 위험 후보 선별
- 2차: 위험 후보만 GPT API로 정밀 분석
- 결과: 속도 대폭 향상, API 비용 절감
"""
import json
import re
import time
from typing import Callable
from openai import OpenAI
from config import (
    OPENAI_API_KEY, GPT_MODEL, GPT_TEMPERATURE,
    GPT_MAX_TOKENS, GPT_TIMEOUT, RATE_LIMIT_DELAY
)


# ============================================================
# 1차 필터링용 키워드 사전
# ============================================================

RISK_KEYWORDS = {
    # 가족정보 관련 키워드
    # 주의: '형', '동생' 등 짧은 단어는 '대형', '이동생' 등에서 오탐지됨
    # 더 구체적인 표현 사용
    'family': {
        'question': [
            '결혼', '아내', '남편', '배우자', '와이프', '신랑', '신부',
            '아들', '딸', '자녀', '자식', '애들', '아이', '아기',
            '부모', '아버지', '어머니', '아빠', '엄마', '부모님',
            '형이', '형은', '형도', '내 형', '제 형', '형제',
            '누나가', '누나는', '내 누나', '제 누나',
            '오빠가', '오빠는', '내 오빠', '제 오빠',
            '언니가', '언니는', '내 언니', '제 언니',
            '동생이', '동생은', '내 동생', '제 동생', '자매',
            '시어머니', '시아버지', '장인', '장모', '처가', '시댁',
            '가족', '집안', '가정'
        ],
        'answer': [
            '아내', '남편', '와이프', '신랑',
            '아들', '딸', '자녀', '막내',
            '첫째', '둘째', '셋째',
            '아버지', '어머니', '아빠', '엄마',
            '형이', '형은', '제 형', '내 형',
            '누나가', '누나는', '제 누나',
            '오빠가', '오빠는', '제 오빠',
            '언니가', '언니는', '제 언니',
            '동생이', '동생은', '제 동생'
        ]
    },
    
    # 개인정보 관련 키워드
    'personal': {
        'question': [
            '나이', '몇살', '몇 살', '연세', '생년', '태어',
            '주소', '사는 곳', '어디 사', '거주', '집이 어디',
            '연락처', '전화번호', '핸드폰', '휴대폰', '번호',
            '이메일', '메일', 'SNS', '인스타', '카톡',
            '학교', '대학', '고등학교', '중학교', '졸업',
            '직장', '회사', '직업', '월급', '연봉', '수입',
            '재산', '집값', '아파트', '전세', '월세',
            '병원', '건강', '수술', '아프', '질병',
            '연애', '사귀', '썸', '이별', '헤어'
        ],
        'answer': [
            '살', '세입니다', '년생',
            '주소', '살아요', '살고',
            '번호', '010', '02', '031', '032',
            '@', '.com', '.net',
            '대학', '졸업',
            '회사', '다녀', '근무',
            '원', '만원', '억',
            '수술', '아파', '병',
            '사귀', '연애', '헤어'
        ]
    },
    
    # 방송 스포일러 관련 키워드
    'spoiler': {
        'question': [
            '탈락', '떨어', '우승', '1등', '2등', '3등',
            '순위', '몇등', '몇 등', '몇위', '몇 위',
            '결과', '결승', '본선', '예선', '진출',
            '합격', '불합격', '통과', '광탈',
            '다음주', '다음 주', '다음화', '다음 화',
            '스포', '미리', '알려', '결말',
            '방송', '녹화', '촬영'
        ],
        'answer': [
            '탈락', '떨어졌', '우승',
            '1등', '2등', '3등', '등했',
            '순위', '위를',
            '결승', '본선', '예선', '진출',
            '합격', '불합격', '통과', '광탈',
            '다음주', '다음 주',
            '스포일러', '미리'
        ]
    }
}

# 호감도 분석용 키워드
SENTIMENT_KEYWORDS = {
    'positive': [
        '응원', '화이팅', '파이팅', '힘내', '대박', '최고',
        '맛있', '예쁘', '멋지', '좋아', '사랑', '감사',
        '대단', '존경', '팬', '행복', '축하', '기대',
        '짱', '굿', '좋은', '훌륭', '완벽', '감동'
    ],
    'negative': [
        '별로', '실망', '싫어', '짜증', '화나', '나쁜',
        '못생', '맛없', '이상해', '왜 그래', '그만',
        '안 좋', '최악', '쓰레기', '꺼져', '나가'
    ]
}

# 질문 유형 분류용 키워드
QUESTION_TYPE_KEYWORDS = {
    'intro': ['자기소개', '소개', '누구', '알려', '말해'],
    'recipe': ['빵', '레시피', '베이킹', '반죽', '발효', '굽', '재료', '만들'],
    'personal': ['결혼', '나이', '가족', '연애', '사생활', '개인'],
    'broadcast': ['방송', '대회', '순위', '탈락', '우승', '녹화', '촬영'],
    'shop': ['가게', '매장', '어디', '영업', '몇시', '주문', '배송', '구매', '살 수']
}


class HybridAnalyzer:
    """키워드 1차 필터링 + GPT 2차 분석 하이브리드 분석기"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = GPT_MODEL
        self.last_call_time = 0
        
        # 통계용
        self.stats = {
            'total_dialogs': 0,
            'keyword_filtered': 0,
            'gpt_analyzed': 0,
            'skipped': 0
        }
    
    def _wait_for_rate_limit(self):
        """Rate limiting을 위한 대기"""
        elapsed = time.time() - self.last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_call_time = time.time()
    
    # ============================================================
    # 1차 분석: 키워드 기반 빠른 분석
    # ============================================================
    
    def _contains_keywords(self, text: str, keywords: list) -> bool:
        """텍스트에 키워드가 포함되어 있는지 확인"""
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return True
        return False
    
    def _detect_risk_keywords(self, question: str, answer: str) -> dict:
        """
        키워드 기반 위험도 1차 감지
        
        핵심 로직: 질문에 위험 키워드가 있고, 답변에도 관련 키워드가 있을 때만 위험 후보
        (질문에만 키워드가 있으면 챗봇이 거부/회피했을 가능성이 높음)
        
        Returns:
            {
                'has_potential_risk': bool,
                'detected_types': ['family', 'personal', 'spoiler'],
                'matched_keywords': {'family': [...], ...}
            }
        """
        result = {
            'has_potential_risk': False,
            'detected_types': [],
            'matched_keywords': {}
        }
        
        for risk_type, keywords in RISK_KEYWORDS.items():
            # 질문에서 위험 키워드 감지
            q_matched = [k for k in keywords['question'] if k in question]
            # 답변에서 위험 키워드 감지
            a_matched = [k for k in keywords['answer'] if k in answer]
            
            # 핵심: 질문과 답변 모두에서 관련 키워드가 있어야 위험 후보
            # (답변에 키워드가 없으면 챗봇이 적절히 거부한 것으로 판단)
            if q_matched and a_matched:
                result['has_potential_risk'] = True
                result['detected_types'].append(risk_type)
                result['matched_keywords'][risk_type] = {
                    'question': q_matched,
                    'answer': a_matched
                }
        
        return result
    
    def _analyze_sentiment_keywords(self, question: str, answer: str) -> dict:
        """키워드 기반 호감도 분석"""
        combined_text = question + " " + answer
        
        positive_count = sum(1 for k in SENTIMENT_KEYWORDS['positive'] 
                           if k in combined_text)
        negative_count = sum(1 for k in SENTIMENT_KEYWORDS['negative'] 
                           if k in combined_text)
        
        # 키워드 추출
        found_keywords = []
        for k in SENTIMENT_KEYWORDS['positive']:
            if k in combined_text:
                found_keywords.append(k)
        for k in SENTIMENT_KEYWORDS['negative']:
            if k in combined_text:
                found_keywords.append(k)
        
        # 감정 판정
        if positive_count > negative_count:
            sentiment = 'positive'
            score = min(5, 3 + positive_count)
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = max(1, 3 - negative_count)
        else:
            sentiment = 'neutral'
            score = 3
        
        return {
            'sentiment': sentiment,
            'score': score,
            'keywords': found_keywords[:5]  # 상위 5개만
        }
    
    def _classify_question_type(self, question: str) -> str:
        """키워드 기반 질문 유형 분류"""
        question_lower = question.lower()
        
        type_scores = {}
        for q_type, keywords in QUESTION_TYPE_KEYWORDS.items():
            score = sum(1 for k in keywords if k in question_lower)
            if score > 0:
                type_scores[q_type] = score
        
        if type_scores:
            return max(type_scores, key=type_scores.get)
        return 'other'
    
    def analyze_keywords_only(self, question: str, answer: str) -> dict:
        """
        키워드만으로 분석 (GPT 호출 없음)
        위험이 감지되지 않은 대화에 사용
        """
        return {
            'favorability': self._analyze_sentiment_keywords(question, answer),
            'question_type': self._classify_question_type(question),
            'risk': {'has_risk': False, 'risks': []},
            'analysis_method': 'keyword_only'
        }
    
    # ============================================================
    # 2차 분석: GPT API 정밀 분석
    # ============================================================
    
    def _get_risk_analysis_prompt(self) -> str:
        """위험도 중심 분석용 프롬프트"""
        return """당신은 TV 프로그램 '천하제빵' 챗봇 대화 분석 전문가입니다.
사용자와 베이커(참가자) 챗봇 간의 대화에서 민감한 정보 노출을 분석합니다.

## 분석 항목

### 1. 호감도 분석 (favorability)
- sentiment: "positive" / "neutral" / "negative"
- score: 1-5점
- keywords: 감정 키워드들

### 2. 질문 유형 (question_type)
- "intro", "recipe", "personal", "broadcast", "shop", "other" 중 하나

### 3. 위험도 분석 (risk) - 가장 중요!
**위험 유형:**
- "family": 가족정보 노출 (배우자, 자녀, 부모 등)
- "personal": 개인사 노출 (결혼, 나이, 연락처 등)
- "spoiler": 방송 스포일러 (탈락, 순위, 결과 등)

**심각도 판정 기준 (매우 중요!):**
- "high": 챗봇이 민감한 정보를 실제로 노출함
- "medium": 질문은 위험하나 챗봇이 모호하게 답변
- "safe": 챗봇이 적절히 거부/회피

**반드시 "safe"로 판정해야 하는 경우:**
- 가게 오픈 여부, 영업시간 안내는 항상 "safe" (공개 정보)
- 매장 주소를 언급하더라도 구체적인 번지(예: 123-45번지, 456호)가 없으면 "safe"
- 예: "강남구에 가게가 있어요", "홍대 근처에서 운영해요" → "safe"
- 예: "서울시 강남구 역삼동 123-45번지입니다" → "medium" 또는 "high"

## 응답 형식

반드시 아래 JSON 형식으로만 응답하세요:
{
    "favorability": {
        "sentiment": "positive" | "neutral" | "negative",
        "score": 1-5,
        "keywords": ["키워드1", "키워드2"]
    },
    "question_type": "intro" | "recipe" | "personal" | "broadcast" | "shop" | "other",
    "risk": {
        "has_risk": true | false,
        "risks": [
            {
                "type": "family" | "personal" | "spoiler",
                "severity": "high" | "medium" | "safe",
                "reason": "판단 근거"
            }
        ]
    }
}"""
    
    def analyze_with_gpt(self, question: str, answer: str, 
                        detected_types: list = None) -> dict:
        """
        GPT API로 정밀 분석 (위험 후보 대화에 사용)
        
        Args:
            question: 사용자 질문
            answer: 챗봇 답변
            detected_types: 1차에서 감지된 위험 유형들 (힌트용)
        """
        self._wait_for_rate_limit()
        
        # 힌트 추가 (선택적)
        hint = ""
        if detected_types:
            type_names = {
                'family': '가족정보',
                'personal': '개인정보',
                'spoiler': '방송 스포일러'
            }
            hints = [type_names.get(t, t) for t in detected_types]
            hint = f"\n\n[참고: 키워드 분석에서 {', '.join(hints)} 관련 위험이 감지됨]"
        
        user_message = f"""다음 대화를 분석하세요:

[사용자 질문]
{question}

[챗봇 답변]
{answer}{hint}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_risk_analysis_prompt()},
                    {"role": "user", "content": user_message}
                ],
                temperature=GPT_TEMPERATURE,
                max_tokens=GPT_MAX_TOKENS,
                timeout=GPT_TIMEOUT,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            result['analysis_method'] = 'gpt'
            return result
            
        except Exception as e:
            print(f"GPT API 오류: {e}")
            return self._default_result()
    
    def _default_result(self):
        """기본 결과 반환 (오류 시)"""
        return {
            "favorability": {"sentiment": "neutral", "score": 3, "keywords": []},
            "question_type": "other",
            "risk": {"has_risk": False, "risks": []},
            "analysis_method": "default"
        }
    
    # ============================================================
    # 통합 분석 메서드
    # ============================================================
    
    def analyze_dialog(self, question: str, answer: str) -> dict:
        """
        단일 대화 하이브리드 분석
        
        1차: 키워드로 위험 감지
        2차: 위험 감지 시 GPT로 정밀 분석
        """
        # 빈 대화 처리
        if not question.strip() and not answer.strip():
            return self._default_result()
        
        # 1차: 키워드 기반 위험 감지
        risk_detection = self._detect_risk_keywords(question, answer)
        
        if risk_detection['has_potential_risk']:
            # 2차: GPT로 정밀 분석
            return self.analyze_with_gpt(
                question, answer, 
                risk_detection['detected_types']
            )
        else:
            # 위험 없음: 키워드 분석만으로 완료
            return self.analyze_keywords_only(question, answer)
    
    def analyze_batch(self, dialogs: list, 
                     progress_callback: Callable = None) -> list:
        """
        여러 대화를 배치로 하이브리드 분석
        
        Args:
            dialogs: [{"dialog_id": str, "question": str, "answer": str, ...}, ...]
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            분석 결과가 추가된 대화 목록
        """
        # 통계 초기화
        self.stats = {
            'total_dialogs': len(dialogs),
            'keyword_filtered': 0,
            'gpt_analyzed': 0,
            'skipped': 0
        }
        
        results = []
        total = len(dialogs)
        
        # 1단계: 키워드 필터링으로 위험 후보 분류
        risk_candidates = []
        safe_dialogs = []
        
        for i, dialog in enumerate(dialogs):
            question = dialog.get("question", "")
            answer = dialog.get("answer", "")
            
            if not question.strip() and not answer.strip():
                # 빈 대화
                result = {**dialog, **self._default_result()}
                safe_dialogs.append((i, result))
                self.stats['skipped'] += 1
            else:
                risk_detection = self._detect_risk_keywords(question, answer)
                if risk_detection['has_potential_risk']:
                    risk_candidates.append((i, dialog, risk_detection))
                else:
                    # 키워드 분석만으로 완료
                    analysis = self.analyze_keywords_only(question, answer)
                    result = {**dialog, **analysis}
                    safe_dialogs.append((i, result))
                    self.stats['keyword_filtered'] += 1
        
        # 진행 상황 출력
        print(f"  [1단계] 키워드 필터링 완료:")
        print(f"    - 전체 대화: {total}개")
        print(f"    - 위험 후보: {len(risk_candidates)}개 (GPT 분석 필요)")
        print(f"    - 안전 대화: {len(safe_dialogs)}개 (키워드 분석 완료)")
        
        # 2단계: 위험 후보만 GPT로 분석
        gpt_results = []
        gpt_total = len(risk_candidates)
        
        if gpt_total > 0:
            print(f"  [2단계] GPT 정밀 분석 시작 ({gpt_total}개)...")
            
            for j, (i, dialog, risk_detection) in enumerate(risk_candidates):
                if progress_callback:
                    progress_callback(j + 1, gpt_total)
                elif (j + 1) % 20 == 0 or j == 0:
                    print(f"    GPT 분석 중... {j+1}/{gpt_total}")
                
                question = dialog.get("question", "")
                answer = dialog.get("answer", "")
                
                analysis = self.analyze_with_gpt(
                    question, answer,
                    risk_detection['detected_types']
                )
                result = {**dialog, **analysis}
                gpt_results.append((i, result))
                self.stats['gpt_analyzed'] += 1
        
        # 3단계: 원래 순서대로 결과 병합
        all_results = safe_dialogs + gpt_results
        all_results.sort(key=lambda x: x[0])
        results = [r[1] for r in all_results]
        
        # 최종 통계 출력
        print(f"  [완료] 분석 통계:")
        print(f"    - 키워드 분석만: {self.stats['keyword_filtered']}개")
        print(f"    - GPT 정밀 분석: {self.stats['gpt_analyzed']}개")
        print(f"    - 빈 대화 스킵: {self.stats['skipped']}개")
        
        return results
    
    def get_stats(self) -> dict:
        """분석 통계 반환"""
        return self.stats


# 테스트용 코드
if __name__ == "__main__":
    analyzer = HybridAnalyzer()
    
    test_cases = [
        # 위험 없음 - 키워드 분석만
        {
            "question": "자기소개 해주세요",
            "answer": "안녕하세요! 저는 빵을 사랑하는 베이커입니다."
        },
        # 위험 후보 - GPT 분석 필요 (가족정보)
        {
            "question": "결혼하셨어요?",
            "answer": "네, 결혼해서 아내와 아들이 있어요."
        },
        # 위험 없음 - 키워드 분석만
        {
            "question": "빵 정말 맛있어 보여요! 응원합니다!",
            "answer": "감사합니다! 더 맛있는 빵 만들도록 노력할게요."
        },
        # 위험 후보 - GPT 분석 필요 (스포일러)
        {
            "question": "우승하셨나요?",
            "answer": "아직 방송 전이라 말씀드리기 어려워요."
        },
        # 위험 후보 - GPT 분석 필요 (개인정보)
        {
            "question": "나이가 어떻게 되세요?",
            "answer": "올해 45살입니다."
        }
    ]
    
    print("=" * 60)
    print("하이브리드 분석기 테스트 (키워드 1차 + GPT 2차)")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}]")
        print(f"질문: {case['question']}")
        print(f"답변: {case['answer']}")
        
        # 키워드 감지 테스트
        risk_detection = analyzer._detect_risk_keywords(
            case['question'], case['answer']
        )
        print(f"키워드 감지: {risk_detection}")
        
        # 전체 분석
        result = analyzer.analyze_dialog(case["question"], case["answer"])
        print(f"분석 방법: {result.get('analysis_method', 'unknown')}")
        print(f"결과: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    print("\n" + "=" * 60)
    print("배치 분석 테스트")
    print("=" * 60)
    
    batch_results = analyzer.analyze_batch(test_cases)
    print(f"\n최종 통계: {analyzer.get_stats()}")
