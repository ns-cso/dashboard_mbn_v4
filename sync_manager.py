"""
동기화 관리 모듈
- last_sync.json 관리
- 증분 데이터 필터링
- 분석 결과 병합
"""
import json
import os
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
from typing import Optional, List, Dict, Any
from config import LAST_SYNC_JSON, ANALYZED_DIALOGS_JSON, OUTPUT_DIR


class SyncManager:
    """동기화 상태 관리"""
    
    def __init__(self):
        self.sync_file = LAST_SYNC_JSON
        self.analyzed_file = ANALYZED_DIALOGS_JSON
        
        # output 디렉토리 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def get_sync_info(self) -> dict:
        """현재 동기화 정보 조회"""
        if os.path.exists(self.sync_file):
            with open(self.sync_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'last_dialog_id': None,
            'last_created_at': None,
            'analyzed_at': None,
            'total_analyzed': 0,
            'analysis_type': None
        }
    
    def update_sync_info(self, last_dialog_id: str, last_created_at: str, 
                        total_analyzed: int, analysis_type: str):
        """동기화 정보 업데이트"""
        sync_info = {
            'last_dialog_id': last_dialog_id,
            'last_created_at': last_created_at,
            'analyzed_at': datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S'),
            'total_analyzed': total_analyzed,
            'analysis_type': analysis_type  # 'full' or 'incremental'
        }
        
        with open(self.sync_file, 'w', encoding='utf-8') as f:
            json.dump(sync_info, f, ensure_ascii=False, indent=2)
        
        return sync_info
    
    def filter_new_dialogs(self, dialogs: List[dict]) -> List[dict]:
        """
        새로운 대화만 필터링 (증분 분석용)
        
        last_dialog_id 이후의 대화만 반환
        """
        sync_info = self.get_sync_info()
        last_created_at = sync_info.get('last_created_at')
        
        if not last_created_at:
            # 첫 분석인 경우 전체 반환
            return dialogs
        
        # created_at 기준으로 필터링 (더 확실한 방법)
        new_dialogs = [d for d in dialogs 
                       if d.get('created_at', '') > last_created_at 
                       and d.get('dialog_id')]
        
        return new_dialogs
    
    def save_analyzed_dialogs(self, analyzed_dialogs: List[dict]):
        """분석된 대화 저장 (캐시)"""
        with open(self.analyzed_file, 'w', encoding='utf-8') as f:
            json.dump(analyzed_dialogs, f, ensure_ascii=False)
    
    def load_analyzed_dialogs(self) -> List[dict]:
        """저장된 분석 대화 로드"""
        if os.path.exists(self.analyzed_file):
            with open(self.analyzed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def merge_analyzed_dialogs(self, existing: List[dict], new: List[dict]) -> List[dict]:
        """
        기존 분석 결과와 새 분석 결과 병합
        
        dialog_id 기준 중복 제거
        """
        # 기존 dialog_id 집합
        existing_ids = {d['dialog_id'] for d in existing}
        
        # 새 대화 중 중복 제거
        unique_new = [d for d in new if d['dialog_id'] not in existing_ids]
        
        # 병합 (기존 + 새 대화)
        merged = existing + unique_new
        
        # created_at 기준 정렬
        merged.sort(key=lambda x: x.get('created_at', ''))
        
        return merged
    
    def reset(self):
        """동기화 정보 초기화 (전체 재분석 전)"""
        if os.path.exists(self.sync_file):
            os.remove(self.sync_file)
        if os.path.exists(self.analyzed_file):
            os.remove(self.analyzed_file)


# 싱글톤 인스턴스
sync_manager = SyncManager()
