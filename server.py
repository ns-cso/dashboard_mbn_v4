#!/usr/bin/env python3
"""
mbn-analytics v4.5 대시보드 서버 (Railway 배포용)

API 엔드포인트:
- GET /api/analyze?mode=full&start=2026-01-13&end=2026-01-19  - 전체 재분석
- GET /api/analyze?mode=incremental                           - 증분 분석
- GET /api/status  - 분석 상태 확인
- GET /api/sync    - 동기화 정보 확인
- POST /api/upload - CSV 파일 업로드 (중복 제거 후 병합 + 자동 재분석)
- GET /api/report/frame    - 리포트 프레임 데이터 (분석 없이)
- GET /api/report/generate - 리포트 생성 (GPT 분석 포함)
- GET /api/report/status   - 리포트 생성 진행 상태
"""
import http.server
import socketserver
import os
import json
import subprocess
import threading
import time
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
from urllib.parse import urlparse, parse_qs
from config import OUTPUT_DIR, DASHBOARD_DATA_JSON, LAST_SYNC_JSON
from report_generator import ReportGenerator, report_status, generate_report_async, REPORT_DATA_JSON

PORT = int(os.environ.get('PORT', 3002))
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Python 실행 경로 찾기
def find_python():
    """사용 가능한 Python 경로 찾기"""
    import sys
    candidates = [
        sys.executable,
        os.path.join(DIRECTORY, '..', 'venv', 'bin', 'python'),
        '/usr/bin/python3',
        'python3',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return 'python3'

PYTHON_PATH = find_python()
DATA_DIR = os.path.join(DIRECTORY, 'data')

# CSV 테이블명과 primary key 매핑
CSV_TABLE_CONFIG = {
    'chatdialog': 'did',
    'chatsession': 'sid',
    'anonymous_user': 'uid',
    'anonymous_user_survey': 'id',
    'chatbot': 'bid',
}

# 분석 진행 상태
analysis_status = {
    'running': False,
    'progress': '',
    'error': None,
    'mode': None
}

# 데이터 갱신 상태
scheduler_status = {
    'data_updated_at': None  # 데이터가 마지막으로 갱신된 시각 (대시보드 자동 갱신용)
}


def merge_csv_data(table_name, new_csv_content):
    """
    업로드된 CSV 데이터를 기존 data/에 중복 제거 후 병합.
    Returns: (new_rows_count: int, duplicate_count: int)
    """
    import csv
    import io

    target_file = os.path.join(DATA_DIR, f"{table_name}.csv")

    # 새 CSV 파싱
    reader = csv.reader(io.StringIO(new_csv_content))
    header = next(reader, None)
    if not header:
        return 0, 0
    new_rows = list(reader)

    if not new_rows:
        return 0, 0

    # 기존 데이터에서 primary key (첫 번째 컬럼) 수집
    existing_ids = set()
    file_exists = os.path.exists(target_file)
    if file_exists:
        with open(target_file, 'r', encoding='utf-8') as f:
            existing_reader = csv.reader(f)
            existing_header = next(existing_reader, None)
            for row in existing_reader:
                if row:
                    existing_ids.add(row[0])

    # 중복 제거
    unique_rows = [row for row in new_rows if row and row[0] not in existing_ids]
    duplicate_count = len(new_rows) - len(unique_rows)

    if not unique_rows:
        return 0, duplicate_count

    # target 파일에 append (없으면 생성)
    with open(target_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(unique_rows)

    return len(unique_rows), duplicate_count


def run_analysis_after_upload(results_summary):
    """CSV 업로드 후 전체 재분석 실행"""
    global analysis_status

    analysis_status['running'] = True
    analysis_status['progress'] = '업로드 데이터 분석 중...'
    analysis_status['error'] = None
    analysis_status['mode'] = 'upload'

    try:
        process = subprocess.Popen(
            [PYTHON_PATH, 'data_processor.py', 'incremental'],
            cwd=DIRECTORY,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[분석] {line.strip()}")

        process.wait()

        if process.returncode != 0:
            analysis_status['error'] = f"분석 실패 (코드: {process.returncode})"
            analysis_status['progress'] = '분석 실패'
        else:
            analysis_status['progress'] = f'업로드 완료 ({results_summary})'
            scheduler_status['data_updated_at'] = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[업로드] 분석 완료! {results_summary}")

    except Exception as e:
        analysis_status['error'] = str(e)
        analysis_status['progress'] = f'오류: {e}'
        print(f"[업로드] 분석 오류: {e}")
    finally:
        analysis_status['running'] = False
        analysis_status['mode'] = None


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/':
            self.path = '/dashboard.html'
            return super().do_GET()
        
        elif parsed.path == '/dashboard_data.json':
            output_file = DASHBOARD_DATA_JSON
            if os.path.exists(output_file):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with open(output_file, 'rb') as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': '분석 데이터가 없습니다. 분석을 먼저 실행하세요.'}).encode())
                return
        
        elif parsed.path == '/api/ip':
            # Railway 서버의 아웃바운드 IP 확인용
            import urllib.request
            try:
                ip = urllib.request.urlopen('https://api.ipify.org', timeout=10).read().decode()
            except Exception as e:
                ip = f'error: {e}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'outbound_ip': ip}).encode())
            return
        
        elif parsed.path == '/api/status':
            status = dict(analysis_status)
            status['scheduler'] = scheduler_status
            
            progress_file = os.path.join(DIRECTORY, 'progress.json')
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress = json.load(f)
                        status['detail'] = progress
                except:
                    pass
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(status, ensure_ascii=False).encode())
            return
        
        elif parsed.path == '/api/sync':
            sync_info = {'last_dialog_id': None, 'analyzed_at': None, 'total_analyzed': 0}
            if os.path.exists(LAST_SYNC_JSON):
                try:
                    with open(LAST_SYNC_JSON, 'r', encoding='utf-8') as f:
                        sync_info = json.load(f)
                except:
                    pass
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(sync_info, ensure_ascii=False).encode())
            return
        
        elif parsed.path == '/api/analyze':
            params = parse_qs(parsed.query)
            mode = params.get('mode', ['full'])[0]
            start_date = params.get('start', [None])[0]
            end_date = params.get('end', [None])[0]
            
            if analysis_status['running']:
                self.send_response(409)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': '분석이 이미 진행 중입니다.'}).encode())
                return
            
            thread = threading.Thread(target=run_analysis, args=(mode, start_date, end_date))
            thread.start()
            
            mode_names = {'full': '전체 재분석', 'incremental': '증분 분석', 'reanalyze': '기간 재분석'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'message': f'{mode_names.get(mode, mode)} 시작됨',
                'mode': mode,
                'start_date': start_date,
                'end_date': end_date
            }).encode())
            return
        
        elif parsed.path == '/api/filter':
            params = parse_qs(parsed.query)
            start_date = params.get('start', [None])[0]
            end_date = params.get('end', [None])[0]
            
            if not start_date or not end_date:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'start와 end 파라미터가 필요합니다.'}).encode())
                return
            
            try:
                from data_processor import filter_existing_data
                filtered_data = filter_existing_data(start_date, end_date)
                
                if filtered_data:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(filtered_data, ensure_ascii=False).encode())
                else:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': '해당 기간에 데이터가 없습니다.'}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return
        
        elif parsed.path == '/api/report/frame':
            params = parse_qs(parsed.query)
            start_date = params.get('start', [None])[0]
            end_date = params.get('end', [None])[0]
            
            if start_date and end_date:
                try:
                    from data_processor import filter_existing_data
                    filtered_data = filter_existing_data(start_date, end_date)

                    # 기존 리포트 데이터가 있으면 로드
                    existing_report = None
                    if os.path.exists(REPORT_DATA_JSON):
                        try:
                            with open(REPORT_DATA_JSON, 'r', encoding='utf-8') as f:
                                existing_report = json.load(f)
                        except:
                            pass

                    if filtered_data:
                        frame_data = {
                            'dashboard_data': filtered_data,
                            'report_data': existing_report
                        }
                    else:
                        frame_data = {'dashboard_data': None, 'report_data': existing_report}
                except Exception as e:
                    frame_data = {'dashboard_data': None, 'report_data': None, 'error': str(e)}
            else:
                generator = ReportGenerator()
                frame_data = generator.get_frame_data()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(frame_data, ensure_ascii=False).encode())
            return
        
        elif parsed.path == '/api/report/generate':
            if report_status['running']:
                self.send_response(409)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': '리포트 생성이 이미 진행 중입니다.'}).encode())
                return

            params = parse_qs(parsed.query)
            start_date = params.get('start', [None])[0]
            end_date = params.get('end', [None])[0]

            thread = threading.Thread(target=generate_report_async, args=(start_date, end_date))
            thread.start()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'message': '리포트 생성 시작됨'}).encode())
            return
        
        elif parsed.path == '/api/report/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(report_status, ensure_ascii=False).encode())
            return
        
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/upload':
            self._handle_csv_upload()
            return

        self.send_response(405)
        self.end_headers()

    def _handle_csv_upload(self):
        """CSV 파일 업로드 처리 (multipart/form-data) - cgi 모듈 없이 수동 파싱"""

        if analysis_status['running']:
            self.send_response(409)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': '분석이 이미 진행 중입니다.'}).encode())
            return

        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'multipart/form-data 형식이어야 합니다.'}).encode())
            return

        try:
            # boundary 추출
            boundary = None
            for part in content_type.split(';'):
                part = part.strip()
                if part.startswith('boundary='):
                    boundary = part[len('boundary='):]
                    break

            if not boundary:
                raise ValueError('boundary not found in Content-Type')

            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            # multipart 파싱
            boundary_bytes = ('--' + boundary).encode()
            parts = body.split(boundary_bytes)

            results = []
            total_new = 0
            total_dup = 0

            for part in parts:
                # 빈 파트나 종료 마커 건너뛰기
                if not part or part.strip() == b'--' or part.strip() == b'':
                    continue

                # 헤더와 본문 분리 (빈 줄로 구분)
                if b'\r\n\r\n' in part:
                    header_section, file_data = part.split(b'\r\n\r\n', 1)
                elif b'\n\n' in part:
                    header_section, file_data = part.split(b'\n\n', 1)
                else:
                    continue

                # 후행 CRLF 제거
                if file_data.endswith(b'\r\n'):
                    file_data = file_data[:-2]

                header_text = header_section.decode('utf-8', errors='replace')

                # filename 추출
                filename = None
                for line in header_text.split('\n'):
                    if 'filename=' in line:
                        # filename="xxx.csv" 또는 filename=xxx.csv
                        import re
                        m = re.search(r'filename="?([^";\r\n]+)"?', line)
                        if m:
                            filename = m.group(1).strip()
                            break

                if not filename:
                    continue

                filename = os.path.basename(filename)

                # 테이블명 매칭
                table_name = None
                for name in CSV_TABLE_CONFIG:
                    if filename.startswith(name):
                        table_name = name
                        break

                if not table_name:
                    results.append({
                        'file': filename,
                        'status': 'skipped',
                        'message': f'알 수 없는 테이블: {filename} (지원: {", ".join(CSV_TABLE_CONFIG.keys())})'
                    })
                    continue

                csv_content = file_data.decode('utf-8')
                new_count, dup_count = merge_csv_data(table_name, csv_content)
                total_new += new_count
                total_dup += dup_count
                results.append({
                    'file': filename,
                    'table': table_name,
                    'status': 'ok',
                    'new_rows': new_count,
                    'duplicates': dup_count,
                })
                print(f"[업로드] {filename} -> {table_name}: {new_count}행 추가, {dup_count}행 중복 제외")

            response = {
                'message': f'업로드 완료: {total_new}행 추가, {total_dup}행 중복 제외',
                'results': results,
                'total_new': total_new,
                'total_duplicates': total_dup,
                'analyzing': total_new > 0,
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode())

            # 새 데이터가 있으면 백그라운드 재분석
            if total_new > 0:
                summary = f"{total_new}행 추가"
                thread = threading.Thread(target=run_analysis_after_upload, args=(summary,))
                thread.start()

        except Exception as e:
            print(f"[업로드] 오류: {e}")
            import traceback
            traceback.print_exc()
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())


def run_analysis(mode, start_date, end_date):
    """백그라운드에서 분석 실행"""
    global analysis_status
    analysis_status['running'] = True
    analysis_status['mode'] = mode
    
    mode_names = {
        'full': '전체 재분석',
        'incremental': '증분 분석',
        'reanalyze': '기간 재분석'
    }
    mode_name = mode_names.get(mode, mode)
    
    analysis_status['progress'] = f'{mode_name} 시작...'
    analysis_status['error'] = None
    
    try:
        cmd = [PYTHON_PATH, 'data_processor.py', mode]
        if start_date and mode in ['full', 'reanalyze']:
            cmd.append(start_date)
        if end_date and mode in ['full', 'reanalyze']:
            cmd.append(end_date)
        
        analysis_status['progress'] = f'{mode_name} 중...'
        
        result = subprocess.run(
            cmd,
            cwd=DIRECTORY,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode == 0:
            analysis_status['progress'] = '분석 완료!'
            scheduler_status['data_updated_at'] = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')
            print(f"분석 완료: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
        else:
            analysis_status['error'] = result.stderr[:500] if result.stderr else result.stdout[:500]
            analysis_status['progress'] = '분석 실패'
            print(f"분석 실패: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        analysis_status['error'] = '분석 시간 초과 (1시간)'
        analysis_status['progress'] = '타임아웃'
    except Exception as e:
        analysis_status['error'] = str(e)
        analysis_status['progress'] = '오류 발생'
        print(f"분석 오류: {e}")
    finally:
        analysis_status['running'] = False


def initial_data_load():
    """서버 시작 시 분석 데이터가 없으면 자동으로 전체 분석 실행"""
    if os.path.exists(DASHBOARD_DATA_JSON):
        print(f"[초기화] 기존 분석 데이터 발견, 초기 로드 건너뜀")
        return

    print(f"[초기화] 분석 데이터 없음 — 전체 분석 시작...")
    try:
        run_analysis('full', None, None)
        print(f"[초기화] 전체 분석 완료!")
    except Exception as e:
        print(f"[초기화] 전체 분석 실패: {e}")


if __name__ == "__main__":
    os.chdir(DIRECTORY)
    
    # output 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 서버 시작 시 분석 데이터가 없으면 자동으로 전체 분석 (백그라운드)
    init_thread = threading.Thread(target=initial_data_load, daemon=True)
    init_thread.start()

    # TCPServer 재사용 허용
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"=" * 60)
        print(f"천하제빵 AI Baker 대시보드 서버 v4.5 (Railway)")
        print(f"=" * 60)
        print(f"URL: http://localhost:{PORT}")
        print(f"")
        print(f"API 엔드포인트:")
        print(f"  POST /api/upload                  - CSV 파일 업로드")
        print(f"  GET  /api/analyze?mode=full       - 전체 재분석")
        print(f"  GET  /api/status                  - 분석 상태 확인")
        print(f"  GET  /api/report/frame            - 리포트 프레임 데이터")
        print(f"  GET  /api/report/generate         - 리포트 생성 (GPT)")
        print(f"=" * 60)
        httpd.serve_forever()
