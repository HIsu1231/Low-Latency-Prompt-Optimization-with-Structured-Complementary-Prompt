import os
import json
import time
import random
import argparse
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from openai import OpenAI

# --- 설정 ---
load_dotenv() 

client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)
MODEL_NAME   = "gpt-4o"
MAX_RETRIES  = 6
BACKOFF_BASE = 2.0

SYSTEM_PROMPT = '''Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.'''

USER_TEMPLATE = (
    "[User Question]\n{input}\n\n"
    "[The Start of Assistant A's Answer]\n{output_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{output_b}\n[The End of Assistant B's Answer]\n"
)

def load_list(path):
    """JSONL 파일을 리스트로 로드"""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            items.append({
                "input": obj["input"].strip(),
                "output": obj["output"].strip()
            })
    return items

def create_batch_requests(list_a, list_b, batch_input_path):
    """배치 API 요청 파일 생성 및 순서 정보 반환"""
    print("배치 요청 파일을 생성중입니다...")
    
    swap_info = []  # 각 요청의 swap 정보를 저장
    
    with open(batch_input_path, 'w', encoding='utf-8') as f:
        for idx, (a, b) in enumerate(zip(list_a, list_b)):
            inp = a["input"]
            out_a = a["output"]
            out_b = b["output"]
            
            # 랜덤하게 순서 바꾸기
            swapped = random.random() < 0.5
            if swapped:
                out_a, out_b = out_b, out_a
            
            swap_info.append(swapped)
            
            # 배치 API 요청 형식
            batch_request = {
                "custom_id": f"request_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(
                            input=inp,
                            output_a=out_a,
                            output_b=out_b
                        )}
                    ],
                    "temperature": 0.0
                }
            }
            
            f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
    
    print(f"배치 요청 파일을 {batch_input_path}에 저장했습니다.")
    return batch_input_path, swap_info

def submit_batch_job(batch_input_path, max_retries=6, wait_base=2, max_wait=60):
    """배치 작업 제출"""
    print("배치 파일을 업로드중입니다...")
    
    # 파일 업로드 - 최대 6번 재시도
    batch_input_file = None
    for attempt in range(max_retries):
        try:
            with open(batch_input_path, "rb") as f:
                batch_input_file = client.files.create(
                    file=f,
                    purpose="batch"
                )
            print(f"배치 파일 업로드 완료: {batch_input_file.id}")
            break
        except Exception as e:
            print(f"[파일 업로드 시도 {attempt + 1}/{max_retries}] 오류: {repr(e)}")
            
            if attempt < max_retries - 1:  # 마지막 시도가 아닌 경우에만 대기
                wait_time = min(wait_base * (2 ** attempt), max_wait)
                wait_time = wait_time + random.uniform(0, wait_time * 0.1)
                
                print(f"{wait_time:.2f}초 후 재시도합니다...")
                time.sleep(wait_time)
            else:
                print("파일 업로드 최대 재시도 횟수를 초과했습니다.")
                return None
    
    # 배치 작업 생성 - 최대 6번 재시도
    for attempt in range(max_retries):
        try:
            batch_job = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "GPT scoring evaluation"}
            )
            print(f"배치 작업이 제출되었습니다. Job ID: {batch_job.id}")
            return batch_job
        except Exception as e:
            print(f"[배치 작업 생성 시도 {attempt + 1}/{max_retries}] 오류: {repr(e)}")
            
            if attempt < max_retries - 1:  # 마지막 시도가 아닌 경우에만 대기
                wait_time = min(wait_base * (2 ** attempt), max_wait)
                wait_time = wait_time + random.uniform(0, wait_time * 0.1)
                
                print(f"{wait_time:.2f}초 후 재시도합니다...")
                time.sleep(wait_time)
            else:
                print("배치 작업 생성 최대 재시도 횟수를 초과했습니다.")
                return None

def wait_for_batch_completion(batch_job_id, max_retries=6, wait_base=2, max_wait=60):
    """배치 작업 완료 대기"""
    print("배치 작업 완료를 기다리는 중입니다...")
    
    while True:
        # 배치 상태 조회 - 최대 6번 재시도
        batch_job = None
        for attempt in range(max_retries):
            try:
                batch_job = client.batches.retrieve(batch_job_id)
                break
            except Exception as e:
                print(f"[배치 상태 조회 시도 {attempt + 1}/{max_retries}] 오류: {repr(e)}")
                
                if attempt < max_retries - 1:  # 마지막 시도가 아닌 경우에만 대기
                    wait_time = min(wait_base * (2 ** attempt), max_wait)
                    wait_time = wait_time + random.uniform(0, wait_time * 0.1)
                    
                    print(f"{wait_time:.2f}초 후 재시도합니다...")
                    time.sleep(wait_time)
                else:
                    print("배치 상태 조회 최대 재시도 횟수를 초과했습니다.")
                    return None
        
        if batch_job is None:
            print("배치 상태 조회에 실패했습니다.")
            return None
        
        # 진행상황 표시
        if hasattr(batch_job, 'request_counts') and batch_job.request_counts:
            total = batch_job.request_counts.total
            completed = batch_job.request_counts.completed
            failed = batch_job.request_counts.failed
            
            progress_pct = (completed / total * 100) if total > 0 else 0
            
            print(f"상태: {batch_job.status} | 진행률: {completed}/{total} ({progress_pct:.1f}%) | 실패: {failed}")
        else:
            print(f"상태: {batch_job.status}")
        
        if batch_job.status == "completed":
            print("배치 작업이 완료되었습니다!")
            
            # 최종 통계 출력
            if hasattr(batch_job, 'request_counts') and batch_job.request_counts:
                total = batch_job.request_counts.total
                completed = batch_job.request_counts.completed
                failed = batch_job.request_counts.failed
                print(f"최종 결과: 총 {total}개 중 {completed}개 완료, {failed}개 실패")
            
            return batch_job.output_file_id
            
        elif batch_job.status == "failed":
            print("배치 작업이 실패했습니다.")
            if hasattr(batch_job, 'request_counts') and batch_job.request_counts:
                total = batch_job.request_counts.total
                completed = batch_job.request_counts.completed
                failed = batch_job.request_counts.failed
                print(f"실패 시점 상황: 총 {total}개 중 {completed}개 완료, {failed}개 실패")
            return None
            
        elif batch_job.status in ["cancelled", "expired"]:
            print(f"배치 작업이 {batch_job.status} 상태입니다.")
            if hasattr(batch_job, 'request_counts') and batch_job.request_counts:
                total = batch_job.request_counts.total
                completed = batch_job.request_counts.completed
                failed = batch_job.request_counts.failed
                print(f"{batch_job.status} 시점 상황: 총 {total}개 중 {completed}개 완료, {failed}개 실패")
            return None
        
        # 30초마다 상태 확인
        time.sleep(30)

def download_and_process_results(output_file_id, list_a, list_b, swap_info, final_output_path, max_retries=6, wait_base=2, max_wait=60):
    """결과 파일 다운로드 및 처리"""
    print("결과 파일을 다운로드하고 처리중입니다...")
    
    # 결과 파일 다운로드 - 최대 6번 재시도
    batch_results = None
    for attempt in range(max_retries):
        try:
            result = client.files.content(output_file_id)
            batch_results = result.content.decode('utf-8')
            break
        except Exception as e:
            print(f"[결과 파일 다운로드 시도 {attempt + 1}/{max_retries}] 오류: {repr(e)}")
            
            if attempt < max_retries - 1:  # 마지막 시도가 아닌 경우에만 대기
                wait_time = min(wait_base * (2 ** attempt), max_wait)
                wait_time = wait_time + random.uniform(0, wait_time * 0.1)
                
                print(f"{wait_time:.2f}초 후 재시도합니다...")
                time.sleep(wait_time)
            else:
                print("결과 파일 다운로드 최대 재시도 횟수를 초과했습니다.")
                return None
    
    if batch_results is None:
        print("결과 파일 다운로드에 실패했습니다.")
        return None
    
    # 결과 파싱 및 통계 수집
    results_by_id = {}
    success_count = 0
    error_count = 0
    
    for line in batch_results.strip().split('\n'):
        if line:
            result_obj = json.loads(line)
            custom_id = result_obj['custom_id']
            if result_obj['response']['status_code'] == 200:
                verdict = result_obj['response']['body']['choices'][0]['message']['content'].strip()
                results_by_id[custom_id] = verdict
                success_count += 1
            else:
                results_by_id[custom_id] = f"ERROR: {result_obj['response']['status_code']}"
                error_count += 1
    
    print(f"배치 결과: 성공 {success_count}개, 오류 {error_count}개")
    
    # 최종 결과 파일 생성
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    
    verdict_stats = {"[[A]]": 0, "[[B]]": 0, "[[C]]": 0, "ERROR": 0, "OTHER": 0}
    
    with open(final_output_path, 'w', encoding='utf-8') as fout:
        for idx, (a, b) in enumerate(zip(list_a, list_b)):
            custom_id = f"request_{idx}"
            verdict = results_by_id.get(custom_id, "ERROR: Result not found")
            
            # 판정 결과 통계 (swap 고려)
            swapped = swap_info[idx]
            if "[[A]]" in verdict:
                if swapped:
                    verdict_stats["[[B]]"] += 1  # swap된 경우 A는 실제로 B
                else:
                    verdict_stats["[[A]]"] += 1  # 원래 순서면 A가 A
            elif "[[B]]" in verdict:
                if swapped:
                    verdict_stats["[[A]]"] += 1  # swap된 경우 B는 실제로 A  
                else:
                    verdict_stats["[[B]]"] += 1  # 원래 순서면 B가 B
            elif "[[C]]" in verdict:
                verdict_stats["[[C]]"] += 1      # 무승부는 순서 관계없음
            elif "ERROR" in verdict:
                verdict_stats["ERROR"] += 1
            else:
                verdict_stats["OTHER"] += 1
            
            inp = a["input"]
            out_a = a["output"]
            out_b = b["output"]
            
            result = {
                "input": inp,
                "output_a": out_a,
                "output_b": out_b,
                "swapped": swapped,
                "verdict": verdict
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"최종 결과가 {final_output_path}에 저장되었습니다.")
    
    # 판정 결과 통계를 txt 파일로 저장
    base_path = os.path.splitext(final_output_path)[0]  # 확장자 제거
    stats_file_path = f"{base_path}_stats.txt"
    
    # 디렉토리 생성 (존재하지 않는 경우)
    os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)
    
    # 통계 내용 생성
    stats_content = []
    stats_content.append("="*50)
    stats_content.append("평가 결과 통계:")
    stats_content.append(f"  Assistant A 승리: {verdict_stats['[[A]]']/len(list_a)*100:.1f}% ({verdict_stats['[[A]]']:3d})")
    stats_content.append(f"  무승부 (Tie):      {verdict_stats['[[C]]']/len(list_a)*100:.1f}% ({verdict_stats['[[C]]']:3d})")
    stats_content.append(f"  Assistant B 승리: {verdict_stats['[[B]]']/len(list_a)*100:.1f}% ({verdict_stats['[[B]]']:3d})")
    stats_content.append(f"  오류:           {verdict_stats['ERROR']:3d}개 ({verdict_stats['ERROR']/len(list_a)*100:.1f}%)")
    if verdict_stats['OTHER'] > 0:
        stats_content.append(f"  기타:           {verdict_stats['OTHER']:3d}개 ({verdict_stats['OTHER']/len(list_a)*100:.1f}%)")
    stats_content.append("="*50)
    
    try:
        with open(stats_file_path, 'w', encoding='utf-8') as stats_file:
            for line in stats_content:
                stats_file.write(line + '\n')
                print(line)
        print(f"통계 결과가 {stats_file_path}에 저장되었습니다.")
    except Exception as e:
        print(f"통계 파일 저장 중 오류 발생: {e}")
        print(f"시도한 경로: {stats_file_path}")
        # 콘솔에라도 출력
        for line in stats_content:
            print(line)
    
    return verdict_stats

def analyze_results(output_path):
    """결과 파일 분석하여 통계 출력"""
    if not os.path.exists(output_path):
        print(f"결과 파일이 존재하지 않습니다: {output_path}")
        return
    
    verdict_stats = {"[[A]]": 0, "[[B]]": 0, "[[C]]": 0, "ERROR": 0, "OTHER": 0}
    total_count = 0
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                verdict = result.get('verdict', '')
                swapped = result.get('swapped', False)
                total_count += 1
                
                # 판정 결과 통계 (swap 고려)
                if "[[A]]" in verdict:
                    if swapped:
                        verdict_stats["[[B]]"] += 1  # swap된 경우 A는 실제로 B
                    else:
                        verdict_stats["[[A]]"] += 1  # 원래 순서면 A가 A
                elif "[[B]]" in verdict:
                    if swapped:
                        verdict_stats["[[A]]"] += 1  # swap된 경우 B는 실제로 A
                    else:
                        verdict_stats["[[B]]"] += 1  # 원래 순서면 B가 B
                elif "[[C]]" in verdict:
                    verdict_stats["[[C]]"] += 1      # 무승부는 순서 관계없음
                elif "ERROR" in verdict:
                    verdict_stats["ERROR"] += 1
                else:
                    verdict_stats["OTHER"] += 1
    
    # 분석 결과를 txt 파일로 저장
    base_path = os.path.splitext(output_path)[0]  # 확장자 제거
    analysis_file_path = f"{base_path}_analysis.txt"
    
    # 디렉토리 생성 (존재하지 않는 경우)
    os.makedirs(os.path.dirname(analysis_file_path), exist_ok=True)
    
    # 분석 내용 생성
    analysis_content = []
    analysis_content.append("="*60)
    analysis_content.append(f"결과 파일 분석: {output_path}")
    analysis_content.append(f"총 평가 건수: {total_count}")
    analysis_content.append("-"*60)
    analysis_content.append("평가 결과 통계:")
    analysis_content.append(f"  Assistant A 승리: {verdict_stats['[[A]]']/total_count*100:.1f}% ({verdict_stats['[[A]]']:3d})")
    analysis_content.append(f"  무승부 (Tie):     {verdict_stats['[[C]]']/total_count*100:.1f}% ({verdict_stats['[[C]]']:3d})")
    analysis_content.append(f"  Assistant B 승리: {verdict_stats['[[B]]']/total_count*100:.1f}% ({verdict_stats['[[B]]']:3d})")
    analysis_content.append(f"  오류:           {verdict_stats['ERROR']:3d}개 ({verdict_stats['ERROR']/total_count*100:.1f}%)")
    if verdict_stats['OTHER'] > 0:
        analysis_content.append(f"  기타:           {verdict_stats['OTHER']:3d}개 ({verdict_stats['OTHER']/total_count*100:.1f}%)")
    analysis_content.append("="*60)
    
    try:
        with open(analysis_file_path, 'w', encoding='utf-8') as analysis_file:
            for line in analysis_content:
                analysis_file.write(line + '\n')
                print(line)
        print(f"분석 결과가 {analysis_file_path}에 저장되었습니다.")
    except Exception as e:
        print(f"분석 파일 저장 중 오류 발생: {e}")
        print(f"시도한 경로: {analysis_file_path}")
        # 콘솔에라도 출력
        for line in analysis_content:
            print(line)
    
    return verdict_stats

def main(input_a_path, input_b_path, output_path):
    list_a = load_list(input_a_path)
    list_b = load_list(input_b_path)

    assert len(list_a) == len(list_b), f"Mismatch in length: A={len(list_a)}, B={len(list_b)}"
    print(f"Found {len(list_a)} input pairs for evaluation.")

    # 배치 요청 파일 경로 (확장자에 관계없이 올바른 임시 파일명 생성)
    base_name = os.path.splitext(output_path)[0]
    batch_input_path = f"{base_name}_batch_input.jsonl"
    
    # 1. 배치 요청 파일 생성
    batch_input_path, swap_info = create_batch_requests(list_a, list_b, batch_input_path)
    
    # 2. 배치 작업 제출
    batch_job = submit_batch_job(batch_input_path)
    
    if batch_job: # submit_batch_job가 None을 반환하면 실패
        # 3. 배치 작업 완료 대기
        output_file_id = wait_for_batch_completion(batch_job.id)
        
        if output_file_id:
            # 4. 결과 처리
            stats = download_and_process_results(output_file_id, list_a, list_b, swap_info, output_path)
            if stats is not None:
                print(f"Done. Saved results to {output_path}")
                print(f"Total evaluated pairs: {len(list_a)}")
            else:
                print("결과 파일 다운로드 및 처리 중 오류가 발생했습니다.")
        else:
            print("배치 작업이 실패했습니다.")
    else:
        print("배치 작업 제출 중 오류가 발생했습니다.")
    
    # 임시 파일 정리
    if os.path.exists(batch_input_path):
        os.remove(batch_input_path)
        print(f"임시 파일 {batch_input_path}를 삭제했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_a", "-a", required=True,
                        help="JSONL file of assistant A (each line: {'input','output'})")
    parser.add_argument("--input_b", "-b", required=True,
                        help="JSONL file of assistant B (each line: {'input','output'})")
    parser.add_argument("--output", "-o", required=True,
                        help="where to write JSONL results")
    parser.add_argument("--analyze", action="store_true",
                        help="Only analyze existing results file (skip evaluation)")
    args = parser.parse_args()
    
    if args.analyze:
        # 기존 결과 파일만 분석
        analyze_results(args.output)
    else:
        # 평가 실행
        main(args.input_a, args.input_b, args.output)

# 사용 예시:
# python scoring_gpt.py -a our_data.jsonl -b nonoutdata.jsonl -o output.jsonl

