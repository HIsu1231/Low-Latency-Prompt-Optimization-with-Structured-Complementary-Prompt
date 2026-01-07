import json

# 파일 경로
file_path = 'evaluation/scoring/by_gpt/kmeans/ModernBERT/Qwen2_1.5B/self_instruct/ModernBERT_vs_Ori.json'

# 카운터
win, lose, tie, fail = 0, 0, 0, 0
fail_list = []

def get_winner(data):
    try:
        verdict = data.get("verdict", "")
        if not isinstance(verdict, str):
            return None
        
        # 마지막에 있는 [[A]], [[B]], [[C]] 패턴을 찾기
        if verdict.endswith("[[A]]"):
            return "model_1"
        elif verdict.endswith("[[B]]"):
            return "model_2"
        elif verdict.endswith("[[C]]"):
            return "tie"
        else:
            return None
    except Exception:
        return None

# 파일 읽기 및 판별
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            swapped = data.get("swapped", None)
            winner = get_winner(data)

            if swapped not in [True, False] or winner not in ["model_1", "model_2", "tie"]:
                fail += 1
                fail_list.append(data.get("input", "UNKNOWN_INPUT"))
                continue

            if winner == "tie":
                tie += 1
            # swapped=False → ours = model_1
            # swapped=True  → ours = model_2
            elif (winner == "model_1" and not swapped) or (winner == "model_2" and swapped):
                win += 1
            else:
                lose += 1

        except Exception:
            fail += 1
            fail_list.append("JSON_ERROR")

# 출력
total = win + lose + tie
print(f"Total: {total}")
if total > 0:
    print(f"Win:  {win / total:.2%} ({win})")
    print(f"Lose: {lose / total:.2%} ({lose})")
    print(f"Tie:  {tie / total:.2%} ({tie})")
else:
    print(f"Win:  0.00% (0)")
    print(f"Lose: 0.00% (0)")
    print(f"Tie:  0.00% (0)")
print(f"Fail: {fail}")
if fail_list:
    print("Sample Fails:")
    for x in fail_list[:5]:
        print(" -", x)
