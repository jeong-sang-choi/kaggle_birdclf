"""
시험 직전 암기용 핵심 알고리즘 템플릿
코딩 실력이 부족할 때 외워서 사용할 수 있는 코드들
"""

# ===== 1. 기본 템플릿들 =====

# 배열 회전 (자주 나옴!)
def rotate_array_template(arr, k):
    n = len(arr)
    k = k % n
    return arr[-k:] + arr[:-k]

# 최대 부분수열 합 (Kadane - 매우 중요!)
def max_subarray_template(arr):
    if not arr: return 0
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# 이진 탐색 (정렬된 배열에서 찾기)
def binary_search_template(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# ===== 2. DFS/BFS 템플릿 =====

# DFS (깊이 우선 탐색)
def dfs_template(graph, start):
    visited = set()
    def dfs_helper(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs_helper(neighbor)
    dfs_helper(start)
    return visited

# BFS (너비 우선 탐색) - 최단 경로에 유용
def bfs_template(graph, start):
    from collections import deque
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

# ===== 3. 동적 프로그래밍 템플릿 =====

# 피보나치 DP
def fibonacci_dp_template(n):
    if n <= 1: return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# 배낭 문제 (0/1 Knapsack)
def knapsack_template(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

# ===== 4. 정렬 템플릿 =====

# 퀵 정렬
def quick_sort_template(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort_template(left) + middle + quick_sort_template(right)

# ===== 5. 문자열 처리 템플릿 =====

# 문자열 압축
def string_compression_template(s):
    if not s: return ""
    result = ""
    count = 1
    current_char = s[0]
    
    for i in range(1, len(s)):
        if s[i] == current_char:
            count += 1
        else:
            result += current_char + str(count)
            current_char = s[i]
            count = 1
    
    result += current_char + str(count)
    return result

# 팰린드롬 확인
def is_palindrome_template(s):
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]

# ===== 6. 자주 나오는 문제 패턴 =====

# 두 수의 합 (해시맵 사용)
def two_sum_template(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# 소수 판별
def is_prime_template(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# ===== 7. 시험에서 바로 사용할 수 있는 완성된 함수들 =====

def exam_ready_functions():
    """
    시험에서 바로 복사해서 사용할 수 있는 완성된 함수들
    """
    
    # 문제 1: 배열 회전
    def rotate_array(arr, k):
        n = len(arr)
        k = k % n
        return arr[-k:] + arr[:-k]
    
    # 문제 2: 최대 부분수열 합
    def max_subarray_sum(arr):
        if not arr: return 0
        max_sum = current_sum = arr[0]
        for num in arr[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        return max_sum
    
    # 문제 3: 이진수 변환
    def decimal_to_binary(n):
        if n == 0: return "0"
        result = ""
        while n > 0:
            result = str(n % 2) + result
            n //= 2
        return result
    
    # 문제 4: 소수 판별
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True
    
    # 문제 5: 문자열 압축
    def compress_string(s):
        if not s: return ""
        result = ""
        count = 1
        current_char = s[0]
        for i in range(1, len(s)):
            if s[i] == current_char:
                count += 1
            else:
                result += current_char + str(count)
                current_char = s[i]
                count = 1
        result += current_char + str(count)
        return result
    
    return {
        "rotate_array": rotate_array,
        "max_subarray_sum": max_subarray_sum,
        "decimal_to_binary": decimal_to_binary,
        "is_prime": is_prime,
        "compress_string": compress_string
    }

# ===== 테스트 =====
if __name__ == "__main__":
    print("=== 시험 직전 암기용 템플릿 테스트 ===\n")
    
    # 기본 템플릿 테스트
    print("1. 배열 회전 테스트:")
    arr = [1, 2, 3, 4, 5]
    print(f"원본: {arr}")
    print(f"2번 회전: {rotate_array_template(arr, 2)}")
    print()
    
    print("2. 최대 부분수열 합 테스트:")
    test_arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"배열: {test_arr}")
    print(f"최대 합: {max_subarray_template(test_arr)}")
    print()
    
    print("3. 이진 탐색 테스트:")
    sorted_arr = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    print(f"배열: {sorted_arr}, 찾을 값: {target}")
    print(f"위치: {binary_search_template(sorted_arr, target)}")
    print()
    
    print("4. 문자열 압축 테스트:")
    test_string = "aaabbc"
    print(f"원본: '{test_string}'")
    print(f"압축: '{string_compression_template(test_string)}'")
    print()
    
    print("5. 소수 판별 테스트:")
    test_numbers = [2, 3, 4, 17, 25]
    for num in test_numbers:
        print(f"{num}은 소수: {is_prime_template(num)}")
    print()
    
    # 완성된 함수들 테스트
    print("=== 완성된 함수들 테스트 ===")
    functions = exam_ready_functions()
    
    # 이진수 변환 테스트
    print(f"10을 이진수로: {functions['decimal_to_binary'](10)}")
    print(f"15를 이진수로: {functions['decimal_to_binary'](15)}") 