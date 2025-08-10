"""
현대오토에버 코딩테스트 연습 문제들
"""

# ===== 문제 1: 배열 회전 (구현/시뮬레이션) =====
def rotate_array(arr, k):
    """
    배열을 k번 오른쪽으로 회전시키는 함수
    예: [1,2,3,4,5], k=2 → [4,5,1,2,3]
    """
    n = len(arr)
    k = k % n  # k가 배열 길이보다 클 경우 처리
    return arr[-k:] + arr[:-k]

# ===== 문제 2: 최대 연속 부분수열 합 (동적 프로그래밍) =====
def max_subarray_sum(arr):
    """
    연속된 부분수열 중 합이 최대가 되는 값을 찾는 함수
    예: [-2,1,-3,4,-1,2,1,-5,4] → 6 (4,-1,2,1)
    """
    if not arr:
        return 0
    
    max_sum = current_sum = arr[0]
    
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# ===== 문제 3: 이진수 변환 (구현) =====
def decimal_to_binary(n):
    """
    십진수를 이진수로 변환하는 함수
    예: 10 → "1010"
    """
    if n == 0:
        return "0"
    
    result = ""
    while n > 0:
        result = str(n % 2) + result
        n //= 2
    
    return result

# ===== 문제 4: 소수 판별 (수학) =====
def is_prime(n):
    """
    주어진 수가 소수인지 판별하는 함수
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    
    return True

# ===== 문제 5: 문자열 압축 (구현) =====
def compress_string(s):
    """
    문자열을 압축하는 함수
    예: "aaabbc" → "a3b2c1"
    """
    if not s:
        return ""
    
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

# ===== 문제 6: 두 수의 합 (해시/투포인터) =====
def two_sum(nums, target):
    """
    배열에서 두 수의 합이 target이 되는 인덱스를 찾는 함수
    예: [2,7,11,15], target=9 → [0,1]
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# ===== 문제 7: 팰린드롬 판별 (문자열) =====
def is_palindrome(s):
    """
    문자열이 팰린드롬인지 판별하는 함수 (대소문자 무시, 특수문자 제거)
    예: "A man, a plan, a canal: Panama" → True
    """
    # 알파벳과 숫자만 남기고 소문자로 변환
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]

# ===== 문제 8: 최대 힙 구현 (자료구조) =====
class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, val):
        self.heap.append(val)
        self._heapify_up()
    
    def pop(self):
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down()
        
        return max_val
    
    def _heapify_up(self):
        idx = len(self.heap) - 1
        while idx > 0:
            parent = (idx - 1) // 2
            if self.heap[idx] > self.heap[parent]:
                self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
                idx = parent
            else:
                break
    
    def _heapify_down(self):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            largest = idx
            
            if left < len(self.heap) and self.heap[left] > self.heap[largest]:
                largest = left
            if right < len(self.heap) and self.heap[right] > self.heap[largest]:
                largest = right
            
            if largest == idx:
                break
            
            self.heap[idx], self.heap[largest] = self.heap[largest], self.heap[idx]
            idx = largest

# ===== 문제 9: 그래프 DFS (그래프 탐색) =====
def dfs(graph, start, visited=None):
    """
    그래프를 DFS로 탐색하는 함수
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start, end=' ')
    
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited

# ===== 문제 10: 정렬 알고리즘 (정렬) =====
def quick_sort(arr):
    """
    퀵 정렬 구현
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# ===== 테스트 함수들 =====
def test_all_functions():
    print("=== 현대오토에버 코딩테스트 연습 문제 테스트 ===\n")
    
    # 문제 1 테스트
    print("1. 배열 회전 테스트:")
    arr = [1, 2, 3, 4, 5]
    print(f"원본: {arr}")
    print(f"2번 회전: {rotate_array(arr, 2)}")
    print(f"3번 회전: {rotate_array(arr, 3)}\n")
    
    # 문제 2 테스트
    print("2. 최대 연속 부분수열 합 테스트:")
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"배열: {arr}")
    print(f"최대 합: {max_subarray_sum(arr)}\n")
    
    # 문제 3 테스트
    print("3. 이진수 변환 테스트:")
    print(f"10 → {decimal_to_binary(10)}")
    print(f"15 → {decimal_to_binary(15)}\n")
    
    # 문제 4 테스트
    print("4. 소수 판별 테스트:")
    test_numbers = [2, 3, 4, 17, 25]
    for num in test_numbers:
        print(f"{num}은 소수: {is_prime(num)}")
    print()
    
    # 문제 5 테스트
    print("5. 문자열 압축 테스트:")
    test_strings = ["aaabbc", "abc", "aabbb"]
    for s in test_strings:
        print(f"'{s}' → '{compress_string(s)}'")
    print()
    
    # 문제 6 테스트
    print("6. 두 수의 합 테스트:")
    nums = [2, 7, 11, 15]
    target = 9
    print(f"배열: {nums}, 목표: {target}")
    print(f"결과: {two_sum(nums, target)}\n")
    
    # 문제 7 테스트
    print("7. 팰린드롬 판별 테스트:")
    test_strings = ["A man, a plan, a canal: Panama", "race a car", "hello"]
    for s in test_strings:
        print(f"'{s}' → {is_palindrome(s)}")
    print()
    
    # 문제 8 테스트
    print("8. 최대 힙 테스트:")
    heap = MaxHeap()
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    for num in numbers:
        heap.push(num)
    
    print("힙에서 최대값들 추출:")
    for _ in range(5):
        print(heap.pop(), end=' ')
    print("\n")
    
    # 문제 9 테스트
    print("9. DFS 테스트:")
    graph = {
        1: [2, 3],
        2: [4, 5],
        3: [6],
        4: [],
        5: [],
        6: []
    }
    print("DFS 순서:", end=' ')
    dfs(graph, 1)
    print("\n")
    
    # 문제 10 테스트
    print("10. 퀵 정렬 테스트:")
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"정렬 전: {arr}")
    print(f"정렬 후: {quick_sort(arr)}")

if __name__ == "__main__":
    test_all_functions() 