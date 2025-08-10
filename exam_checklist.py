"""
현대오토에버 코딩테스트 시험 직전 체크리스트
"""

# ===== 시험 직전 체크리스트 =====
CHECKLIST = """
✅ 시험 직전 체크리스트:

1. 기본 문법 숙지
   - Python 기본 문법 (리스트, 딕셔너리, 문자열)
   - 시간복잡도 개념
   - 입출력 방법

2. 핵심 알고리즘
   - 정렬: 퀵정렬, 병합정렬
   - 탐색: 이진탐색, DFS, BFS
   - 동적프로그래밍: 최대 부분수열, 배낭문제
   - 그리디: 최소 스패닝 트리

3. 자료구조
   - 스택/큐
   - 힙 (우선순위 큐)
   - 해시맵
   - 유니온-파인드

4. 문제 해결 전략
   - 문제 읽기 (3번)
   - 예시 확인
   - 시간복잡도 계산
   - 테스트케이스 검증
"""

# ===== 핵심 알고리즘 요약 =====
def quick_reference():
    """
    시험 직전 빠른 참고용 핵심 알고리즘
    """
    
    # 1. 이진 탐색
    def binary_search(arr, target):
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
    
    # 2. DFS (깊이 우선 탐색)
    def dfs(graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        for neighbor in graph.get(start, []):
            if neighbor not in visited:
                dfs(graph, neighbor, visited)
        return visited
    
    # 3. BFS (너비 우선 탐색)
    def bfs(graph, start):
        from collections import deque
        queue = deque([start])
        visited = {start}
        while queue:
            vertex = queue.popleft()
            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited
    
    # 4. 동적 프로그래밍 - 피보나치
    def fibonacci_dp(n):
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
    
    # 5. 최대 부분수열 합 (Kadane's Algorithm)
    def max_subarray_sum(arr):
        if not arr:
            return 0
        max_sum = current_sum = arr[0]
        for num in arr[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        return max_sum
    
    # 6. 유니온-파인드
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            if self.rank[px] < self.rank[py]:
                self.parent[px] = py
            elif self.rank[px] > self.rank[py]:
                self.parent[py] = px
            else:
                self.parent[py] = px
                self.rank[px] += 1
            return True
    
    return {
        "binary_search": binary_search,
        "dfs": dfs,
        "bfs": bfs,
        "fibonacci_dp": fibonacci_dp,
        "max_subarray_sum": max_subarray_sum,
        "UnionFind": UnionFind
    }

# ===== 시험 팁 =====
EXAM_TIPS = """
🎯 시험 팁:

1. 문제 풀이 순서
   - 쉬운 문제부터 풀기
   - 각 문제당 20-30분 제한
   - 시간이 오래 걸리면 다음 문제로

2. 코드 작성 시
   - 변수명을 명확하게
   - 주석으로 로직 설명
   - 예외 케이스 처리

3. 디버깅
   - print문으로 중간 결과 확인
   - 작은 테스트케이스로 검증
   - 시간복잡도 확인

4. 마음가짐
   - 차분하게 문제 읽기
   - 포기하지 말고 최선을 다하기
   - 시간 관리가 중요
"""

# ===== 자주 나오는 문제 유형 =====
COMMON_PROBLEM_TYPES = """
📋 자주 나오는 문제 유형:

1. 구현/시뮬레이션 (30%)
   - 배열 조작, 문자열 처리
   - 시간/날짜 계산
   - 게임 규칙 구현

2. 탐색 (25%)
   - DFS/BFS
   - 완전탐색
   - 이진탐색

3. 동적프로그래밍 (20%)
   - 최적화 문제
   - 메모이제이션
   - 상태 전이

4. 자료구조 (15%)
   - 스택/큐 활용
   - 힙 (우선순위 큐)
   - 해시맵

5. 그리디/정렬 (10%)
   - 정렬 후 처리
   - 그리디 선택
"""

if __name__ == "__main__":
    print(CHECKLIST)
    print("\n" + "="*50 + "\n")
    print(EXAM_TIPS)
    print("\n" + "="*50 + "\n")
    print(COMMON_PROBLEM_TYPES)
    
    # 핵심 알고리즘 테스트
    print("\n=== 핵심 알고리즘 테스트 ===")
    algorithms = quick_reference()
    
    # 이진 탐색 테스트
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"이진 탐색 {arr}에서 7 찾기: {algorithms['binary_search'](arr, 7)}")
    
    # 최대 부분수열 합 테스트
    test_arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"최대 부분수열 합 {test_arr}: {algorithms['max_subarray_sum'](test_arr)}")
    
    # 피보나치 테스트
    print(f"피보나치 10: {algorithms['fibonacci_dp'](10)}") 