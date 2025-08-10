"""
현대오토에버 코딩테스트 고급 문제들
"""

# ===== 문제 1: 미로 탈출 (BFS) =====
def escape_maze(maze):
    """
    미로에서 시작점(0,0)에서 끝점(n-1,m-1)까지의 최단 경로 찾기
    0: 통로, 1: 벽
    """
    if not maze or not maze[0]:
        return -1
    
    n, m = len(maze), len(maze[0])
    if maze[0][0] == 1 or maze[n-1][m-1] == 1:
        return -1
    
    # 방향: 상, 하, 좌, 우
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    queue = [(0, 0, 1)]  # (x, y, distance)
    visited = set()
    visited.add((0, 0))
    
    while queue:
        x, y, dist = queue.pop(0)
        
        if x == n-1 and y == m-1:
            return dist
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < n and 0 <= ny < m and 
                maze[nx][ny] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))
    
    return -1

# ===== 문제 2: 배낭 문제 (동적 프로그래밍) =====
def knapsack(weights, values, capacity):
    """
    0/1 배낭 문제 해결
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

# ===== 문제 3: 최소 스패닝 트리 (Kruskal) =====
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

def kruskal_mst(edges, n):
    """
    Kruskal 알고리즘으로 최소 스패닝 트리 찾기
    edges: [(weight, u, v), ...]
    """
    edges.sort()  # 가중치 순으로 정렬
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = []
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst_weight += weight
            mst_edges.append((u, v))
    
    return mst_weight, mst_edges

# ===== 문제 4: 최장 증가 부분수열 (LIS) =====
def longest_increasing_subsequence(arr):
    """
    최장 증가 부분수열의 길이 찾기
    """
    if not arr:
        return 0
    
    # 이진 탐색을 이용한 O(nlogn) 해결법
    tails = [arr[0]]
    
    for num in arr[1:]:
        if num > tails[-1]:
            tails.append(num)
        else:
            # 이진 탐색으로 적절한 위치 찾기
            left, right = 0, len(tails) - 1
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            tails[left] = num
    
    return len(tails)

# ===== 문제 5: 문자열 매칭 (KMP) =====
def kmp_search(text, pattern):
    """
    KMP 알고리즘으로 문자열 매칭
    """
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    if not pattern or not text:
        return []
    
    lps = compute_lps(pattern)
    i = j = 0
    matches = []
    
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

# ===== 문제 6: 토폴로지 정렬 =====
def topological_sort(graph):
    """
    DAG의 토폴로지 정렬
    """
    def dfs(node):
        visited.add(node)
        visiting.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor in visiting:
                return False  # 사이클 발견
            if neighbor not in visited:
                if not dfs(neighbor):
                    return False
        
        visiting.remove(node)
        result.append(node)
        return True
    
    visited = set()
    visiting = set()
    result = []
    
    for node in graph:
        if node not in visited:
            if not dfs(node):
                return None  # 사이클이 있으면 None 반환
    
    return result[::-1]

# ===== 문제 7: 슬라이딩 윈도우 최대값 =====
def max_sliding_window(nums, k):
    """
    슬라이딩 윈도우에서 최대값 찾기
    """
    if not nums or k == 0:
        return []
    
    from collections import deque
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # 윈도우 범위를 벗어난 인덱스 제거
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # 현재 값보다 작은 값들 제거
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # 윈도우가 완성되면 최대값 추가
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# ===== 문제 8: 이진 트리 최대 깊이 =====
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth_binary_tree(root):
    """
    이진 트리의 최대 깊이 찾기
    """
    if not root:
        return 0
    
    return max(max_depth_binary_tree(root.left), 
               max_depth_binary_tree(root.right)) + 1

# ===== 문제 9: 회문 분할 =====
def palindrome_partition(s):
    """
    문자열을 회문 부분문자열들로 분할하는 모든 방법 찾기
    """
    def is_palindrome(s, start, end):
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True
    
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            if is_palindrome(s, start, end - 1):
                path.append(s[start:end])
                backtrack(end, path)
                path.pop()
    
    result = []
    backtrack(0, [])
    return result

# ===== 문제 10: LRU 캐시 구현 =====
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = ListNode(0, 0)
        self.tail = ListNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.val
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = ListNode(key, value)
        self.cache[key] = node
        self._add(node)
        
        if len(self.cache) > self.capacity:
            node = self.head.next
            self._remove(node)
            del self.cache[node.key]
    
    def _add(self, node):
        p = self.tail.prev
        p.next = node
        self.tail.prev = node
        node.prev = p
        node.next = self.tail
    
    def _remove(self, node):
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p

class ListNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

# ===== 테스트 함수 =====
def test_advanced_problems():
    print("=== 현대오토에버 고급 문제 테스트 ===\n")
    
    # 문제 1 테스트
    print("1. 미로 탈출 테스트:")
    maze = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ]
    print(f"최단 거리: {escape_maze(maze)}\n")
    
    # 문제 2 테스트
    print("2. 배낭 문제 테스트:")
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 10
    print(f"최대 가치: {knapsack(weights, values, capacity)}\n")
    
    # 문제 3 테스트
    print("3. 최소 스패닝 트리 테스트:")
    edges = [(1, 0, 1), (2, 0, 2), (3, 1, 2), (4, 1, 3), (5, 2, 3)]
    n = 4
    weight, mst = kruskal_mst(edges, n)
    print(f"최소 가중치: {weight}")
    print(f"MST 간선들: {mst}\n")
    
    # 문제 4 테스트
    print("4. 최장 증가 부분수열 테스트:")
    arr = [10, 22, 9, 33, 21, 50, 41, 60]
    print(f"LIS 길이: {longest_increasing_subsequence(arr)}\n")
    
    # 문제 5 테스트
    print("5. KMP 문자열 매칭 테스트:")
    text = "ABABDABACDABABCABAB"
    pattern = "ABABCABAB"
    matches = kmp_search(text, pattern)
    print(f"매칭 위치: {matches}\n")
    
    # 문제 6 테스트
    print("6. 토폴로지 정렬 테스트:")
    graph = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: [5],
        5: []
    }
    topo_order = topological_sort(graph)
    print(f"토폴로지 순서: {topo_order}\n")
    
    # 문제 7 테스트
    print("7. 슬라이딩 윈도우 최대값 테스트:")
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    max_window = max_sliding_window(nums, k)
    print(f"슬라이딩 윈도우 최대값: {max_window}\n")
    
    # 문제 8 테스트
    print("8. 이진 트리 최대 깊이 테스트:")
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)
    print(f"최대 깊이: {max_depth_binary_tree(root)}\n")
    
    # 문제 9 테스트
    print("9. 회문 분할 테스트:")
    s = "aab"
    partitions = palindrome_partition(s)
    print(f"회문 분할: {partitions}\n")
    
    # 문제 10 테스트
    print("10. LRU 캐시 테스트:")
    lru = LRUCache(2)
    lru.put(1, 1)
    lru.put(2, 2)
    print(f"get(1): {lru.get(1)}")
    lru.put(3, 3)
    print(f"get(2): {lru.get(2)}")

if __name__ == "__main__":
    test_advanced_problems() 