"""
현대오토에버 코딩테스트 최종 모의고사
시험 직전 마지막 연습용
"""

import time
from collections import deque, defaultdict

# ===== 문제 1: 자동차 부품 관리 시스템 (구현) =====
def car_parts_management(parts_data, queries):
    """
    자동차 부품의 재고를 관리하는 시스템
    
    parts_data: [{"id": "P001", "name": "엔진", "stock": 10, "price": 500000}, ...]
    queries: ["check P001", "add P001 5", "sell P001 3", "list low_stock"]
    """
    parts = {}
    for part in parts_data:
        parts[part["id"]] = {
            "name": part["name"],
            "stock": part["stock"],
            "price": part["price"]
        }
    
    results = []
    for query in queries:
        cmd = query.split()
        
        if cmd[0] == "check":
            part_id = cmd[1]
            if part_id in parts:
                part = parts[part_id]
                results.append(f"{part_id}: {part['name']}, 재고: {part['stock']}, 가격: {part['price']}")
            else:
                results.append(f"부품 {part_id}을 찾을 수 없습니다.")
        
        elif cmd[0] == "add":
            part_id, quantity = cmd[1], int(cmd[2])
            if part_id in parts:
                parts[part_id]["stock"] += quantity
                results.append(f"{part_id} 재고 {quantity}개 추가됨")
            else:
                results.append(f"부품 {part_id}을 찾을 수 없습니다.")
        
        elif cmd[0] == "sell":
            part_id, quantity = cmd[1], int(cmd[2])
            if part_id in parts:
                if parts[part_id]["stock"] >= quantity:
                    parts[part_id]["stock"] -= quantity
                    total_price = parts[part_id]["price"] * quantity
                    results.append(f"{part_id} {quantity}개 판매됨, 총액: {total_price}")
                else:
                    results.append(f"{part_id} 재고 부족 (현재: {parts[part_id]['stock']})")
            else:
                results.append(f"부품 {part_id}을 찾을 수 없습니다.")
        
        elif cmd[0] == "list" and cmd[1] == "low_stock":
            low_stock_parts = [pid for pid, part in parts.items() if part["stock"] < 5]
            if low_stock_parts:
                results.append(f"재고 부족 부품: {', '.join(low_stock_parts)}")
            else:
                results.append("재고 부족 부품 없음")
    
    return results

# ===== 문제 2: 네트워크 연결 상태 확인 (BFS) =====
def network_connectivity_check(n, connections, start_node):
    """
    네트워크에서 특정 노드에서 연결 가능한 모든 노드 찾기
    
    n: 노드 개수
    connections: [(node1, node2), ...]
    start_node: 시작 노드
    """
    # 그래프 구성
    graph = defaultdict(list)
    for node1, node2 in connections:
        graph[node1].append(node2)
        graph[node2].append(node1)
    
    # BFS로 연결된 노드들 찾기
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return sorted(list(visited))

# ===== 문제 3: 최적 경로 찾기 (다익스트라) =====
def find_optimal_path(n, roads, start, end):
    """
    도시 간 최단 경로 찾기
    
    n: 도시 개수
    roads: [(city1, city2, distance), ...]
    start: 시작 도시
    end: 목적지 도시
    """
    # 그래프 구성
    graph = defaultdict(list)
    for city1, city2, distance in roads:
        graph[city1].append((city2, distance))
        graph[city2].append((city1, distance))
    
    # 다익스트라 알고리즘
    distances = {i: float('inf') for i in range(n)}
    distances[start] = 0
    visited = set()
    
    while len(visited) < n:
        # 방문하지 않은 노드 중 최단 거리 노드 찾기
        current = None
        min_dist = float('inf')
        for i in range(n):
            if i not in visited and distances[i] < min_dist:
                min_dist = distances[i]
                current = i
        
        if current is None:
            break
        
        visited.add(current)
        
        # 인접 노드들의 거리 업데이트
        for neighbor, distance in graph[current]:
            if neighbor not in visited:
                new_distance = distances[current] + distance
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
    
    return distances[end] if distances[end] != float('inf') else -1

# ===== 문제 4: 데이터 압축 및 복원 (구현) =====
def data_compression_restore(data):
    """
    데이터 압축 및 복원
    
    압축: "aaabbc" → "a3b2c1"
    복원: "a3b2c1" → "aaabbc"
    """
    def compress(s):
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
    
    def restore(s):
        if not s:
            return ""
        
        result = ""
        i = 0
        while i < len(s):
            char = s[i]
            i += 1
            
            # 숫자 읽기
            count_str = ""
            while i < len(s) and s[i].isdigit():
                count_str += s[i]
                i += 1
            
            count = int(count_str) if count_str else 1
            result += char * count
        
        return result
    
    # 압축
    compressed = compress(data)
    
    # 복원
    restored = restore(compressed)
    
    return {
        "original": data,
        "compressed": compressed,
        "restored": restored,
        "compression_ratio": len(compressed) / len(data) if data else 0
    }

# ===== 문제 5: 작업 스케줄링 최적화 (그리디) =====
def job_scheduling_optimization(jobs):
    """
    작업들을 스케줄링하여 최대 이익을 찾기
    
    jobs: [(id, start_time, end_time, profit), ...]
    """
    # 끝나는 시간 순으로 정렬
    jobs.sort(key=lambda x: x[2])
    
    n = len(jobs)
    dp = [0] * n
    dp[0] = jobs[0][3]  # 첫 번째 작업의 이익
    
    for i in range(1, n):
        # 현재 작업을 포함하지 않는 경우
        dp[i] = dp[i-1]
        
        # 현재 작업을 포함하는 경우
        current_profit = jobs[i][3]
        
        # 현재 작업과 겹치지 않는 마지막 작업 찾기 (이진 탐색)
        j = i - 1
        while j >= 0 and jobs[j][2] > jobs[i][1]:
            j -= 1
        
        if j >= 0:
            current_profit += dp[j]
        
        dp[i] = max(dp[i], current_profit)
    
    # 선택된 작업들 찾기
    selected_jobs = []
    i = n - 1
    while i >= 0:
        if i == 0 or dp[i] != dp[i-1]:
            selected_jobs.append(jobs[i][0])
            # 이전 작업으로 이동
            j = i - 1
            while j >= 0 and jobs[j][2] > jobs[i][1]:
                j -= 1
            i = j
        else:
            i -= 1
    
    return {
        "max_profit": dp[n-1],
        "selected_jobs": selected_jobs[::-1]
    }

# ===== 문제 6: 메모리 할당 시뮬레이터 (시뮬레이션) =====
def memory_allocation_simulator(operations):
    """
    메모리 할당/해제 시뮬레이터
    
    operations: ["allocate 100", "allocate 200", "free 0", "defragment"]
    """
    memory_blocks = []  # [(id, size, allocated, start_address), ...]
    next_id = 0
    next_address = 0
    
    def allocate(size):
        nonlocal next_id, next_address
        
        # Best Fit: 가장 작은 적절한 블록 찾기
        best_block = None
        best_fragmentation = float('inf')
        
        for i, (block_id, block_size, allocated, start_addr) in enumerate(memory_blocks):
            if not allocated and block_size >= size:
                fragmentation = block_size - size
                if fragmentation < best_fragmentation:
                    best_fragmentation = fragmentation
                    best_block = i
        
        if best_block is not None:
            # 기존 블록 분할
            block_id, block_size, _, start_addr = memory_blocks[best_block]
            memory_blocks[best_block] = (block_id, size, True, start_addr)
            
            # 남은 공간이 있으면 새 블록 생성
            if block_size > size:
                memory_blocks.append((next_id, block_size - size, False, start_addr + size))
                next_id += 1
            
            return block_id
        else:
            # 새로운 블록 할당
            memory_blocks.append((next_id, size, True, next_address))
            next_id += 1
            next_address += size
            return next_id - 1
    
    def free(block_id):
        for i, (bid, size, allocated, start_addr) in enumerate(memory_blocks):
            if bid == block_id and allocated:
                memory_blocks[i] = (bid, size, False, start_addr)
                return True
        return False
    
    def defragment():
        nonlocal next_address
        # 할당된 블록들을 앞으로 이동
        allocated = [(bid, size, start_addr) for bid, size, allocated, start_addr in memory_blocks if allocated]
        free_size = sum(size for _, size, allocated, _ in memory_blocks if not allocated)
        
        memory_blocks.clear()
        next_id = 0
        next_address = 0
        
        for _, size, _ in allocated:
            memory_blocks.append((next_id, size, True, next_address))
            next_id += 1
            next_address += size
        
        if free_size > 0:
            memory_blocks.append((next_id, free_size, False, next_address))
    
    def get_memory_status():
        total_allocated = sum(size for _, size, allocated, _ in memory_blocks if allocated)
        total_free = sum(size for _, size, allocated, _ in memory_blocks if not allocated)
        fragmentation = len([b for b in memory_blocks if not b[2]])
        
        return {
            "total_allocated": total_allocated,
            "total_free": total_free,
            "fragmentation_count": fragmentation,
            "blocks": memory_blocks
        }
    
    results = []
    for operation in operations:
        parts = operation.split()
        if parts[0] == "allocate":
            size = int(parts[1])
            block_id = allocate(size)
            results.append(f"allocated block {block_id}")
        elif parts[0] == "free":
            block_id = int(parts[1])
            if free(block_id):
                results.append(f"freed block {block_id}")
            else:
                results.append(f"block {block_id} not found")
        elif parts[0] == "defragment":
            defragment()
            results.append("defragmentation completed")
        elif parts[0] == "status":
            status = get_memory_status()
            results.append(f"Memory status: {status}")
    
    return results

# ===== 테스트 실행 =====
def run_final_mock_test():
    print("=== 현대오토에버 코딩테스트 최종 모의고사 ===\n")
    
    # 문제 1 테스트
    print("문제 1: 자동차 부품 관리 시스템")
    parts_data = [
        {"id": "P001", "name": "엔진", "stock": 10, "price": 500000},
        {"id": "P002", "name": "브레이크", "stock": 3, "price": 100000},
        {"id": "P003", "name": "타이어", "stock": 20, "price": 80000}
    ]
    queries = [
        "check P001",
        "add P001 5", 
        "sell P001 3",
        "check P002",
        "list low_stock"
    ]
    result1 = car_parts_management(parts_data, queries)
    for r in result1:
        print(f"  {r}")
    print()
    
    # 문제 2 테스트
    print("문제 2: 네트워크 연결 상태 확인")
    n = 6
    connections = [(0, 1), (1, 2), (2, 3), (4, 5)]
    start_node = 0
    connected = network_connectivity_check(n, connections, start_node)
    print(f"  노드 {start_node}에서 연결 가능한 노드들: {connected}")
    print()
    
    # 문제 3 테스트
    print("문제 3: 최적 경로 찾기")
    n = 4
    roads = [(0, 1, 10), (0, 2, 6), (1, 2, 2), (1, 3, 15), (2, 3, 4)]
    start, end = 0, 3
    min_distance = find_optimal_path(n, roads, start, end)
    print(f"  {start}에서 {end}까지 최단 거리: {min_distance}")
    print()
    
    # 문제 4 테스트
    print("문제 4: 데이터 압축 및 복원")
    test_data = ["aaabbc", "abc", "aabbb", "aaaa"]
    for data in test_data:
        result = data_compression_restore(data)
        print(f"  원본: '{data}'")
        print(f"  압축: '{result['compressed']}'")
        print(f"  복원: '{result['restored']}'")
        print(f"  압축률: {result['compression_ratio']:.2f}")
        print()
    
    # 문제 5 테스트
    print("문제 5: 작업 스케줄링 최적화")
    jobs = [
        (1, 1, 3, 5),
        (2, 2, 5, 6), 
        (3, 4, 6, 5),
        (4, 6, 7, 4),
        (5, 5, 8, 11),
        (6, 7, 9, 2)
    ]
    result5 = job_scheduling_optimization(jobs)
    print(f"  최대 이익: {result5['max_profit']}")
    print(f"  선택된 작업: {result5['selected_jobs']}")
    print()
    
    # 문제 6 테스트
    print("문제 6: 메모리 할당 시뮬레이터")
    operations = [
        "allocate 100",
        "allocate 200",
        "allocate 150", 
        "free 1",
        "allocate 50",
        "defragment"
    ]
    result6 = memory_allocation_simulator(operations)
    for r in result6:
        print(f"  {r}")
    print()

if __name__ == "__main__":
    start_time = time.time()
    run_final_mock_test()
    end_time = time.time()
    print(f"총 소요 시간: {end_time - start_time:.2f}초") 