"""
현대오토에버 코딩테스트 모의고사
"""

# ===== 문제 1: 자동차 주차 관리 시스템 =====
def parking_system(records):
    """
    자동차 주차 기록을 처리하여 요금을 계산하는 함수
    
    records: ["05:34 5961 IN", "06:00 0000 IN", "06:34 0000 OUT", ...]
    """
    def time_to_minutes(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    parking = {}  # {차량번호: 입차시간}
    total_time = {}  # {차량번호: 총 주차시간}
    
    for record in records:
        time, car_id, action = record.split()
        minutes = time_to_minutes(time)
        
        if action == "IN":
            parking[car_id] = minutes
        else:  # OUT
            if car_id in parking:
                duration = minutes - parking[car_id]
                total_time[car_id] = total_time.get(car_id, 0) + duration
                del parking[car_id]
    
    # 아직 나가지 않은 차량들 처리 (23:59에 나간 것으로 간주)
    end_time = time_to_minutes("23:59")
    for car_id, in_time in parking.items():
        duration = end_time - in_time
        total_time[car_id] = total_time.get(car_id, 0) + duration
    
    # 요금 계산
    result = []
    for car_id in sorted(total_time.keys()):
        total_minutes = total_time[car_id]
        if total_minutes <= 90:  # 기본시간 90분
            fee = 5000
        else:
            additional_time = total_minutes - 90
            additional_fee = (additional_time + 9) // 10 * 600  # 10분 단위로 올림
            fee = 5000 + additional_fee
        result.append([car_id, fee])
    
    return result

# ===== 문제 2: 로봇 청소기 경로 최적화 =====
def robot_cleaner_path(room):
    """
    로봇 청소기가 모든 영역을 청소하는 최소 이동 횟수 찾기
    
    room: 2D 배열 (0: 빈 공간, 1: 장애물, 2: 청소기 시작 위치)
    """
    def find_start(room):
        for i in range(len(room)):
            for j in range(len(room[0])):
                if room[i][j] == 2:
                    return i, j
        return 0, 0
    
    def count_dirty(room):
        count = 0
        for row in room:
            count += row.count(0)
        return count
    
    def dfs(x, y, cleaned, moves):
        if cleaned == total_dirty:
            return moves
        
        min_moves = float('inf')
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < len(room) and 0 <= ny < len(room[0]) and 
                room[nx][ny] == 0):
                room[nx][ny] = 3  # 청소 완료 표시
                result = dfs(nx, ny, cleaned + 1, moves + 1)
                room[nx][ny] = 0  # 되돌리기
                min_moves = min(min_moves, result)
        
        return min_moves
    
    start_x, start_y = find_start(room)
    total_dirty = count_dirty(room)
    
    if total_dirty == 0:
        return 0
    
    return dfs(start_x, start_y, 0, 0)

# ===== 문제 3: 네트워크 연결 최적화 =====
def network_optimization(n, connections):
    """
    n개의 노드를 연결하는 최소 비용 찾기 (MST)
    
    connections: [(node1, node2, cost), ...]
    """
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
    
    # Kruskal 알고리즘
    connections.sort(key=lambda x: x[2])  # 비용 순으로 정렬
    uf = UnionFind(n)
    total_cost = 0
    connected_edges = 0
    
    for node1, node2, cost in connections:
        if uf.union(node1, node2):
            total_cost += cost
            connected_edges += 1
            if connected_edges == n - 1:  # 모든 노드가 연결됨
                break
    
    return total_cost if connected_edges == n - 1 else -1

# ===== 문제 4: 데이터 압축 알고리즘 =====
def data_compression(data):
    """
    연속된 같은 문자를 압축하는 함수
    
    예: "aaabbc" → "a3b2c1"
    """
    if not data:
        return ""
    
    result = ""
    count = 1
    current_char = data[0]
    
    for i in range(1, len(data)):
        if data[i] == current_char:
            count += 1
        else:
            result += current_char + str(count)
            current_char = data[i]
            count = 1
    
    result += current_char + str(count)
    return result

# ===== 문제 5: 작업 스케줄링 =====
def job_scheduling(jobs):
    """
    작업들을 스케줄링하여 최대 이익을 찾는 함수
    
    jobs: [(start_time, end_time, profit), ...]
    """
    # 끝나는 시간 순으로 정렬
    jobs.sort(key=lambda x: x[1])
    
    n = len(jobs)
    dp = [0] * n
    dp[0] = jobs[0][2]  # 첫 번째 작업의 이익
    
    for i in range(1, n):
        # 현재 작업을 포함하지 않는 경우
        dp[i] = dp[i-1]
        
        # 현재 작업을 포함하는 경우
        current_profit = jobs[i][2]
        
        # 현재 작업과 겹치지 않는 마지막 작업 찾기
        j = i - 1
        while j >= 0 and jobs[j][1] > jobs[i][0]:
            j -= 1
        
        if j >= 0:
            current_profit += dp[j]
        
        dp[i] = max(dp[i], current_profit)
    
    return dp[n-1]

# ===== 문제 6: 파일 시스템 탐색 =====
def file_system_search(files, queries):
    """
    파일 시스템에서 쿼리에 맞는 파일들을 찾는 함수
    
    files: [{"name": "file1.txt", "size": 100, "type": "txt"}, ...]
    queries: [{"type": "txt", "min_size": 50}, ...]
    """
    def matches_query(file_info, query):
        for key, value in query.items():
            if key == "min_size":
                if file_info.get("size", 0) < value:
                    return False
            elif key == "max_size":
                if file_info.get("size", 0) > value:
                    return False
            elif key == "type":
                if file_info.get("type") != value:
                    return False
            elif key == "name_contains":
                if value not in file_info.get("name", ""):
                    return False
        return True
    
    results = []
    for query in queries:
        matching_files = [f for f in files if matches_query(f, query)]
        results.append(matching_files)
    
    return results

# ===== 문제 7: 메모리 관리 시스템 =====
def memory_management(operations):
    """
    메모리 할당/해제 작업을 처리하는 함수
    
    operations: ["allocate 100", "allocate 200", "free 0", ...]
    """
    memory_blocks = []  # [(id, size, allocated), ...]
    next_id = 0
    
    def allocate(size):
        nonlocal next_id
        # 가장 작은 적절한 블록 찾기 (Best Fit)
        best_block = None
        best_fragmentation = float('inf')
        
        for i, (block_id, block_size, allocated) in enumerate(memory_blocks):
            if not allocated and block_size >= size:
                fragmentation = block_size - size
                if fragmentation < best_fragmentation:
                    best_fragmentation = fragmentation
                    best_block = i
        
        if best_block is not None:
            block_id, block_size, _ = memory_blocks[best_block]
            memory_blocks[best_block] = (block_id, size, True)
            
            # 남은 공간이 있으면 새 블록 생성
            if block_size > size:
                memory_blocks.append((next_id, block_size - size, False))
                next_id += 1
            
            return block_id
        else:
            # 새로운 블록 할당
            memory_blocks.append((next_id, size, True))
            next_id += 1
            return next_id - 1
    
    def free(block_id):
        for i, (bid, size, allocated) in enumerate(memory_blocks):
            if bid == block_id and allocated:
                memory_blocks[i] = (bid, size, False)
                return True
        return False
    
    def defragment():
        # 할당된 블록들을 앞으로 이동
        allocated = [(bid, size) for bid, size, allocated in memory_blocks if allocated]
        free_size = sum(size for _, size, allocated in memory_blocks if not allocated)
        
        memory_blocks.clear()
        next_id = 0
        
        for _, size in allocated:
            memory_blocks.append((next_id, size, True))
            next_id += 1
        
        if free_size > 0:
            memory_blocks.append((next_id, free_size, False))
    
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
    
    return results

# ===== 테스트 함수 =====
def run_mock_test():
    print("=== 현대오토에버 코딩테스트 모의고사 ===\n")
    
    # 문제 1 테스트
    print("문제 1: 자동차 주차 관리 시스템")
    records = [
        "05:34 5961 IN",
        "06:00 0000 IN", 
        "06:34 0000 OUT",
        "07:59 5961 OUT",
        "07:59 0148 IN",
        "18:59 0000 IN",
        "19:09 0148 OUT",
        "22:59 5961 IN",
        "23:00 5961 OUT"
    ]
    result = parking_system(records)
    print(f"주차 요금: {result}\n")
    
    # 문제 2 테스트
    print("문제 2: 로봇 청소기 경로 최적화")
    room = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 2, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    min_moves = robot_cleaner_path(room)
    print(f"최소 이동 횟수: {min_moves}\n")
    
    # 문제 3 테스트
    print("문제 3: 네트워크 연결 최적화")
    n = 4
    connections = [(0, 1, 10), (0, 2, 6), (0, 3, 5), (1, 3, 15), (2, 3, 4)]
    min_cost = network_optimization(n, connections)
    print(f"최소 연결 비용: {min_cost}\n")
    
    # 문제 4 테스트
    print("문제 4: 데이터 압축 알고리즘")
    test_data = ["aaabbc", "abc", "aabbb", "aaaa"]
    for data in test_data:
        compressed = data_compression(data)
        print(f"'{data}' → '{compressed}'")
    print()
    
    # 문제 5 테스트
    print("문제 5: 작업 스케줄링")
    jobs = [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4), (5, 8, 11), (7, 9, 2)]
    max_profit = job_scheduling(jobs)
    print(f"최대 이익: {max_profit}\n")
    
    # 문제 6 테스트
    print("문제 6: 파일 시스템 탐색")
    files = [
        {"name": "document.txt", "size": 100, "type": "txt"},
        {"name": "image.jpg", "size": 500, "type": "jpg"},
        {"name": "video.mp4", "size": 1000, "type": "mp4"},
        {"name": "report.txt", "size": 200, "type": "txt"}
    ]
    queries = [
        {"type": "txt", "min_size": 50},
        {"max_size": 300},
        {"name_contains": "doc"}
    ]
    search_results = file_system_search(files, queries)
    for i, result in enumerate(search_results):
        print(f"쿼리 {i+1}: {len(result)}개 파일")
    print()
    
    # 문제 7 테스트
    print("문제 7: 메모리 관리 시스템")
    operations = [
        "allocate 100",
        "allocate 200", 
        "allocate 150",
        "free 1",
        "allocate 50",
        "defragment"
    ]
    memory_results = memory_management(operations)
    for result in memory_results:
        print(result)

if __name__ == "__main__":
    run_mock_test() 