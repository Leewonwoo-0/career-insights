#프로그래밍 실습 2

"""
프로젝트 설명
1. 해당 프로젝트는 티머니 데이터를 기반으로 Neural Network를 구현해보고자 하였습니다.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#폰트설정 + 한글폰트 숫자깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상 정의 (신경전달물질)
COLOR_PAID = np.array([0.0, 1.0, 1.0])  # Cyan (유임승차)
COLOR_FREE = np.array([1.0, 0.0, 1.0])  # Magenta (무임승차)

class Neural_transit_brain:
    def __init__(self, fee_path, time_path):
        self.fee_path = fee_path
        self.time_path = time_path
        self.hour = 24
        self.load_data() # 데이터 로드, 데이터 정리, -> self.line(호선), self.fee_data, self.time_data 생성
        self.initialize_brain_structure() # 뇌 영역별 매핑 -> self.brain_regions[뇌 영역]:좌표값, 해당역들 / self.station_to_region[역]:뇌 영역 저장
        self.get_time_intensity() # self.circ_params_dict,(뇌 영역별 일주기 기반 가중치 dict 생성) 및 일주기 뇌 영역별 가중치 할당

    def load_data(self):
        self.fee_data = pd.read_csv(self.fee_path, encoding='cp949')
        self.time_data = pd.read_csv(self.time_path, encoding='cp949')
        self.fee_data.columns = ['기간', '호선명', '역ID', '역명', '유임승차', '유임하차', '무임승차', '무임하차']
        # 추가 칼럼
        # pandas 문법상 self.df_data[컬럼명] = 값을 입력하면 자동으로 생성됨.
        self.fee_data['총하차'] = self.fee_data['유임하차'] + self.fee_data['무임하차']
        self.fee_data['유임비율'] = self.fee_data['유임하차'] / (self.fee_data['총하차'])
        self.fee_data['무임비율'] = self.fee_data['무임하차'] / (self.fee_data['총하차'])
        # 총 칼럼 ['기간', '호선명', '역ID', '역명', '유임승차', '유임하차', '무임승차', '무임하차', '총승차', '유임비율', '무임비율']
        self.line = self.fee_data['호선명'].unique()
        #unique 함수를 통해, 중복호선 제거 후 self.line에 저장
        print(f'호선: {len(self.line)}개')

    def initialize_brain_structure(self):
        # 딕셔너리 구조로, 데이터를 저장
        # 뇌 영역 위치 좌표값 모델링
        # 시각적으로 표현할 역을 뇌 영역에 대응시켜 모델링
        self.brain_regions = {
            # 전두엽: 계획, 의사결정 → 비즈니스 중심지
            '전두엽_좌': {'pos': (0.2, 0.7), 'stations': ['강남', '여의도', '광화문', '삼성', '역삼']},
            '전두엽_우': {'pos': (0.8, 0.7), 'stations': ['잠실', '신촌', '홍대입구', '구로디지털단지']},

            # 측두엽: 기억, 학습 → 대학가, 문화지역
            '측두엽_좌': {'pos': (0.15, 0.45), 'stations': ['혜화', '신림', '서울대입구', '이대']},
            '측두엽_우': {'pos': (0.85, 0.45), 'stations': ['건대입구', '성신여대입구', '고려대']},

            # 두정엽: 감각 통합 → 교통 허브
            '두정엽': {'pos': (0.5, 0.8), 'stations': ['서울역', '용산', '교대', '사당', '왕십리']},

            # 후두엽: 시각 → 관광지
            '후두엽': {'pos': (0.5, 0.3), 'stations': ['명동', '이태원', '동대문', '종각', '회현']},

            # 소뇌: 운동 조정 → 주거지역
            '소뇌_좌': {'pos': (0.3, 0.15), 'stations': ['노원', '상계', '도봉', '수유', '미아']},
            '소뇌_우': {'pos': (0.7, 0.15), 'stations': ['송파', '강동', '천호', '길동']},

            # 뇌간: 기본 생명 → 외곽/신도시
            '뇌간': {'pos': (0.5, 0.05), 'stations': ['인천', '부천', '수원', '일산', '서현']}
        }

        # 뇌 영역 매핑 생성, 고정된 딕셔너리를 기반으로 매핑하기에 튜플을 사용하여 메모리값에 해당하는 고정된 값을 가진 배열을 생성
        # 제 1정규형
        self.station_to_region = {}
        for region, info in self.brain_regions.items():
            for station in info['stations']:
                self.station_to_region[station] = region

        # station_to_region list는 각각의 역에 해당하는 뇌 기능을 갖고있는 list
        print(f'뇌 영역 개수  {len(self.brain_regions)}')
        print(f'역 매핑 구조 -> {self.station_to_region}')



    # 일주기 리듬 - 생리학·수학적 주기 시스템을
    # 물리학 뉴턴 제2법칙 -> 후크의법칙 -> 복소수 + 테일러 급수 기반 자연상수 전개를 통해 오일러 공식 cosθ + isinθ을 유도하여
    # 기본수식: I(t)=A⋅cos((2π/T)*(t−ϕ))+B을 구하고, 이를 기반으로 일 주기 리듬을 구현해보았습니다.
    # time: 1시간, 전체Time: 24시간
    # 뇌 기관의 기능에 따라 주기에서 피크시점을 분류하여 파라메터 값을 할당 했습니다.
    # A = 진폭, B = 기준선, ϕ = 위상
    # ex) 전두엽 A = 0.6, B = 0.4, ϕ=11 (11시 피크) => 오전 ~ 이른 오후

    def calculate_circadian_rhythm(self, params_dict):
        c = {
            region: np.clip(
                B + A * np.cos(2 * np.pi / self.hour * (np.arange(24) - phi)),
                0, None
            )
            for region, (A, B, phi) in params_dict.items()
        }
        # dict 안에 for문을 넣어서 코드를 대폭 줄여봤습니다.

        # softmax 계산 (시간별 분포가 아닌 전체 영역별 비율)
        arr = np.array(list(c.values()))  # (영역수 × 24)

        #원래 softmax 수식에선 x/tau 승을 자연상수에 넣어주지만, 기본 주기리듬은 tau를 1로 가져서 생략 했습니다.
        exp_c = np.exp(arr)
        softmax = exp_c / exp_c.sum(axis=0, keepdims=True)

        for region, soft_vals in zip(self.brain_regions.keys(), softmax):
            c[region] = soft_vals + 1
        return c


    def get_time_intensity(self):
        # 실제 지하철 이용 패턴에 맞춘 일주기 파라미터
        # (A: 진폭, B: 기준선, φ: 피크시간)
        # 주의: time_data는 4시부터 시작 (인덱스 0=4시), 따라서 φ는 (실제시간 - 4) % 24
        self.circ_params_dict = {
            # 비즈니스 중심지 (강남, 여의도, 광화문 등)
            '전두엽_좌': (0.6, 0.4, 5),   # 오전 9시 출근 피크 (9-4=5)
            '전두엽_우': (0.6, 0.4, 14),  # 저녁 6시 퇴근 피크 (18-4=14, 신촌/홍대)

            # 대학가/문화지역 (대학로, 신림, 건대 등)
            '측두엽_좌': (0.5, 0.45, 7),   # 오전 11시 수업 시작 (11-4=7)
            '측두엽_우': (0.5, 0.45, 15),  # 저녁 7시 문화활동 (19-4=15)

            # 교통 허브 (서울역, 용산, 교대 등) - 하루 종일 높은 수준
            '두정엽': (0.3, 0.6, 5),       # 오전 9시 (9-4=5), 작은 변화폭, 높은 베이스라인

            # 관광지 (명동, 이태원, 동대문 등)
            '후두엽': (0.5, 0.45, 10),     # 오후 2시 관광 피크 (14-4=10)

            # 주거지역 (노원, 상계, 송파 등)
            '소뇌_좌': (0.6, 0.35, 4),     # 오전 8시 출근 피크 (8-4=4)
            '소뇌_우': (0.6, 0.35, 14),    # 저녁 6시 귀가 피크 (18-4=14)

            # 외곽/신도시 (인천, 수원, 일산 등)
            '뇌간': (0.5, 0.4, 4)           # 오전 8시 서울 출근 피크 (8-4=4)
        }
        result = self.calculate_circadian_rhythm(self.circ_params_dict)
        return result

    def mix_colors(self, paid_ratio, free_ratio):
        """
        유임/무임 비율로 색상 혼합 (Neural Network 입력 통합)
            paid_ratio: 유임 가중치 (0-1)
            free_ratio: 무임 가중치 (0-1)
            np.array: RGB 색상 (R, G, B)
        """
        # 색상 혼합: y = w₁x₁ + w₂x₂
        mixed_color = paid_ratio * COLOR_PAID + free_ratio * COLOR_FREE
        return mixed_color

    def cal_color(self):
        result = {}
        # print("\n=== cal_color() 디버깅 ===")
        for regions_key ,regions_value in self.brain_regions.items():
            fee_total_rate = 0
            count = 0
            for station_name in regions_value['stations']:
                str_contains = self.fee_data[self.fee_data['역명'].str.contains(station_name, na=False, regex=False)]
                if len(str_contains) > 0:
                    # [행,열]
                    fee_column = str_contains['유임비율'].sum()
                    fee_total_rate += fee_column
                    count += len(str_contains)

            if count > 0:
                fee_result = fee_total_rate/count
                free_result = 1 - fee_result

                # print(f"{regions_key}: 유임={fee_result:.3f}, 무임={free_result:.3f}")

                color_of_region = self.mix_colors(fee_result, free_result)
                result[regions_key] = color_of_region
            else:
                print(f"{regions_key}: 데이터 없음!")

        return result

    def extract_line_connections(self):
        """
        호선별 데이터에서 뇌 영역 연결 자동 추출 (Neural Network Axon 매핑)
        1. 각 호선의 상위 5개 이용량 역 추출
        2. 각 역이 속한 뇌 영역 찾기
        3. 속한 뇌 영역이 2개 이상일 경우 뇌 영역 간 연결 생성 (축삭돌기 역할)
        """
        line_connections = {}
        for line_name in self.line:
            # 해당 호선의 역들 중 이용량 상위 5개 추출
            line_data = self.fee_data[self.fee_data['호선명'] == line_name]
            top_stations = line_data.nlargest(5, '총하차')

            # 각 역이 속한 뇌 영역 찾기
            regions_in_line = []
            for station_name in top_stations['역명']:
                for region_key in self.station_to_region.keys():
                    if region_key in station_name:
                        region = self.station_to_region[region_key]
                        # 중복 제거: 같은 뇌 영역은 한 번만 추가
                        if region not in regions_in_line:
                            regions_in_line.append(region)
                        break

            # 뇌 영역이 2개 이상 있어야 연결 가능
            if len(regions_in_line) >= 2:
                # 연속된 뇌 영역들을 연결
                connections = []
                for i in range(len(regions_in_line) - 1):
                    connection = (regions_in_line[i], regions_in_line[i+1])
                    connections.append(connection)

                line_connections[line_name] = connections

        return line_connections

    def calculate_axon_brightness(self, line_connections, mapped_totals):
        """
        축삭(연결선) 밝기 계산
        각 호선의 연결된 뇌 영역 역들의 시간대별 이용량 계산

        Args:
            line_connections: 호선별 뇌 영역 연결 정보
            mapped_totals: 전체 시간대별 합계 (calculate_bright에서 계산됨)

        반환: {line_name: {(region1, region2): [24시간 비율 리스트]}}
        """
        # 호선별 축삭 밝기 계산
        axon_brightness = {}

        for line_name, connections in line_connections.items():
            axon_brightness[line_name] = {}

            for connection in connections:
                region1, region2 = connection

                # 24시간 데이터 저장용
                connection_hour_totals = [0] * 24

                # 해당 호선에 속한 역들 가져오기
                line_data = self.fee_data[self.fee_data['호선명'] == line_name]

                for _, row in line_data.iterrows():
                    station_name = row['역명']

                    # 두 뇌 영역 중 하나에 속하는지 확인
                    in_region1 = any(s in station_name for s in self.brain_regions[region1]['stations']) # 포함되면 True
                    in_region2 = any(s in station_name for s in self.brain_regions[region2]['stations'])

                    if in_region1 or in_region2:
                        # time_data에서 해당 역의 시간대별 데이터 추출
                        station_rows = self.time_data[self.time_data['지하철역'].str.contains(station_name, na=False, regex=False)]
                        if len(station_rows) > 0:
                            time_colums = station_rows.iloc[:, 5:52:2].sum(axis=0)
                            for hour in range(24):
                                hour_total = float(time_colums.iloc[hour])
                                connection_hour_totals[hour] += hour_total

                # 비율 계산
                connection_ratios = []
                for hour in range(24):
                    if mapped_totals[hour] > 0:
                        ratio = connection_hour_totals[hour] / mapped_totals[hour]
                    else:
                        ratio = 0
                    connection_ratios.append(ratio)

                axon_brightness[line_name][connection] = connection_ratios

        return axon_brightness


    def calculate_bright(self):
        """
        1. 밝기 가중치(시간대 데이터)
            해당 뇌 영역에 해당하는 역들의 시간대별 총 인원/ 전체 뇌 영역에 해당하는 역들의 시간대별 총 인원
        """

        all_hours_data = {hour: {} for hour in range(24)}
        mapped_totals = [0] * 24  # 각 시간대별 전체 합

        # 뇌 영역 반복문
        for region_name, region_info in self.brain_regions.items():
            # 24시간 데이터 저장용
            region_hour_totals = [0] * 24

            # 해당 뇌 영역의 모든 역 반복문
            for station in region_info['stations']:
                station_rows = self.time_data[self.time_data['지하철역'].str.contains(station, na=False, regex=False)]
                if len(station_rows) > 0:
                    #[행,열]
                    time_colums = station_rows.iloc[:, 5:52:2].sum(axis=0) # 하차 기준 데이터만 뽑아서 사용
                    for hour in range(24):
                        hour_total = float(time_colums.iloc[hour])
                        region_hour_totals[hour] += hour_total # 특정 뇌 영역 total
                        mapped_totals[hour] += hour_total # 전체 뇌 영역 total
            # 각 시간대에 데이터 저장
            for hour in range(24):
                all_hours_data[hour][region_name] = {'total': region_hour_totals[hour]}


        # 비율 계산
        for hour in range(24):
            for region_name in self.brain_regions.keys():
                total = all_hours_data[hour][region_name]['total']
                if mapped_totals[hour] > 0:
                    ratio = total / mapped_totals[hour]
                else:
                    ratio = 0  # 데이터 없으면 0
                all_hours_data[hour][region_name]['ratio'] = ratio

        return all_hours_data, mapped_totals  # mapped_totals도 함께 반환

    def visualize_brain(self):
        """
        뇌 신경망 시각화 (24시간 애니메이션)
        - 뇌 영역 = Circle (색상 + 밝기)
        - 호선 연결 = Line (축삭돌기)
        - 실시간 시간대 업데이트
        """
        # 데이터 준비
        print("데이터 준비 중...")
        region_colors = self.cal_color()  # 뇌 영역별 색상
        brightness_data, mapped_totals = self.calculate_bright()  # 24시간 밝기 데이터, 전체 합계
        circadian_intensity = self.get_time_intensity()  # 일주기 리듬
        line_connections = self.extract_line_connections()  # 호선 연결
        axon_brightness = self.calculate_axon_brightness(line_connections, mapped_totals)  # 축삭 밝기

        # Figure 설정
        fig, ax = plt.subplots(figsize=(14, 12), facecolor='black')

        def update(frame):
            """
            프레임 업데이트 함수
            frame: 0~23 (시간, 4시~3시)
            """
            # 실제 시간 계산 (인덱스 0 = 4시)
            actual_hour = (frame + 4) % 24
            ax.clear() # 초기화, (frame 갱신)
            ax.set_xlim(0, 1) # x축, y축 (0, 1) Axes(축) 그리는 범위 설정, 자동조절 방지
            ax.set_ylim(0, 1)
            ax.set_aspect('equal') # x,y 비율 일정
            ax.axis('off') # matplotlib 기본 값 - 축/ 눈금/ 테두리 숨김
            ax.set_facecolor('black')

            # 1. 호선 연결선(축삭)
            for line_name, connections in line_connections.items():
                for region1, region2 in connections:
                    if region1 in self.brain_regions and region2 in self.brain_regions:
                        pos1 = self.brain_regions[region1]['pos']
                        pos2 = self.brain_regions[region2]['pos']

                        # 축삭 밝기 가져오기 (시간대별 연결 강도)
                        connection = (region1, region2)
                        if line_name in axon_brightness and connection in axon_brightness[line_name]:
                            # 해당 시간의 축삭 활성도 (비율)
                            axon_ratio = axon_brightness[line_name][connection][frame]

                            # 축삭 밝기 및 두께 계산
                            axon_alpha = 0.1 + 0.7 * axon_ratio
                            axon_width = 0.5 + 3.5 * axon_ratio

                            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                                   color='cyan', alpha=axon_alpha, linewidth=axon_width, zorder=1)
                        else:
                            # 데이터 없으면 기본 회색 선
                            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                                   color='gray', alpha=0.2, linewidth=0.5, zorder=1)

            # 2. 뇌 영역 원 그리기
            for region_name, region_info in self.brain_regions.items():
                pos = region_info['pos']

                # 기본 색상
                base_color = region_colors[region_name]

                # 밝기 가중치 (실제 이용량 비율)
                brightness_ratio = brightness_data[frame][region_name]['ratio']

                # 일주기 리듬 가중치
                circadian_weight = circadian_intensity[region_name][frame]

                # 최종 색상 = 기본색 × 밝기 × 일주기
                final_brightness = brightness_ratio * circadian_weight
                # [R, G, B] 배열에 각각 밝기 가중치의 곱 -> 색상 밝기 [R*a,G*a,B*a] 일때 a는 색상 밝기 가중치
                final_color = base_color * final_brightness
                final_color = np.clip(final_color, 0, 1) # matpolotlib -> 0~1 색상만 지원

                # 원 크기 (활성도에 비례)
                circle_size = 0.05 + 0.03 * final_brightness

                # 원 만들기
                circle = Circle(pos, circle_size, color=final_color,
                               alpha=0.8, zorder=2)
                # 원 그리기 목록 추가
                ax.add_patch(circle)

                # 텍스트 (뇌 영역 이름), ha,va -> 가로, 세로 정렬
                ax.text(pos[0], pos[1], region_name.replace('_', '\n'),
                       ha='center', va='center', fontsize=8,
                       color='white', weight='bold', zorder=3)

            # 3. 시간 표시
            time_text = f'{actual_hour:02d}:00'
            ax.text(0.5, 0.95, time_text, ha='center', va='top',
                   fontsize=40, color='white', weight='bold')

            # 4. 범례
            ax.text(0.05, 0.02, '하늘색: 유임승차 | 보라색: 무임승차',
                   fontsize=15, color='white', alpha=0.7)

        print("시작")
        plt.ion()  # 화면 갱신용

        frame = 0
        try:
            while True:  # 무한 루프
                update(frame)
                plt.draw()
                plt.pause(2)  # 2초 대기 (렌더링 포함)
                frame = (frame + 1) % 24  # 24 프레임 순환
        except KeyboardInterrupt:
            print("\n애니메이션 중지")
            plt.ioff()
            plt.close()

def main():
    """
    메인 실행 함수
    """
    # 파일 경로 설정
    fee_path = 'subwayfee.csv'
    time_path = 'subwaytime.csv'

    print("=" * 50)
    print("서울 지하철 Neural Network 시각화")
    print("=" * 50)
    #클래스 선언
    brain = Neural_transit_brain(fee_path, time_path)

    # 시각화 시작
    print("\n시각화를 시작합니다...")
    brain.visualize_brain()



main()
