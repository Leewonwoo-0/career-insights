# 프로그래밍 실습과제 1

# # 기후 데이터 기반 30일 이후 기온 추론 시스템
# ### 1. 사용 수식
# - 경사하강법, Sin, Cos 기반 계절 가중치를 기반으로 데이터를 학습시켜, 30일 이후의 기온 데이터를 추론하는 프로그램을 개발하였습니다.
# - 수식은 Claude Opus 4.1 모델의 도움을 받아 작성하였습니다.
import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Malgun Gothic"
PATH = "seoul.csv"
df = pd.read_csv(PATH, encoding="cp949")
# 컬럼 이름 직접 지정 (첫 번째가 빈 컬럼)
df.columns = ['빈칸', '날짜', '지점', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)']

# 필요한 컬럼만 선택
df = df[['날짜', '최고기온(℃)', '최저기온(℃)']]

# 날짜 변환 및 정렬, 기존 구현방식과 같이 csv 모듈을 이용하여 구현하려 했으나,
# 데이터 가공중 코드가 너무 길어져 pandas를 이용하여 제작하였습니다.
df['날짜'] = pd.to_datetime(df['날짜'], errors="coerce")
#숫자가 아니면 버리는 함수
df = df.dropna()
#날짜 정렬함수, drop-> 기존 인덱스를 버리고 새로운 인덱스로 저장 drop=True
df = df.sort_values('날짜').reset_index(drop=True)

print(f"데이터 로드 완료: {len(df)}개 행")
print(f"날짜 범위: {df['날짜'].min()} ~ {df['날짜'].max()}")

# 이후 모델링 코드
if len(df) >= 2:
    doy = df['날짜'].dt.dayofyear.to_numpy()
    sin_seasonal = np.sin(2*np.pi*doy/365.0)
    cos_seasonal = np.cos(2*np.pi*doy/365.0)

#X에 칼럼스택으로 저장  최고, 최저, 사인, 코사인, 절편(1)
#Y는 예측 값
    X = np.column_stack([
        df['최고기온(℃)'].values[:-1],
        df['최저기온(℃)'].values[:-1],
        sin_seasonal[:-1],
        cos_seasonal[:-1],
        np.ones(len(df)-1)
    ])
    Y = np.column_stack([
        df['최고기온(℃)'].values[1:],
        df['최저기온(℃)'].values[1:]
    ])

#그라디언트(경사하강법)
#데이터에 가중치 행렬연산을 적용한 후 예측 데이터와 실제 데이터를 비교하여
#에러 크기에 따라 가중치를 새로 갱신하는 형태의 연산을 사용한다. 
    def gradient_descent(X, Y, learning_rate=0.0001, iterations=10000):
        m = X.shape[0] # 표본 개수 m
        n = X.shape[1] # 특성 수 n
        #가중치 행렬 n*2로 생성, 초기값은 난수
        theta = np.random.randn(n, 2) * 0.01

        # 표본 x에 경사하강 시행 수(iterations) 만큼 업데이트
        # X, theta 행렬연산
        for i in range(iterations):
            #예측 값
            predictions = X @ theta
            #예측 - 실제
            errors = predictions - Y
            #기울기
            gradient = (1/m) * X.T @ errors
            #업데이트
            theta = theta - learning_rate * gradient

            if i % 1000 == 0:
                cost = np.mean(errors**2)
                print(f"반복 {i}: Cost = {cost:.4f}")

        return theta

    print("\n경사하강법 학습...")
    theta_gd = gradient_descent(X, Y, learning_rate=0.0001, iterations=5000)



# numpy에 최소제곱법 학습방식에 대한 함수를 제공하기에 추가로 사용해 보았습니다.
# 최소제곱이란, 선형모형에서 정확히 일치하는 세타(가중치)를 못 찾는 경우, 잔차(실제값-예측값)의 제곱합이 가장 작아지게 만드는 세타를 의미한다.
# 이는 경사하강법과 달리 선형대수 분해를 기반으로 바로 해를 구한다는 특성을 갖고 있다.
    
    print("\n최소제곱법 학습...")
    theta_ls, *_ = np.linalg.lstsq(X, Y, rcond=None)

    print("\n경사하강법 가중치:")
    print(theta_gd)
    print("\n최소제곱법 가중치:")
    print(theta_ls)

    theta = theta_ls

    # 미래 예측
    H = 30
    last_date = df['날짜'].iloc[-1]
    # 30일 데이터 생성
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=H)
    #라벨기반으로 해석하기에, iloc기반으로 스칼라 값 접근
    current_tmax = float(df['최고기온(℃)'].iloc[-1])
    current_tmin = float(df['최저기온(℃)'].iloc[-1])

    pred_tmax, pred_tmin = [], []

    # 실제 예측 (H=30일)
    for k in range(H):
        doy_k = future_dates[k].timetuple().tm_yday
        sin_k = np.sin(2*np.pi*doy_k/365.0)
        cos_k = np.cos(2*np.pi*doy_k/365.0)

        x_current = np.array([current_tmax, current_tmin, sin_k, cos_k, 1.0])
        y_pred = x_current @ theta

        current_tmax = float(y_pred[0])
        current_tmin = float(y_pred[1])

        pred_tmax.append(current_tmax)
        pred_tmin.append(current_tmin)

    # 그래프 그리기
    N_recent = min(100, len(df))
    recent_dates = df['날짜'].iloc[-N_recent:]
    recent_tmax = df['최고기온(℃)'].iloc[-N_recent:].to_numpy()
    recent_tmin = df['최저기온(℃)'].iloc[-N_recent:].to_numpy()

    time_idx_recent = np.arange(len(recent_dates))
    time_idx_future = np.arange(len(recent_dates), len(recent_dates) + H)

    # 5일 이동평균
    def moving_average(data, window=5):
        if len(data) < window:
            return data
        #convolve 함수는 1차원 배열의 이산 선형 컨볼루션을 계산하는 넘파이의 함수이다
        #이산 선형 컨볼루션을 적용한 이유는 5개의 데이터 단위 평균값을 구해 그래프로 표현하기 위해 사용하였다.
        #np.ones(window)/window는 각각의 값에 해당하는 가중치 곱을 의미한다, n번째 데이터/Window(총 데이터 개수)
        #mode= valid를 통해 연산가능한 값이 있는 배열만 포함하도록 하였다.
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        padding = np.full(window-1, smoothed[0])
        return np.concatenate([padding, smoothed])

    recent_tmax_smooth = moving_average(recent_tmax, 5)
    recent_tmin_smooth = moving_average(recent_tmin, 5)

    #화면 사이즈 정의 및 plot 3d 구현을 위해 축 추, 111-> 1행 x 1열 그리그 1번째 칸
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

#b- -> 파란색 + 실선(solid), linewidth -> 선 두께, alpha -> 불투명도
# 구한 값을 그래프로 표현

# 5일 이동평균 궤적
    ax.plot(time_idx_recent, recent_tmax_smooth, recent_tmin_smooth,
            'b-', linewidth=2.5, alpha=0.8, label=f"최근 {N_recent}일 실제 데이터 (5일 이동평균)")

#원시 데이터 궤적
    ax.plot(time_idx_recent, recent_tmax, recent_tmin,
            'b-', linewidth=0.5, alpha=0.2)

# 관측 마지막 점
    ax.scatter(time_idx_recent[-1], recent_tmax[-1], recent_tmin[-1],
               color='green', s=100, marker='o', label="마지막 관측일")

# 예측 구간 궤적
    ax.plot(time_idx_future, pred_tmax, pred_tmin,
            'r-', linewidth=2.5, alpha=0.8, label=f"미래 {H}일 예측")

# 관측 마지막점, 예측 첫 점 점선 연결
    ax.plot([time_idx_recent[-1], time_idx_future[0]],
            [recent_tmax[-1], pred_tmax[0]],
            [recent_tmin[-1], pred_tmin[0]],
            'g--', linewidth=1.5, alpha=0.6)

    # X축 라벨을 월 단위로 설정
    all_dates = pd.concat([recent_dates, pd.Series(future_dates)])
    all_dates.reset_index(drop=True, inplace=True)

    tick_positions = []
    tick_labels = []
    last_month = None

    for i, date in enumerate(all_dates):
        if last_month != date.month:
            tick_positions.append(i)
            tick_labels.append(date.strftime('%Y-%m'))
            last_month = date.month

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    ax.set_xlabel('날짜', fontsize=11)
    ax.set_ylabel('최고기온(℃)', fontsize=11)
    ax.set_zlabel('최저기온(℃)', fontsize=11)
    ax.set_title(f'서울시 일별 기온: 최근 {N_recent}일 실제 vs 미래 {H}일 예측\n(계절성 포함 선형모델)', fontsize=13)

    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.ion()
    plt.tight_layout()
    plt.show()

    print(f"\n예측 완료")
    print(f"최근 날짜: {last_date.strftime('%Y-%m-%d')}")
    print(f"예측 기간: {future_dates[0].strftime('%Y-%m-%d')} ~ {future_dates[-1].strftime('%Y-%m-%d')}")
    print(f"\n예측 결과 (처음 5일):")
    for i in range(min(5, H)):
        print(f"  {future_dates[i].strftime('%Y-%m-%d')}: 최고 {pred_tmax[i]:.1f}℃, 최저 {pred_tmin[i]:.1f}℃")
