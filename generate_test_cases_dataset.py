import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- Параметры генерации ---
num_users = 1000
max_days = 365
start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
deposits_days_range = (180, 365)
withdrawals_days_range = (50, 100)

# --- Ручные тест-кейсы ---
manual_cases1 = [
    {'user_id': 1, 'date': '2024-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 1, 'date': '2024-02-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0}
    ]

manual_cases2 = [
    {'user_id': 2, 'date': '2024-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 2, 'date': '2024-02-28', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0}
]

manual_cases3 = [
    {'user_id': 3, 'date': '2024-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 3, 'date': '2024-02-28', 'sum_of_deposits': 100.0, 'sum_of_withdrawals': 0.0}
]

manual_cases4 = [
    {'user_id': 4, 'date': '2024-01-01', 'sum_of_deposits': 100.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 4, 'date': '2024-02-28', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0}
]


manual_cases5 = [
    {'user_id': 5, 'date': '2024-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 5, 'date': '2024-02-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 5, 'date': '2024-03-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},

    {'user_id': 5, 'date': '2025-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 5, 'date': '2025-02-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 5, 'date': '2025-03-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},        
]

manual_cases55 = [
    {'user_id': 55, 'date': '2024-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 55, 'date': '2024-02-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 55, 'date': '2024-03-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},

    {'user_id': 55, 'date': '2024-04-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 55, 'date': '2024-05-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 55, 'date': '2024-06-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},        
]

manual_cases555 = [
    {'user_id': 555, 'date': '2024-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 555, 'date': '2024-02-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 555, 'date': '2024-03-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},

    {'user_id': 555, 'date': '2024-04-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 555, 'date': '2024-05-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 555, 'date': '2024-06-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},        

    {'user_id': 555, 'date': '2024-07-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 555, 'date': '2024-08-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 555, 'date': '2024-09-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},            

    {'user_id': 555, 'date': '2024-10-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 555, 'date': '2024-11-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 555, 'date': '2024-12-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},            
]

manual_cases6 = [
    {'user_id': 6, 'date': '2023-01-01', 'sum_of_deposits': 1000.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 6, 'date': '2023-02-01', 'sum_of_deposits': 1000.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 6, 'date': '2023-03-01', 'sum_of_deposits': 1000.0, 'sum_of_withdrawals': 0.0},

    {'user_id': 6, 'date': '2024-01-01', 'sum_of_deposits': 100.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 6, 'date': '2024-02-01', 'sum_of_deposits': 100.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 6, 'date': '2024-03-01', 'sum_of_deposits': 100.0, 'sum_of_withdrawals': 0.0},        

    {'user_id': 6, 'date': '2025-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 6, 'date': '2025-02-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 6, 'date': '2025-03-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},        
]

manual_cases7 = [
    {'user_id': 7, 'date': '2023-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 7, 'date': '2023-02-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 7, 'date': '2023-03-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},

    {'user_id': 7, 'date': '2024-01-01', 'sum_of_deposits': 100.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 7, 'date': '2024-02-01', 'sum_of_deposits': 100.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 7, 'date': '2024-03-01', 'sum_of_deposits': 100.0, 'sum_of_withdrawals': 0.0},        

    {'user_id': 7, 'date': '2025-01-01', 'sum_of_deposits': 1000.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 7, 'date': '2025-02-01', 'sum_of_deposits': 1000.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 7, 'date': '2025-03-01', 'sum_of_deposits': 1000.0, 'sum_of_withdrawals': 0.0},        
]

# Базовые транзакции
manual_cases_special1 = [
    {'user_id': 8, 'date': '2023-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 8, 'date': '2023-02-01', 'sum_of_deposits': 0.0, 'sum_of_withdrawals': 1000.0},
]

# Добавим ежедневные крошечные депозиты в течение 1 года
start_date = pd.to_datetime("2023-03-01")
end_date = start_date + pd.Timedelta(days=365)

dates = pd.date_range(start=start_date, end=end_date, freq='D')
tiny_deposits = [
    {'user_id': 8, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 1e-10, 'sum_of_withdrawals': 0.0}
    for d in dates
]

# Объединяем всё в один список
manual_cases_special1 += tiny_deposits

# Базовые транзакции
manual_cases_special2 = [
    {'user_id': 9, 'date': '2023-01-01', 'sum_of_deposits': 10.0, 'sum_of_withdrawals': 0.0},
    {'user_id': 9, 'date': '2023-02-01', 'sum_of_deposits': 0.0, 'sum_of_withdrawals': 10_000.0},
]

# Добавим ежедневные крошечные депозиты в течение 1 года
start_date = pd.to_datetime("2023-03-01")
end_date = start_date + pd.Timedelta(days=365)

dates = pd.date_range(start=start_date, end=end_date, freq='D')
tiny_deposits = [
    {'user_id': 9, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 1e-10, 'sum_of_withdrawals': 0.0}
    for d in dates
]
# Объединяем всё в один список
manual_cases_special2 += tiny_deposits

# Добавим ежедневные крошечные депозиты в течение 1 года
start_date = pd.to_datetime("2024-01-01")
end_date = start_date + pd.Timedelta(days=180)

dates = pd.date_range(start=start_date, end=end_date, freq='D')
daily_deposits = [
    {'user_id': 10, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 10, 'sum_of_withdrawals': 0.0}
    for d in dates
]
# Объединяем всё в один список
manual_cases_special1 += tiny_deposits

# Добавим ежедневные крошечные депозиты в течение 1 года
start_date = pd.to_datetime("2024-01-01")
end_date = start_date + pd.Timedelta(days=180)

dates = pd.date_range(start=start_date, end=end_date, freq='D')
daily_deposits = [
    {'user_id': 10, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 10, 'sum_of_withdrawals': 0.0}
    for d in dates
]

# Даты для каждого user_id
dates1000 = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
dates100 = pd.date_range(start="2024-01-01", end="2024-12-31", freq='D')
dates10 = pd.date_range(start="2025-01-01", end="2025-12-31", freq='D')

# Генерация данных
daily_deposits_1000_100_10 = (
    [
        {'user_id': 60, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 1000, 'sum_of_withdrawals': 0.0}
        for d in dates1000
    ] +
    [
        {'user_id': 60, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 100, 'sum_of_withdrawals': 0.0}
        for d in dates100
    ] +
    [
        {'user_id': 60, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 10, 'sum_of_withdrawals': 0.0}
        for d in dates10
    ]
)



# Даты для каждого user_id
dates1000 = pd.date_range(start="2022-01-01", end="2022-12-31", freq='D')
dates100 = pd.date_range(start="2024-01-01", end="2024-12-31", freq='D')
dates10 = pd.date_range(start="2026-01-01", end="2026-12-31", freq='D')

# Генерация данных
daily_deposits_1000_100_10_with_break = (
    [
        {'user_id': 600, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 1000, 'sum_of_withdrawals': 0.0}
        for d in dates1000
    ] +
    [
        {'user_id': 600, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 100, 'sum_of_withdrawals': 0.0}
        for d in dates100
    ] +
    [
        {'user_id': 600, 'date': d.strftime('%Y-%m-%d'), 'sum_of_deposits': 10, 'sum_of_withdrawals': 0.0}
        for d in dates10
    ]
)



# Задаем дату начала и конца
start_date = pd.to_datetime("2024-01-01")
end_date = start_date + pd.Timedelta(days=180)
# Создаем даты с недельным шагом
dates = pd.date_range(start=start_date, end=end_date, freq='W')  # 'W' означает еженедельно (по воскресеньям)
# Генерация записей
weekly_deposits = [
    {
        'user_id': 11,
        'date': d.strftime('%Y-%m-%d'),
        'sum_of_deposits': 75,
        'sum_of_withdrawals': 0.0
    }
    for d in dates
]
# Преобразуем в DataFrame
df_weekly = pd.DataFrame(weekly_deposits)

manual_cases = (
    manual_cases1 +
    manual_cases2+
    manual_cases3+
    manual_cases4+
    manual_cases5+
    manual_cases55+
    manual_cases555+
    manual_cases6+
    daily_deposits_1000_100_10+
    daily_deposits_1000_100_10_with_break+
    manual_cases7+
    manual_cases_special1+
    manual_cases_special2+
    daily_deposits+
    weekly_deposits
               ) 


# --- Добавляем ручные кейсы ---
manual_df = pd.DataFrame(manual_cases)
manual_df['date'] = pd.to_datetime(manual_df['date'])  # строку → дату

# --- Собираем основной DataFrame ---
df = manual_df


df = df.sort_values(by=['user_id', 'date']).reset_index(drop=True)

# --- Сохраняем и выводим ---
print(df.head(10))
df.to_csv('payment_data_manual_test_cases.csv', index=False)