import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# параметры генерации
num_users = 1000
max_days = 365
start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
deposits_days_range = (180, 365)
withdrawals_days_range = (50, 100)

# функция для генерации дат событий
def generate_event_dates(start_date, max_days, event_days_count):
    days = np.arange(1, max_days + 1)
    selected_days = np.random.choice(days, size=event_days_count, replace=False)
    selected_days.sort()
    dates = [start_date + timedelta(days=int(d - 1)) for d in selected_days]
    return dates

# функция для генерации сумм по логнормальному распределению с рандомными параметрами
def generate_amounts(size, mu_range=(4, 6), sigma_range=(0.3, 0.8)):
    mu = np.random.uniform(*mu_range)
    sigma = np.random.uniform(*sigma_range)
    amounts = np.random.lognormal(mean=mu, sigma=sigma, size=size)
    return amounts

# список для накопления всех записей
records = []

# генерация данных по каждому пользователю
for user_id in range(1, num_users + 1):
    # депозиты
    num_deposits = np.random.randint(*deposits_days_range)
    deposit_dates = generate_event_dates(start_date, max_days, num_deposits)
    deposit_amounts = generate_amounts(num_deposits)

    for date, amount in zip(deposit_dates, deposit_amounts):
        records.append({
            'user_id': user_id,
            'date': date,
            'sum_of_deposit': round(amount, 2),
            'sum_of_withdraw': 0.0
        })

    # выводы
    num_withdrawals = np.random.randint(*withdrawals_days_range)
    withdrawal_dates = generate_event_dates(start_date, max_days, num_withdrawals)
    withdrawal_amounts = generate_amounts(num_withdrawals)

    for date, amount in zip(withdrawal_dates, withdrawal_amounts):
        records.append({
            'user_id': user_id,
            'date': date,
            'sum_of_deposit': 0.0,
            'sum_of_withdraw': round(amount, 2)
        })

# сборка в DataFrame
df = pd.DataFrame(records)

# агрегация по user_id и дате, чтобы объединить суммы на совпадающие даты
df = df.groupby(['user_id', 'date'], as_index=False).agg({
    'sum_of_deposit': 'sum',
    'sum_of_withdraw': 'sum'
})

# сортировка по user_id и дате
df = df.sort_values(by=['user_id', 'date']).reset_index(drop=True)

# выводим первые строки
print(df.head(10))

# сохраняем в CSV
df.to_csv('payment_data.csv', index=False)