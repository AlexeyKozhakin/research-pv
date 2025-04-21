import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Настройки
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 14
LINE_WIDTH = 3


#utils
import pandas as pd

def get_column_pv(data_one_user, modes=['pv0_90', 'pv0_90_b', 'pv0_max']):

    # Сортируем на всякий случай по дате
    data_one_user = data_one_user.sort_values(by='d').reset_index(drop=True)

    # Создаем пустой список для хранения результатов
    player_values = {}
    for mode in modes:
        player_values[mode] = []

    # Проходим по всем строкам и вызываем функцию на срезе до текущей строки включительно
    for i in range(len(data_one_user)):
        current_slice = data_one_user.iloc[:i+1]
        pv = predict_pv_exp_full(current_slice)
        for mode in modes:
            player_values[mode].append(pv[mode])

    # Добавляем результат в колонку
    for mode in modes:
        data_one_user[f'player_value_{mode}'] = player_values[mode]
    return data_one_user

import pandas as pd

def get_column_pv_true(data_one_user, modes=['pv0_90', 'pv0_90_b', 'pv0_max']):

    # Сортируем на всякий случай по дате
    data_one_user = data_one_user.sort_values(by='d').reset_index(drop=True)

    # Создаем пустой список для хранения результатов
    player_values = {}
    for mode in modes:
        player_values[mode] = []

    # Проходим по всем строкам и вызываем функцию на срезе до текущей строки включительно
    for i in range(len(data_one_user)):
        current_slice = data_one_user.iloc[:i+1]
        pv = predict_pv_net_dep_true(current_slice)
        for mode in modes:
            player_values[mode].append(pv[mode])

    # Добавляем результат в колонку
    for mode in modes:
        data_one_user[f'player_value_{mode}'] = player_values[mode]
    return data_one_user


#MODEL
def moving_average(df, window=3):
    """Computes a moving average for the monthly difference."""
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.sort_values('date')
    df['moving_avg'] = df['difference'].rolling(window=window, min_periods=1).mean()
    return df[['year', 'month', 'difference', 'moving_avg']]


def monthly_difference(df):
    """Calculates monthly differences in 'red' values."""
    # Убедимся, что 'date' в нужном формате
    df['date'] = pd.to_datetime(df['date'])

    # Получаем максимальное значение 'red' по каждому месяцу
    monthly_max = df.groupby(df['date'].dt.to_period('M'))['red'].max()

    # Преобразуем в DataFrame и сбросим индекс
    monthly_max = monthly_max.reset_index()
    monthly_max['date'] = monthly_max['date'].dt.to_timestamp()

    # Вычисляем разницу с предыдущим месяцем
    monthly_max['difference'] = monthly_max['red'].diff().fillna(monthly_max['red'])

    # Добавим год и месяц как отдельные колонки (если нужно)
    monthly_max['year'] = monthly_max['date'].dt.year
    monthly_max['month'] = monthly_max['date'].dt.month

    # Результат
    result = monthly_max[['date', 'year', 'month', 'red', 'difference']]
    return result[['year', 'month', 'difference']]

def predict_player_full_info(data_one_user):
    """Calculates numerical player value metrics."""
    result_df = monthly_difference(data_one_user)
    #print('results_df', result_df)
    moving_avg_df = moving_average(result_df)
    #print('Moving average',moving_avg_df)
    mean_pv = moving_avg_df['moving_avg'].mean()
    std_pv = moving_avg_df['moving_avg'].std()
    threshold = mean_pv + 1.28 * std_pv
    filtered = moving_avg_df[moving_avg_df['moving_avg'] <= threshold]
    pv0_90 = filtered['moving_avg'].max()
    pv0_90_b = threshold
    pv0_max = moving_avg_df['moving_avg'].max()
    dpv = float(data_one_user['red'].iloc[-1] - data_one_user['blue'].iloc[-1])
    #pv_num = pv0 + dpv
    #return {"pv": pv_num, "pv0_90": pv0_90, "pv0_90_b": pv0_90_b, "dpv": dpv}
    return {"pv0_90": pv0_90, "pv0_90_b": pv0_90_b, "pv0_max": pv0_max}

def get_red_line_for_true_model_net_dep(data_one_user):
    # Сортировка и расчёт blue
    data_one_user = data_one_user.sort_values(by='d').reset_index(drop=True)

    # Копируем первую строку
    fake_row = data_one_user.iloc[[0]].copy()

    # Заменяем нужные поля
    fake_date = data_one_user['d'].min() - pd.Timedelta(days=1)
    fake_row['d'] = fake_date
    fake_row['sum_of_deposits'] = 0
    fake_row['sum_of_withdrawals'] = 0

    # Объединяем и пересчитываем red
    data_one_user = pd.concat([fake_row, data_one_user], ignore_index=True)


    data_one_user['net_dep'] = data_one_user['sum_of_deposits'] - data_one_user['sum_of_withdrawals']
    data_one_user['blue'] = data_one_user['net_dep'].cumsum()
    data_one_user['red'] = data_one_user['blue'].cummax()
    data_one_user['date'] = pd.to_datetime(data_one_user['d'])

    return data_one_user


def get_red_line_exp_dec(data_one_user):
    # Сортировка по дате
    data_one_user = data_one_user.sort_values(by='d').reset_index(drop=True)

    # Копируем первую строку
    fake_row = data_one_user.iloc[[0]].copy()

    # Заменяем нужные поля
    fake_date = data_one_user['d'].min() - pd.Timedelta(days=1)
    fake_row['d'] = fake_date
    fake_row['sum_of_deposits'] = 0
    fake_row['sum_of_withdrawals'] = 0

    # Объединяем и пересчитываем red
    data_one_user = pd.concat([fake_row, data_one_user], ignore_index=True)

    df = data_one_user.sort_values('d').reset_index(drop=True)

    # Расчёт net_dep
    df['net_dep'] = df['sum_of_deposits'] - df['sum_of_withdrawals']
    df['net_dep_cum_sum'] = df['net_dep'].cumsum()

    # Преобразуем даты в массив
    dates = df['d'].values.astype('datetime64[D]')
    date_diff_matrix = (dates[:, None] - dates[None, :]).astype('timedelta64[D]').astype(int)

    # Мы хотим только те, где i <= j (прошлое к текущему)
    mask = date_diff_matrix >= 0

    # Параметр затухания
    tau = 60

    # Вычисляем веса затухания
    decay_weights = np.exp(-date_diff_matrix / tau) * mask

    # Вектор net_dep
    net_dep_vector = df['net_dep'].values

    # Вычисляем экспоненциально-взвешенную сумму
    net_dep_cum_exp = decay_weights @ net_dep_vector

    # Добавляем в датафрейм
    df['net_dep_cum_exp'] = net_dep_cum_exp
    df['blue'] = df['net_dep_cum_exp']
    df['red'] = df['blue'].cummax()
    data_one_user = df
    data_one_user['date'] = pd.to_datetime(data_one_user['d'])
    return data_one_user

def get_red_line_adapt_exp_dec(data_one_user):
    # Сортировка по дате
    data_one_user = data_one_user.sort_values(by='d').reset_index(drop=True)

    # Копируем первую строку
    fake_row = data_one_user.iloc[[0]].copy()

    # Заменяем нужные поля
    fake_date = data_one_user['d'].min() - pd.Timedelta(days=1)
    fake_row['d'] = fake_date
    fake_row['sum_of_deposits'] = 0
    fake_row['sum_of_withdrawals'] = 0

    # Объединяем и пересчитываем red
    data_one_user = pd.concat([fake_row, data_one_user], ignore_index=True)

    df = data_one_user.sort_values('d').reset_index(drop=True)

    # Расчёт net_dep
    df['net_dep'] = df['sum_of_deposits'] - df['sum_of_withdrawals']
    df['net_dep_cum_sum'] = df['net_dep'].cumsum()

    # Преобразуем даты в массив
    dates = df['d'].values.astype('datetime64[D]')
    date_diff_matrix = (dates[:, None] - dates[None, :]).astype('timedelta64[D]').astype(int)

    # Мы хотим только те, где i <= j (прошлое к текущему)
    mask = date_diff_matrix >= 0

    # Параметр затухания
    tau = 60

    # Вычисляем веса затухания
    decay_weights = np.exp(-date_diff_matrix / tau) * mask

    # Вектор net_dep
    net_dep_vector = df['net_dep'].values
    dep_vector = df['sum_of_deposits'].cumsum().values
    #withd_vector = df['sum_of_withdrawals'].cumsum().values
    #withd_vector = df['sum_of_withdrawals'].cumsum().values
    withd_vector = (decay_weights * df['sum_of_withdrawals'].values[None, :]).sum(axis=1)

    # Вычисляем экспоненциально-взвешенную сумму
    #net_dep_cum_exp = dep_vector - (withd_vector @ decay_weights)
    net_dep_cum_exp = dep_vector - withd_vector

    # Добавляем в датафрейм
    df['net_dep_cum_exp'] = net_dep_cum_exp
    df['blue'] = df['net_dep_cum_exp']
    df['red'] = df['blue'].cummax()
    data_one_user = df
    data_one_user['date'] = pd.to_datetime(data_one_user['d'])
    return df


def predict_pv_exp_full(data_one_user):
    
    data_one_user = get_red_line_exp_dec(data_one_user)
    #result = predict_player_full_info(data_one_user)
    result = {"pv0_90": data_one_user['red'].max()/2, "pv0_90_b": data_one_user['red'].max()/2, "pv0_max": data_one_user['red'].max()/2}
    return result

def predict_pv_net_dep_true(data_one_user):
    
    data_one_user = get_red_line_for_true_model_net_dep(data_one_user)
    result = predict_player_full_info(data_one_user)
    
    return result

#Load Data
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Константы визуализации
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 14
LINE_WIDTH = 3

# Загрузка данных
df_payments = pd.read_csv('/Users/alexeykozhakin/PycharmProjects/ft-ds-player-value/notebooks-tutorials/data/df_23_24_loyal.csv')
df_payments['d'] = pd.to_datetime(df_payments['d'], errors='coerce')

# 📌 Выбор пользователя
# Популярные user_ids
default_user_ids = [444104, 613, 6283, 150, 908096, 8294, 415681, 1272361]
options = default_user_ids + ["Other"]

# Выбор из списка
selected_option = st.selectbox("Choose User ID", options=options, index=0)

# Если выбрали "Other", даём поле для ввода
if selected_option == "Other":
    user_id = st.number_input("Enter custom User ID", min_value=1, step=1)
else:
    user_id = int(selected_option)

st.write(f"✅ Selected User ID: {user_id}")


# Display options in sidebar
st.sidebar.header("📌 Display Options")

# Red/Blue line toggles
show_true_blue = st.sidebar.checkbox("Net Dep (Blue Line)", value=True)
show_true_red = st.sidebar.checkbox("Net Dep (Red Line)", value=True)
show_exp_blue = st.sidebar.checkbox("Exp Decay (Blue Line)", value=True)
show_exp_red = st.sidebar.checkbox("Exp Decay (Red Line)", value=True)

# Deposits/Withdrawals
show_deposits = st.sidebar.checkbox("Deposits", value=True)
show_withdrawals = st.sidebar.checkbox("Withdrawals", value=True)

# Player Value
show_pv_m = st.sidebar.checkbox("Player Value Model", value=True)
show_pv_t = st.sidebar.checkbox("Player Value True", value=True)

# ------------------ Calculation Trigger ------------------
if st.button("🔁 Run Calculation"):

    # 📦 Load and filter data
    def filter_by_user_id(df, user_id):
        return df[df['user_id'] == user_id]

    df_payments = pd.read_csv('/Users/alexeykozhakin/PycharmProjects/ft-ds-player-value/notebooks-tutorials/data/df_23_24_loyal.csv')
    df_payments['d'] = pd.to_datetime(df_payments['d'], errors='coerce')
    data = filter_by_user_id(df_payments, user_id)

    # Models
    true_model = get_red_line_for_true_model_net_dep(data)
    exp_model = get_red_line_exp_dec(data)

    # ========== 1. Full Combined Chart ==========
    st.subheader("📊 Full Overview Chart")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    if show_true_blue:
        ax1.plot(true_model['date'], true_model['blue'], c='navy', label='Net Dep Blue', linewidth=LINE_WIDTH)
    if show_true_red:
        ax1.plot(true_model['date'], true_model['red'], c='tomato', label='Net Dep Red', linewidth=LINE_WIDTH)
    if show_exp_blue:
        ax1.plot(exp_model['date'], exp_model['blue'], c='blue', label='Exp Decay Blue', linewidth=LINE_WIDTH)
    if show_exp_red:
        ax1.plot(exp_model['date'], exp_model['red'], c='red', label='Exp Decay Red', linewidth=LINE_WIDTH)
    if show_deposits:
        ax2.vlines(data['d'], ymin=0, ymax=data['sum_of_deposits'], color='green', alpha=0.6, label='Deposits')
    if show_withdrawals:
        ax2.vlines(data['d'], ymin=-data['sum_of_withdrawals'], ymax=0, color='orange', alpha=0.6, label='Withdrawals')
    if show_pv_m:
        data = get_column_pv(data)
        ax2.plot(data['d'], data['player_value_pv0_max'], color='m', label='PV Model Max', linewidth=LINE_WIDTH)
    if show_pv_t:
        data = get_column_pv_true(data)
        ax2.plot(data['d'], data['player_value_pv0_90'], color='g', label='True PV Q90', linestyle='--', linewidth=LINE_WIDTH)
        ax2.plot(data['d'], data['player_value_pv0_90_b'], color='g', label='True PV Q90 Bound', linestyle=':', linewidth=LINE_WIDTH)
        ax2.plot(data['d'], data['player_value_pv0_max'], color='g', label='True PV Max', linewidth=LINE_WIDTH)

    ax1.set_title("All Lines", fontsize=TITLE_FONTSIZE)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Net Dep / Exp Decay")
    ax2.set_ylabel("Amounts / PV")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax1.grid(True)
    st.pyplot(fig)

    # ========== 2. Red/Blue Lines Only ==========
    st.subheader("🔵 Red/Blue Net Dep & Exp Decay")
    fig2, ax = plt.subplots(figsize=(14, 5))
    if show_true_blue:
        ax.plot(true_model['date'], true_model['blue'], c='navy', label='Net Dep Blue')
    if show_true_red:
        ax.plot(true_model['date'], true_model['red'], c='deeppink', label='Net Dep Red')
    if show_exp_blue:
        ax.plot(exp_model['date'], exp_model['blue'], c='blue', label='Exp Decay Blue')
    if show_exp_red:
        ax.plot(exp_model['date'], exp_model['red'], c='red', label='Exp Decay Red')
    ax.set_title("Net Dep & Exp Decay Lines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig2)

    # ========== 3. Deposits & Withdrawals Only ==========
    st.subheader("💰 Deposits and Withdrawals")
    fig3, ax = plt.subplots(figsize=(14, 4))
    if show_deposits:
        ax.vlines(data['d'], ymin=0, ymax=data['sum_of_deposits'], color='green', alpha=0.6, label='Deposits')
    if show_withdrawals:
        ax.vlines(data['d'], ymin=-data['sum_of_withdrawals'], ymax=0, color='orange', alpha=0.6, label='Withdrawals')
    ax.set_title("Deposits / Withdrawals")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig3)

    # ========== 4. Player Value Only ==========
    st.subheader("📈 Player Value Evolution")
    fig4, ax = plt.subplots(figsize=(14, 5))
    if show_pv_m:
        data = get_column_pv(data)
        ax.plot(data['d'], data['player_value_pv0_max'], color='m', label='Model PV Max')
    if show_pv_t:
        data = get_column_pv_true(data)
        ax.plot(data['d'], data['player_value_pv0_90'], color='g', linestyle='--', label='True PV Q90')
        ax.plot(data['d'], data['player_value_pv0_90_b'], color='g', linestyle=':', label='True PV Q90 Bound')
        ax.plot(data['d'], data['player_value_pv0_max'], color='g', label='True PV Max')
    ax.set_title("Player Value Lines")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig4)

else:
    st.info("👆 Select a user and customize display options, then press '🔁 Run Calculation'")