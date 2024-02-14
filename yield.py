import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import numpy as np


def days_since_coupon(today, maturity_date):
    """
    Given the current date, and a maturity date,
    calculate the number of days since the last coupon payment
    """
    coupon_date = maturity_date
    while today < coupon_date:
        coupon_date = coupon_date + relativedelta(months=-6)

    return (today - coupon_date).days


def periods_to_maturity(today, maturity_date):
    """
    Given the current date, and a maturity date,
    calculate the number of periods until maturity.
    The function returns 1 when maturity is less than 6 months away
    """
    count = 0
    while today < maturity_date:
        today = today + relativedelta(months=+6)
        count += 1

    return count


def calc_dirty_price(clean_price, days_since, coupon_rate):
    """
    Given the clean price, number of days since the last coupon payment, and
    the coupon rate, calculate the dirty price of the bond.
    """
    return clean_price + ((days_since * coupon_rate * 100) / 365)


def bootstrap_yield_curve(bonds_df, day_idx):
    """
    Bootstrap the spot rate for bonds in bond_df using the price of the bond
    on date_list[day_idx]
    """
    # guess initial spot rate
    spot_rates = np.full(len(bonds_df), 0.1)
    sorted_bonds = bonds_df.sort_values(
        by='periods_to_maturity').reset_index(drop=True)

    for i, bond in sorted_bonds.iterrows():
        price = bond[date_list[day_idx]]
        coupon_rate = bond['coupon_rate']
        maturity = bond['periods_to_maturity']

        cash_flows = np.array(
            [coupon_rate] * int(maturity - 1) + [100 + coupon_rate])
        time_periods = np.arange(1, maturity + 1)
        # Use previously calculated spot rates for discounted cash flows
        if i == 0:
            discounted_cash_flows = cash_flows / \
                (1 + spot_rates[i])**time_periods
        else:
            discounted_cash_flows = [
                cf / (1 + spot_rates[j])**time_periods[j] for j, cf in enumerate(cash_flows)]
            discounted_cash_flows = np.sum(discounted_cash_flows)

        residual = price - discounted_cash_flows
        print(i, price, discounted_cash_flows)
        if residual <= 0:
            # Handle cases where residual is too low
            print(f"Warning: Residual for bond with maturity {
                  maturity} is too low. Adjusting spot rate calculation.")
            # Use previous spot rate as an approximation
            spot_rate = spot_rates[i-1]
        else:
            spot_rate = ((100 / residual)**(1 / maturity)) - 1
        spot_rates[i] = spot_rate

    return spot_rates


def bootstrap_spot_curve(bonds_df, day_idx):
    # Guess initial spot rate
    spot_rates = np.full(len(bonds_df), 0.1)
    sorted_bonds = bonds_df.sort_values(
        by='periods_to_maturity').reset_index(drop=True)

    for i, bond in sorted_bonds.iterrows():
        price = bond[date_list[day_idx]]
        coupon_rate = bond['coupon_rate']
        maturity = bond['periods_to_maturity']

        cash_flows = np.array(
            [coupon_rate] * int(maturity - 1) + [100 + coupon_rate])
        time_periods = np.arange(1, maturity + 1)
        # Use previously calculated spot rates for discounted cash flows
        if i == 0:
            discounted_cash_flows = cash_flows / \
                ((1 + spot_rates[i])**time_periods)
        else:
            discounted_cash_flows = [
                cf / ((1 + spot_rates[j])**time_periods[j]) for j, cf in enumerate(cash_flows)]
            discounted_cash_flows = np.sum(discounted_cash_flows)

        discount_factor = price / (100 + coupon_rate)
        spot_rate = ((1 / discount_factor)**(1 / maturity)) - 1
        spot_rates[i] = spot_rate

    return spot_rates


def calculate_forward_rates(spot_rates):
    forward_rates = []
    for i in range(1, 5):  # Adjust range to calculate forward rates for terms 2 to 5 years
        forward_rate = (((1 + spot_rates[i])**(i+1)) / ((1 + spot_rates[0])**1))**(1 / i) - 1
        forward_rates.append(forward_rate)
    return forward_rates


# main starts here
np.set_printoptions(precision=4, suppress=True)

# Read data from Excel file starting from cell F2
file_path = "BondData.xlsx"
date_list = [datetime.datetime(2024, 1, day) for day in range(8, 13)] + [
    datetime.datetime(2024, 1, day) for day in range(15, 20)
]
df = pd.read_excel(
    file_path,
    usecols="B:O",
    names=["coupon_rate", "ISIN",
           "issue_date", "maturity_date"] + date_list,
)


df["issue_date"] = pd.to_datetime(df["issue_date"], format="%m/%d/%Y")
df["maturity_date"] = pd.to_datetime(df["maturity_date"], format="%m/%d/%Y")


for curr_date in date_list:
    for bond_idy, clean_price in enumerate(df[curr_date]):
        days_since = days_since_coupon(
            curr_date, df["maturity_date"][bond_idy])
        dirty_price = calc_dirty_price(
            clean_price, days_since, df["coupon_rate"][bond_idy]
        )

        df.loc[bond_idy, curr_date] = dirty_price


benchmarkBondsISIN = ['CA135087J546', 'CA135087N910', 'CA135087D507', 'CA135087P246',
                      'CA135087P816', 'CA135087L930', 'CA135087M847', 'CA135087N837', 'CA135087H235', 'CA135087Q491']

mask = df['ISIN'].isin(benchmarkBondsISIN)
benchmarkBondsdf = df[mask]


benchmarkBondsdf.loc[:, 'bond_price_avg'] = benchmarkBondsdf[date_list].mean(
    axis=1)
benchmarkBondsdf.loc[:, 'coupon_rate'] = benchmarkBondsdf['coupon_rate'].apply(
    lambda x: x * 100)
benchmarkBondsdf.loc[:, 'periods_to_maturity'] = benchmarkBondsdf['maturity_date'].apply(
    lambda m_date: periods_to_maturity(date_list[-1], m_date))


fig1 = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()

yield_log_returns = []
forward_log_returns = []

# yield curve
for day_idx in range(10):
    spot_rates = bootstrap_yield_curve(benchmarkBondsdf, day_idx)
    fig1.add_trace(go.Scatter(
        x=list(range(10)), y=spot_rates * 100, mode='lines+markers', name=f'Yield curve from data from day {day_idx}'))
    if day_idx == 0:
        yield_log_return = np.zeros_like(spot_rates)  # No log-returns for the first day
    else:
        previous_spot_rates = bootstrap_yield_curve(benchmarkBondsdf, day_idx - 1)
        yield_log_return = np.log(spot_rates / previous_spot_rates)
    # Append the log-returns for the current day to the list
    yield_log_returns.append(yield_log_return)

fig1.update_layout(title='Bootstrapped Yield Curves',
                   xaxis_title='Maturity (Years)', yaxis_title='Spot Rate (%)')
fig1.show()


# spot curve
for day_idx, date in enumerate(date_list):
    spot_rates = bootstrap_spot_curve(benchmarkBondsdf, day_idx)
    fig2.add_trace(go.Scatter(
        x=list(range(1, len(benchmarkBondsdf) + 1)), y=spot_rates * 100,
        mode='lines+markers', name=f'Spot curve from data from day {day_idx}'))

fig2.update_layout(title='Bootstrapped Spot Curves',
                   xaxis_title='Maturity (Years)', yaxis_title='Spot Rate (%)')
fig2.show()


# forward curve
for day_idx, date_column in enumerate(date_list):
    spot_rates = bootstrap_spot_curve(benchmarkBondsdf, day_idx)
    forward_rates = calculate_forward_rates(spot_rates)
    fig3.add_trace(go.Scatter(
        x=[f"1yr-{k}yr" for k in range(2, 6)], y=np.array(forward_rates) * 100,  # Adjust x-axis values for terms 2 to 5 years
        mode='lines+markers', name=f'Forward curve from data from day {day_idx}'))
    if day_idx == 0:
        forward_log_return = np.zeros_like(forward_rates)  # No log-returns for the first day
    else:
        previous_spot_rates = bootstrap_spot_curve(benchmarkBondsdf, day_idx - 1)
        previous_forward_rates = calculate_forward_rates(previous_spot_rates)
        forward_log_return = np.log(np.array(forward_rates) / np.array(previous_forward_rates))
    # Append the log-returns for the current day to the list
    forward_log_returns.append(forward_log_return)

fig3.update_layout(title='1-Year Forward Curve with Terms Ranging from 2 to 5 Years',
                   xaxis_title='Forward Rate Times (years)', yaxis_title='Forward Rate (%)')

fig3.show()

# Calculate covariance matrices
yield_covariance_matrix = np.cov(yield_log_returns)
forward_covariance_matrix = np.cov(forward_log_returns)


# Calculate eigenvalues and eigenvectors
yield_eigenvalues, yield_eigenvectors = np.linalg.eig(yield_covariance_matrix)
forward_eigenvalues, forward_eigenvectors = np.linalg.eig(forward_covariance_matrix)


print("Yield evalues: ", yield_eigenvalues)
print("Yield evectors: ", yield_eigenvectors)

print("Yield evalues: ", forward_eigenvalues)
print("Yield evectors: ", forward_eigenvectors)

def numpy_to_latex(matrix):
    latex_str = "\\begin{bmatrix}\n"
    for row in matrix:
        latex_str += " & ".join([f"{elem:.2g}" for elem in row])
        latex_str += " \\\\\n"
    latex_str += "\\end{bmatrix}"
    return latex_str

print(numpy_to_latex(yield_covariance_matrix))
print(numpy_to_latex(forward_covariance_matrix))

