import pandas as pd


def ChansGame(data, freq_table, elem):
    all_Yes = sum(freq_table["Yes"])
    all_No = sum(freq_table["No"])
    S_YaN = freq_table['Yes'][elem] + freq_table['No'][elem]

    all_Elem = len(data)
    PElem = round(S_YaN / all_Elem, 8)
    PYesAElem = round(freq_table['Yes'][elem] / all_Yes, 8)
    PYes = round(all_Yes / all_Elem, 8)
    P_Y_s = round((PYesAElem * PYes) / PElem, 8)

    PNoAElem = round(freq_table['No'][elem] / all_No, 8)
    PNo = round(all_No / all_Elem, 8)
    P_N_s = round((PNoAElem * PNo) / PElem, 8)

    print(f"{elem} --- Yes = {round(P_Y_s, 3)}, No = {round(P_N_s, 3)}")
    return P_Y_s, P_N_s, PYes, PNo


data = pd.read_csv('data.csv')

freq_table_outlook = pd.crosstab(data['Outlook'], data['Play'])
freq_table_humidity = pd.crosstab(data['Humidity'], data['Play'])
freq_table_wind = pd.crosstab(data['Wind'], data['Play'])

P_Y_s1, P_N_s1, PYes1, PNo1 = ChansGame(data, freq_table_outlook, "Overcast")
P_Y_s2, P_N_s2, PYes2, PNo2 = ChansGame(data, freq_table_humidity, "High")
P_Y_s3, P_N_s3, PYes3, PNo3 = ChansGame(data, freq_table_wind, "Strong")

Probability_yes = round(P_Y_s1 * P_Y_s2 * P_Y_s3 * PYes1 * PYes2 * PYes3, 8)
Probability_no = round(P_N_s1 * P_N_s2 * P_N_s3 * PNo1 * PNo2 * PNo3, 8)

print(f"P(Yes) = {round(Probability_yes / (Probability_yes + Probability_no), 3)}")
print(f"P(No) = {round(Probability_no / (Probability_yes + Probability_no), 3)}")
