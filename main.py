import pandas as pd
from compute_form_features import compute_form, Types


def main():
    df = pd.read_csv(r"D:\PycharmProjects\Bundesliga\Datasets\bundesliga_2024.csv")
    #print(df.head())

    form_df = compute_form(df, 5, 12, 91, Types.BOTH)
    print(form_df)




if __name__ == "__main__":
    main()
