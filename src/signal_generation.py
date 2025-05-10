import pandas as pd

def generate_signals(df, hmm_results, favorable_states):
    df["HMM"] = hmm_results
    df["MA_Signal"] = (df["MA_9"] > df["MA_21"]).astype(int)
    df["HMM_Signal"] = df["HMM"].apply(lambda x: 1 if x in favorable_states else 0)
    df["Main_Signal"] = ((df["MA_Signal"] == 1) & (df["HMM_Signal"] == 1)).astype(int).shift(1)
    return df
