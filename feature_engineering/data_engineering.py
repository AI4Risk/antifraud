import pandas as pd
import numpy as np
from math import isnan
import multiprocessing as mp
import sys
from tqdm import tqdm


def data_engineer_example(data_dir):
    data = pd.read_csv(data_dir)
    data['time_stamp'] = pd.to_datetime(data['time_stamp'])

    time_span = []
    # 1sec, 1 min, 1 day ...
    for i in [60, 3600, 86400, 172800, 2628000, 7884000, 15768000, 31536000]:
        time_span.append(pd.Timedelta(seconds=i))

    train = []
    Oct = []
    Nov = []
    Dec = []

    start_time = "2015/1/1 00:00"

    for i in data.iterrows():
        data2 = []
        temp_data = data[data['card_id'] == i[1]['card_id']]
        temp_county_id = i[1]['loc_cty']
        temp_merch_id = i[1]['loc_merch']
        temp_time = i[1]['time_stamp']
        temp_label = i[1]['is_fraud']
        a_grant = i[1]['amt_grant']
        a_purch = i[1]['amt_purch']
        for loc in data['loc_cty'].unique():
            data1 = []
            if (loc in temp_data['loc_cty'].unique()):
                card_tuple = temp_data['loc_cty'] == loc
                single_loc_card_data = temp_data[card_tuple]
                time_list = single_loc_card_data['time_stamp']
                for length in time_span:
                    lowbound = (time_list >= (temp_time - length))
                    upbound = (time_list <= temp_time)
                    correct_data = single_loc_card_data[lowbound & upbound]
                    Avg_grt_amt = correct_data['amt_grant'].mean()
                    Totl_grt_amt = correct_data['amt_grant'].sum()
                    Avg_pur_amt = correct_data['amt_purch'].mean()
                    Totl_pur_amt = correct_data['amt_purch'].sum()
                    Num = correct_data['amt_grant'].count()
                    if (isnan(Avg_grt_amt)):
                        Avg_grt_amt = 0
                    if (isnan(Avg_pur_amt)):
                        Avg_pur_amt = 0
                    data1.append([a_grant, Avg_grt_amt, Totl_grt_amt,
                                 a_purch, Avg_pur_amt, Totl_pur_amt, Num])
            else:
                for length in time_span:
                    data1.append([0, 0, 0, 0, 0, 0, 0])
            data2.append(data1)
        if (temp_time > pd.to_datetime(start_time)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=9 * 2628000)):
                train.append([temp_label, np.array(data2)])
        if (temp_time > pd.to_datetime(start_time) + pd.Timedelta(seconds=9 * 2628000)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=10 * 2628000)):
                Oct.append([temp_label, np.array(data2)])
        if (temp_time > pd.to_datetime(start_time) + pd.Timedelta(seconds=10 * 2628000)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=11 * 2628000)):
                Nov.append([temp_label, np.array(data2)])
        if (temp_time > pd.to_datetime(start_time) + pd.Timedelta(seconds=11 * 2628000)):
            if (temp_time <= pd.to_datetime(start_time) + pd.Timedelta(seconds=12 * 2628000)):
                Dec.append([temp_label, np.array(data2)])
    np.save(file='train', arr=train)
    np.save(file='Oct', arr=Oct)
    np.save(file='Nov', arr=Nov)
    np.save(file='Dec', arr=Dec)
    return 0


def featmap_gen(tmp_card, tmp_df=None):
    # time_span = [20, 60, 120, 300, 600, 1500, 3600, 10800, 32400, 64800, 129600, 259200]
    time_span = [5, 20]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    for trans_idx, trans_feat in tmp_df.iterrows():
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            new_df['trans_at_avg_{}'.format(
                tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(
                tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(
                tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(
                tname)] = temp_amt - correct_data['Amount'].mean()
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(
                correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(
                correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(
                correct_data.Type.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)


def data_engineer_benchmark(feat_df):
    pool = mp.Pool(processes=4)
    args_all = [(card_n, card_df)
                for card_n, card_df in feat_df.groupby("Source")]
    jobs = [pool.apply_async(featmap_gen, args=args) for args in args_all]
    # post_fe_df = [job.get() for job in jobs]
    post_fe_df = []
    num_job = len(jobs)
    for i, job in enumerate(jobs):
        post_fe_df.append(job.get())
        sys.stdout.flush()
        sys.stdout.write("FE: {}/{}\r".format(i+1, num_job))
        sys.stdout.flush()
    post_fe_df = pd.concat(post_fe_df)
    post_fe_df = post_fe_df.fillna(0.)
    return post_fe_df


def calcu_trading_entropy(
    data_2: pd.DataFrame
) -> float:
    """calculate trading entropy of given data
    Args:
        data (pd.DataFrame): 2 cols, Amount and Type
    Returns:
        float: entropy
    """
    # if empty
    if len(data_2) == 0:
        return 0

    amounts = np.array([data_2[data_2['Type'] == type]['Amount'].sum()
                       for type in data_2['Type'].unique()])
    proportions = amounts / amounts.sum() if amounts.sum() else np.ones_like(amounts)
    ent = -np.array([proportion*np.log(1e-5 + proportion)
                    for proportion in proportions]).sum()
    return ent


def span_data_2d(
        data: pd.DataFrame,
        time_windows: list = [1, 3, 5, 10, 20, 50, 100, 500]
) -> np.ndarray:
    """transform transaction record into feature matrices

    Args:
        df (pd.DataFrame): transaction records
        time_windows (list): feature generating time length

    Returns:
        np.ndarray: (sample_num, |time_windows|, feat_num) transaction feature matrices
    """
    data = data[data['Labels'] != 2]
    # data = data[data['Amount'] != 0]

    nume_feature_ret, label_ret = [], []
    for row_idx in tqdm(range(len(data))):
        record = data.iloc[row_idx]
        acct_no = record['Source']
        feature_of_one_record = []

        for time_span in time_windows:
            feature_of_one_timestamp = []
            prev_records = data.iloc[(row_idx - time_span):row_idx, :]
            prev_and_now_records = data.iloc[(
                row_idx - time_span):row_idx + 1, :]
            prev_records = prev_records[prev_records['Source'] == acct_no]

            # AvgAmountT
            feature_of_one_timestamp.append(
                prev_records['Amount'].sum() / time_span)
            # TotalAmountTs
            feature_of_one_timestamp.append(prev_records['Amount'].sum())
            # BiasAmountT
            feature_of_one_timestamp.append(
                record['Amount'] - feature_of_one_timestamp[0])
            # NumberT
            feature_of_one_timestamp.append(len(prev_records))
            # MostCountryT/MostTerminalT -> no data for these items

            # MostMerchantT
            # feature_of_one_timestamp.append(prev_records['Type'].mode()[0] if len(prev_records) != 0 else 0)

            # TradingEntropyT ->  TradingEntropyT = EntT − NewEntT
            old_ent = calcu_trading_entropy(prev_records[['Amount', 'Type']])
            new_ent = calcu_trading_entropy(
                prev_and_now_records[['Amount', 'Type']])
            feature_of_one_timestamp.append(old_ent - new_ent)

            feature_of_one_record.append(feature_of_one_timestamp)

        nume_feature_ret.append(feature_of_one_record)
        label_ret.append(record['Labels'])

    nume_feature_ret = np.array(nume_feature_ret).transpose(0, 2, 1)

    # sanity check
    assert nume_feature_ret.shape == (
        len(data), 5, len(time_windows)), "output shape invalid."

    return nume_feature_ret.astype(np.float32), np.array(label_ret).astype(np.int64)
