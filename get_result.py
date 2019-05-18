import os
import pandas as pd


if __name__ == '__main__':
    path = "tmp/emotion_out/"
    pd_all = pd.read_csv(os.path.join(path, "test_results.tsv") ,sep='\t',header=None)

    data = pd.DataFrame(columns=['polarity'])
    print(pd_all.shape)

    for index in pd_all.index:
        neutral_score = pd_all.loc[index].values[0]
        positive_score = pd_all.loc[index].values[1]
        negative_score = pd_all.loc[index].values[2]

        if max(neutral_score, positive_score, negative_score) == neutral_score:
            # data.append(pd.DataFrame([index, "neutral"],columns=['id','polarity']),ignore_index=True)
            data.loc[index+1] = ["neutral"]
        elif max(neutral_score, positive_score, negative_score) == positive_score:
            #data.append(pd.DataFrame([index, "positive"],columns=['id','polarity']),ignore_index=True)
            data.loc[index+1] = [ "positive"]
        else:
            #data.append(pd.DataFrame([index, "negative"],columns=['id','polarity']),ignore_index=True)
            data.loc[index+1] = [ "negative"]
        #print(negative_score, positive_score, negative_score)

    data.to_csv(os.path.join(path, "pre_sample.tsv"),sep = ',')
    #print(data)