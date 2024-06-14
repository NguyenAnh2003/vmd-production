from vmd.metric.metric import Correct_Rate, Accuracy, Align, ins_del_sub
from vmd.metric.mdd_result import md_d_result_analyst

def md_d_result(predict_info, out_file):
    list_predict = [[str(element) for element in elements] for elements in predict_info["predict"]]
    list_canonical = [[str(element) for element in elements] for elements in predict_info["canonical"]]
    list_transcript = [[str(element) for element in elements] for elements in predict_info["transcript"]]
    
    data = {key: {'canonical': can, 'transcript': trans, 'predict': pred} for key, can, trans, pred in
                  zip(predict_info["id"], list_canonical, list_transcript, list_predict)}
    
    total_correct_rate = 0
    total_accuracy = 0
    total_len = 0

    canon_pred = []
    canon_trans = []
    trans_pred = []

    for data_id in data:
        pred = data[data_id]['predict']
        trans = data[data_id]['transcript']
        canon = data[data_id]['canonical']

        canon_pred += _get_align(data_id, canon, pred)
        trans_pred += _get_align(data_id, trans, pred)
        canon_trans += _get_align(data_id, canon, trans)
        
        correct_rate, len_ = Correct_Rate(trans, pred)
        acc, len_ = Accuracy(trans, pred)

        total_correct_rate += correct_rate
        total_accuracy += acc
        total_len += len_

    corr_rate = (total_len - total_correct_rate) / total_len
    acc = (total_len - total_accuracy) / total_len

    corr_rate = round(corr_rate, 4)
    acc = round(acc, 4)
    print('*' * 2 + f" MD&D Result" + '*' * 2, file=out_file)
    print('Correct Rate:', corr_rate, file=out_file)
    print("Accuracy:", acc, file=out_file)

    recall, precision, f1_score = md_d_result_analyst(canon_trans, canon_pred, trans_pred, out_file)
    return corr_rate, acc, recall, precision, f1_score


def _get_op(seq1, seq2):
    op = []
    for i in range(len(seq1)):
        if seq1[i] != "<eps>" and seq2[i] == "<eps>":
            op.append('D')
        elif seq1[i] == "<eps>" and seq2[i] != "<eps>":
            op.append('I')
        elif (seq1[i] != seq2[i]) and seq2[i] != "<eps>" and seq1[i] != "<eps>":
            op.append("S")
        else:
            op.append("C")
    return op


def _get_align(k, s1, s2):
    a1, a2 = Align(s1, s2)
    a3 = []
    I, D, S = ins_del_sub(a1, a2)

    C = len(a1) - I - D - S

    return [
        k + ' ref ' + ' '.join(a1),
        k + ' hyp ' + ' '.join(a2),
        k + ' op ' + ' '.join(_get_op(a1, a2)),
        k + f' #csid {C} {S} {I} {D}'
    ]