""" edit distance processing prediction """

def process(preds: list, labels: list):
    len_preds = len(preds)
    len_labels = len(labels)
    dp = [[0 for _ in range(len_labels + 1)]
          for _ in range(len_preds + 1)]

    # Initialize by the maximum edits possible
    for i in range(len_preds + 1):
        dp[i][0] = i
    for j in range(len_labels + 1):
        dp[0][j] = j

    # Compute the DP Matrix
    for i in range(1, len_preds + 1):
        for j in range(1, len_labels + 1):
            # If the characters are same # no changes required
            if labels[j - 1] == preds[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # Minimum of three operations possible
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],
                                   dp[i - 1][j - 1],
                                   dp[i - 1][j])
    # Change
    i = len(preds)
    j = len(labels)

    numCor = 0
    numIns = 0
    numDel = 0
    numSub = 0

    listCor = []
    listIns = []
    listDel = []
    listSub = []

    # Check till the end
    while i > 0 and j > 0:
        # If characters are same
        if preds[i - 1] == labels[j - 1]:
            numCor += 1
            listCor.append(preds[i - 1])
            i -= 1
            j -= 1

        # Replace
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            numSub += 1
            listSub.append((preds[i - 1], labels[j - 1]))
            j -= 1
            i -= 1

        # Insert
        elif dp[i][j] == dp[i - 1][j] + 1:
            numIns += 1
            listIns.append(preds[i - 1])
            i -= 1

        # Delete
        elif dp[i][j] == dp[i][j - 1] + 1:
            numDel += 1
            listDel.append(labels[j - 1])
            j -= 1

    results = {'numCor': numCor, 'numSub': numSub, 'numIns': numIns,
               'numDel': numDel, 'numCount': len(labels), 'listCor': listCor,
               'listSub': listSub, 'listIns': listIns, 'listDel': listDel}
    return results

def preprocess_predict(predict, input_size):
    pred = []
    for i in range(len(predict[:input_size])):
        if predict[i] == 0:# or predict[i] == 1:
            continue
        if i == 0:
            pred.append(predict[i])
        if i > 0 and predict[i] != predict[i-1]:
            if len(pred) == 0:
                pred.append(predict[i])
            elif predict[i] != pred[-1]:
                pred.append(predict[i])
    return pred