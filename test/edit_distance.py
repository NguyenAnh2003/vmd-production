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
        if predict[i] == 0 or predict[i] == 1:
            continue
        if i == 0:
            pred.append(predict[i])
        if i > 0 and predict[i] != predict[i-1]:
            if len(pred) == 0:
                pred.append(predict[i])
            elif predict[i] != pred[-1]:
                pred.append(predict[i])
    return pred

# def process(preds: list, labels: list):
#     # costs will holds the costs, like in the Levenshtein distance algorithm
#     costs = [[0 for inner in range(len(preds) + 1)] for outer in range(len(labels) + 1)]
#     # backtrace will hold the operations we've done.
#     # so we could later backtrace, like the WER algorithm requires us to.
#     backtrace = [[0 for inner in range(len(preds) + 1)] for outer in range(len(labels) + 1)]
#     OP_OK = 0
#     OP_SUB = 1
#     OP_INS = 2
#     OP_DEL = 3
#     DEL_PENALTY = 1
#     INS_PENALTY = 1
#     SUB_PENALTY = 1
#
#     # First column represents the case where we achieve zero
#     # hypothesis words by deleting all reference words.
#     for i in range(1, len(labels) + 1):
#         costs[i][0] = DEL_PENALTY * i
#         backtrace[i][0] = OP_DEL
#
#     # First row represents the case where we achieve the hypothesis
#     # by inserting all hypothesis words into a zero-length reference.
#     for j in range(1, len(preds) + 1):
#         costs[0][j] = INS_PENALTY * j
#         backtrace[0][j] = OP_INS
#
#     # computation
#     for i in range(1, len(labels) + 1):
#         for j in range(1, len(preds) + 1):
#             if labels[i - 1] == preds[j - 1]:
#                 costs[i][j] = costs[i - 1][j - 1]
#                 backtrace[i][j] = OP_OK
#             else:
#                 substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY
#                 insertionCost = costs[i][j - 1] + INS_PENALTY
#                 deletionCost = costs[i - 1][j] + DEL_PENALTY
#
#                 costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
#                 if costs[i][j] == substitutionCost:
#                     backtrace[i][j] = OP_SUB
#                 elif costs[i][j] == insertionCost:
#                     backtrace[i][j] = OP_INS
#                 else:
#                     backtrace[i][j] = OP_DEL
#
#     # back trace though the best route:
#     i = len(labels)
#     j = len(preds)
#     numSub = 0
#     numDel = 0
#     numIns = 0
#     numCor = 0
#     while i > 0 or j > 0:
#         if backtrace[i][j] == OP_OK:
#             numCor += 1
#             i -= 1
#             j -= 1
#         elif backtrace[i][j] == OP_SUB:
#             numSub += 1
#             i -= 1
#             j -= 1
#         elif backtrace[i][j] == OP_INS:
#             numIns += 1
#             j -= 1
#         elif backtrace[i][j] == OP_DEL:
#             numDel += 1
#             i -= 1
#     # wer_result = round((numSub + numDel + numIns) / (float)(len(labels)), 3)
#     results = {'numCor': numCor, 'numSub': numSub, 'numIns': numIns, 'numDel': numDel,
#                "numCount": len(labels)}
#     return results
