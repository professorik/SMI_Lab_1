import pandas as pd
import numpy as np

O = np.array([3, 2, 5, 4, 4, 2, 5, 1, 3, 1, 3, 3, 5, 3, 3, 1, 3, 3, 2, 2, 2, 4, 3, 4, 4, 3, 2])
A = np.array([[0.2802, 0.2252, 0.0129, 0.3000, 0.1817],
              [0.2077, 0.3196, 0.0186, 0.2229, 0.2312],
              [0.3087, 0.1740, 0.2888, 0.0009, 0.2276],
              [0.1945, 0.0825, 0.2018, 0.2561, 0.2651],
              [0.1680, 0.1105, 0.2879, 0.2590, 0.1746]])
B = np.array([[0.1170, 0.1392, 0.1301, 0.2870, 0.3267],
              [0.4795, 0.2471, 0.1224, 0.0791, 0.0719],
              [0.0310, 0.3324, 0.1466, 0.2669, 0.2232],
              [0.2555, 0.1537, 0.3016, 0.0385, 0.2507],
              [0.1022, 0.1260, 0.1700, 0.4227, 0.1792]])
P = np.array([0.3130, 0.1523, 0.3038, 0.1346, 0.0963])

if __name__ == '__main__':
    N = len(A)
    M = len(B[0])
    T = len(O)
    delta = np.zeros((N, T))
    psi = np.zeros((N, T), dtype=int)
    delta[:, 0] = P.T * B[:, O[0] - 1]
    for t in range(1, T):
        for j in range(N):
            tp = delta[:, t - 1] * A[:, j]
            psi[j, t] = tp.argmax() + 1
            delta[j, t] = tp[psi[j, t] - 1] * B[j, O[t] - 1]

    df = pd.DataFrame(delta)
    for i in range(0, T - 10, 10):
        print(df.iloc[:, i: i + 10].to_string(index=False))
    print(df.iloc[:, -(T % 10):].to_string(index=False))
    print(pd.DataFrame(psi).to_string(header=False, index=False))

    Q = np.zeros(T, dtype=int)
    Q[T - 1] = delta[:, T - 1].argmax() + 1
    answer = delta[Q[T - 1] - 1, T - 1]
    for t in range(T - 1, 0, -1):
        Q[t - 1] = psi[Q[t] - 1, t]
    print("P* =", answer)
    print("Q =", Q)

    answer_2 = P[Q[0] - 1] * B[Q[0] - 1, O[0] - 1]
    for i in range(1, T):
        answer_2 *= A[Q[i - 1] - 1, Q[i] - 1] * B[Q[i] - 1, O[i] - 1]
    print("Pr(O, Q|lambda) =", answer)

    print("Pr(O, Q|lambda) = P* is", abs(answer - answer_2) < 1e-7)
