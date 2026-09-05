import numpy as np

def conservative_choice(q_values, gate):
    q_values = np.asarray(q_values, dtype=np.float32)
    if q_values.size <= 1:
        return 0, 0.0

    teacher_q = float(q_values[0])
    best_alt_index = int(np.argmax(q_values[1:])) + 1
    gap = float(q_values[best_alt_index]) - teacher_q

    if gap >= float(gate):
        return best_alt_index, gap
    return 0, gap
