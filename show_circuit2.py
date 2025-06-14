import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 이전 함수 정의 (swap_test_measurement만 수정)
# ==========================================

L = 1
USE_PARALLEL_EMBEDDING = 2

# enhanced_qvae_layer 함수는 이전과 동일
def enhanced_qvae_layer(inputs, weights, layer_idx, n_layers, n_qubits, reupload=True, alternate_embedding=False):
    if not reupload or layer_idx == 0:
        for i, feature in enumerate(inputs):
            for p in range(USE_PARALLEL_EMBEDDING):
                qubit_idx = i * USE_PARALLEL_EMBEDDING + p
                if qubit_idx < n_qubits:
                    if alternate_embedding and (i + p) % 2 == 1:
                        qml.RX(feature, wires=qubit_idx)
                    else:
                        qml.RY(feature, wires=qubit_idx)
    for w in range(n_qubits):
        qml.RY(weights[w, 0], wires=w)
        qml.RZ(weights[w, 1], wires=w)
    if n_qubits > 1:
        for w in range(n_qubits):
            qml.CNOT(wires=[w, (w + 1) % n_qubits])
    if reupload and layer_idx < n_layers - 1:
        for i, feature in enumerate(inputs):
            for p in range(USE_PARALLEL_EMBEDDING):
                qubit_idx = i * USE_PARALLEL_EMBEDDING + p
                if qubit_idx < n_qubits:
                    if alternate_embedding and (i + p) % 2 == 1:
                        qml.RX(feature, wires=qubit_idx)
                    else:
                        qml.RY(feature, wires=qubit_idx)

def swap_test_measurement_modified(n_data_qubits, n_ref_qubits, total_qubits, n_trash):
    """
    [수정된 버전] 별도의 trash와 ref 큐비트를 사용하도록 변경
    """
    control_qubit = total_qubits - 1
    qml.Hadamard(wires=control_qubit)
    
    # trash와 ref 큐비트의 시작 위치를 명시적으로 분리
    trash_start = n_data_qubits
    ref_start = n_data_qubits + n_trash
    
    for i in range(n_trash):
        trash_qubit = trash_start + i
        ref_qubit = ref_start + i
        qml.CSWAP(wires=[control_qubit, trash_qubit, ref_qubit])
        
    qml.Hadamard(wires=control_qubit)
    return qml.expval(qml.PauliZ(control_qubit))

# 메인 회로 함수 (수정된 swap_test를 호출하도록 변경)
def enhanced_qvae_circuit_modified(x, weights, n_qubits, total_qubits):
    USE_DATA_REUPLOADING = True
    USE_ALTERNATE_EMBEDDING = True
    USE_SWAP_TEST = True
    N_REFERENCE_QUBITS = 2
    N_TRASH_QUBITS = 2
    for l in range(L):
        enhanced_qvae_layer(
            inputs=x,
            weights=weights[l],
            layer_idx=l,
            n_layers=L,
            n_qubits=n_qubits,
            reupload=USE_DATA_REUPLOADING,
            alternate_embedding=USE_ALTERNATE_EMBEDDING
        )
    return swap_test_measurement_modified(n_qubits, N_REFERENCE_QUBITS, total_qubits, N_TRASH_QUBITS)

# ==========================================
# 2. 13 큐비트 시각화 실행
# ==========================================

print("별도의 Trash 큐비트를 사용하는 13 큐비트 회로도를 생성합니다...")
# 13 큐비트 설정
n_data = 8
n_trash = 2
n_ref = 2
n_control = 1
total_qubits_13 = n_data + n_trash + n_ref + n_control # 8+2+2+1 = 13 (오류: n_control은 이미 포함됨, 8+2+2=12, 총 13이 되려면 n_control이 마지막이므로 12까지 있어야 함)
# 정정: 8 data, 2 trash, 2 ref, 1 control -> 총 13개의 와이어가 필요 (0-12)
total_qubits_13 = 13

dev = qml.device("default.qubit", wires=total_qubits_13)

@qml.qnode(dev)
def visualize_13_qubit_circuit():
    x_features = np.random.uniform(0, np.pi, 4)
    weights = np.random.uniform(0, 2 * np.pi, (L, n_data, 2))
    
    # 큐비트 0-7에 데이터 처리 레이어 적용
    enhanced_qvae_circuit_modified(x_features, weights, n_data, total_qubits_13)
    return qml.state()

fig, ax = qml.draw_mpl(visualize_13_qubit_circuit)()
ax.set_title("Conceptual 13-Qubit Circuit with Separate Trash Qubits", fontsize=16)
plt.show()