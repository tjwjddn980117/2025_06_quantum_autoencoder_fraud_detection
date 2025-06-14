import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 이전 컨텍스트의 모든 함수 정의
# ==========================================

# 헬퍼(Helper) 함수: qae_layer, enhanced_qvae_layer, swap_test_measurement
L = 2  # 시각화를 위해 레이어 수를 2로 줄임
USE_PARALLEL_EMBEDDING = 2 # 병렬 임베딩 차원

def qae_layer(theta):
    """
    qae_layer: 파라미터화된 회전 + 순환 CNOT 얽힘
    """
    n_qubits = theta.shape[0]
    for w in range(n_qubits):
        qml.RX(theta[w, 0], wires=w)
        qml.RY(theta[w, 1], wires=w)
        qml.RZ(theta[w, 2], wires=w)
    for w in range(n_qubits):
        qml.CNOT(wires=[w, (w + 1) % n_qubits])

def enhanced_qvae_layer(inputs, weights, layer_idx, n_layers, n_qubits, reupload=True, alternate_embedding=False):
    """
    enhanced_qvae_layer: 데이터 임베딩, 회전, 얽힘, 리업로딩을 포함하는 통합 레이어
    """
    # 데이터 임베딩 (첫 레이어에 항상 적용)
    if not reupload or layer_idx == 0:
        for i, feature in enumerate(inputs):
            for p in range(USE_PARALLEL_EMBEDDING):
                qubit_idx = i * USE_PARALLEL_EMBEDDING + p
                if qubit_idx < n_qubits:
                    if alternate_embedding and (i + p) % 2 == 1:
                        qml.RX(feature, wires=qubit_idx)
                    else:
                        qml.RY(feature, wires=qubit_idx)
    
    # 파라미터화된 회전
    for w in range(n_qubits):
        qml.RY(weights[w, 0], wires=w)
        qml.RZ(weights[w, 1], wires=w)
    
    # CNOT 얽힘
    if n_qubits > 1:
        for w in range(n_qubits):
            qml.CNOT(wires=[w, (w + 1) % n_qubits])
    
    # 데이터 리업로딩 (마지막 레이어 제외)
    if reupload and layer_idx < n_layers - 1:
        for i, feature in enumerate(inputs):
            for p in range(USE_PARALLEL_EMBEDDING):
                qubit_idx = i * USE_PARALLEL_EMBEDDING + p
                if qubit_idx < n_qubits:
                    if alternate_embedding and (i + p) % 2 == 1:
                        qml.RX(feature, wires=qubit_idx)
                    else:
                        qml.RY(feature, wires=qubit_idx)

def swap_test_measurement(n_data_qubits, n_ref_qubits, total_qubits, n_trash):
    """
    SWAP 테스트: 출력 상태와 참조 상태 간의 충실도(fidelity) 측정
    """
    control_qubit = total_qubits - 1  # 마지막 큐비트를 보조(ancilla)로 사용
    qml.Hadamard(wires=control_qubit)
    
    data_start = n_data_qubits - n_trash
    ref_start = n_data_qubits
    
    for i in range(n_ref_qubits):
        data_qubit = data_start + i
        ref_qubit = ref_start + i
        qml.CSWAP(wires=[control_qubit, data_qubit, ref_qubit])
    
    qml.Hadamard(wires=control_qubit)
    return qml.expval(qml.PauliZ(control_qubit))

# 메인 회로 함수: angle_embedding_circuit, enhanced_qvae_circuit
def angle_embedding_circuit(x, weights, n_qubits):
    qml.AngleEmbedding(features=x, wires=range(min(len(x), n_qubits)), rotation="Y")
    for l in range(L):
        qae_layer(weights[l])
    return qml.expval(qml.PauliZ(n_qubits - 1))

def enhanced_qvae_circuit(x, weights, n_qubits, total_qubits):
    # 시각화를 위해 강화된 기능들을 True로 설정
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
    
    if USE_SWAP_TEST and total_qubits > n_qubits:
        return swap_test_measurement(n_qubits, N_REFERENCE_QUBITS, total_qubits, N_TRASH_QUBITS)
    else:
        return qml.expval(qml.PauliZ(n_qubits - 1))

# ==========================================
# 2. 회로 시각화 실행
# ==========================================

# --- Standard Angle Embedding Circuit 시각화 ---
print("Standard Angle Embedding Circuit 다이어그램을 생성합니다...")
n_qubits_angle = 4
dev_angle = qml.device("default.qubit", wires=n_qubits_angle)

@qml.qnode(dev_angle)
def visualize_angle_circuit():
    x = np.random.uniform(0, np.pi, n_qubits_angle)
    weights = np.random.uniform(0, 2 * np.pi, (L, n_qubits_angle, 3))
    angle_embedding_circuit(x, weights, n_qubits_angle)
    return qml.state()

fig1, ax1 = qml.draw_mpl(visualize_angle_circuit)()
ax1.set_title("Standard Angle Embedding Circuit", fontsize=16)
plt.show()


# --- Enhanced QVAE Circuit 시각화 ---
print("\nEnhanced QVAE Circuit 다이어그램을 생성합니다...")
n_qubits_enh = 4  # 데이터 큐비트
n_ref_qubits_enh = 2 # 참조 큐비트
total_qubits_enh = n_qubits_enh + n_ref_qubits_enh + 1 # 데이터 + 참조 + 보조(SWAP 테스트용)

dev_enh = qml.device("default.qubit", wires=total_qubits_enh)

@qml.qnode(dev_enh)
def visualize_enhanced_circuit():
    x = np.random.uniform(0, np.pi, n_qubits_enh)
    weights = np.random.uniform(0, 2 * np.pi, (L, n_qubits_enh, 2))
    enhanced_qvae_circuit(x, weights, n_qubits_enh, total_qubits_enh)
    return qml.state()

fig2, ax2 = qml.draw_mpl(visualize_enhanced_circuit)()
ax2.set_title("Enhanced QVAE Circuit", fontsize=16)
plt.show()
