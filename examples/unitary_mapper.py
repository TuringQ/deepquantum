# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dqp
#     language: python
#     name: dqp
# ---

# %%
import deepquantum as dq
import numpy as np

# %% [markdown]
# # quantum u gate map

# %% [markdown]
#  Use `UnitaryMapper` for mapping the quantum gate to photonic quantum circuit. \
#  nmode: 2*n_qubits + 2 \
#  auxiliary modes: [0,0] or [1,0] in the last 2modes \
# succcess probability: preferred 1/3 for 2qubtis, 1/4 for 3 qubits

# %% [markdown]
# ## map the quantum gate

# %%
swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

# %%
nqubit = 2
nmode = 6
ugate = cnot
aux = [0, 0]
aux_pos = [4, 5]
success = 1 / 3
umap = dq.UnitaryMapper(nqubit=nqubit, nmode=nmode, ugate=ugate, success=success, aux=aux, aux_pos=aux_pos)

# %%
umap.ugate = cnot
umap.success = 1 / 3
re = umap.solve_eqs_real(total_trials=1, trials=10, precision=1e-5)  # for real solution

# %%
## check result
result = re[0][0][0]
transfer_mat = umap.get_transfer_mat(result)
umap.plot_u(transfer_mat, vmin=-1)
# np.save("cnot_test.npy", re) # save the first result

# %% [markdown]
# ## check results

# %%
cnot_test = result
init_state = [1, 0, 1, 0, 0, 0]
test_circuit = dq.QumodeCircuit(nmode=6, init_state=init_state, basis=True)
test_circuit.any(cnot_test, list(range(6)))
test_circuit.draw()

# %%
re = test_circuit(state=[1, 0, 0, 1, 0, 0])

# %%
print(re)

# %% [markdown]
# # decompose clements

# %% [markdown]
#  decomposing the optical qunatum circuit(unitary matrix) to clements structure

# %% [markdown]
# ## 6x6 case

# %%
u6x6 = np.array(
    [
        [1, 0, 1, -1, 0, 0],
        [0, 1, 0, 0, 0, np.sqrt(2)],
        [1, 0, 0, 1, 1, 0],
        [-1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, -1, 0],
        [0, np.sqrt(2), 0, 0, 0, -1],
    ]
) / np.sqrt(3)

# %%
ud = dq.UnitaryDecomposer(u6x6)
mzi_info = ud.decomp()
mzi_info[1]

# %%
p_mzi = dq.DrawClements(6, mzi_info[0])
p_mzi.plotting_clements()

# %% [markdown]
# ## 8x8 case

# %%
u8x8 = np.eye(8, 8)
ud = dq.UnitaryDecomposer(u8x8)
mzi_info = ud.decomp()

# %%
p_mzi = dq.DrawClements(8, mzi_info[0])
p_mzi.plotting_clements()

# %%
print(p_mzi.ps_position)

# %% [markdown]
# ## Clements in ansatz

# %%
clements = dq.Clements(nmode=6, init_state=[1, 0, 1, 0, 0, 0], cutoff=3)

# %%
clements.draw()

# %%
data = clements.dict2data(mzi_info[2])  # encoding the 6x6 data
re = clements(data=data)
print(re)

# %%
clements.draw()  # 6x6 CNOT Clements structure parameters
