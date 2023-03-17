# Todo


- 参考他的参数放置顺序 https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.cu.html
- batch broadcast move inside *_matrix funcion

- 门支持三种类型的
```
Displacement(r=tensor(1.0), phi=tensor(0.0))
Displacement(r=Parameter, phi=Parameter)
Displacement(r=Parameter, phi=tensor(0.0))
```

- 训练好的模型 要支持 不同batch_size的推理

- add `quad_expectation`, `homodyne_measure` to fock.ops
